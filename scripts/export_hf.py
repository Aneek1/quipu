"""Build the Hugging Face upload folder for quipu-114m or quipu-moe, and prove it loads.

    python -m uv run python scripts/export_hf.py [--out hf_export]                  # quipu-114m
    python -m uv run python scripts/export_hf.py --config results/moe/run_config.toml \
        --ckpt-dir checkpoints-moe --out hf_export_moe [--int4] [--chat] [--card-only]

quipu-114m (model.kind "dense"), unchanged: config.json, model.safetensors (fp32,
final step), milestones/*.safetensors (bf16, as trained), modeling_quipu.py and
README.md, then a logit-parity check through the standalone loader on real
validation tokens.

quipu-moe (model.kind "moe"), released as AneekC/quipu-moe-1B-A149M (base) and
AneekC/quipu-moe-1B-A149M-chat (--chat, the M12 fine-tune):
- config.json (every ModelConfig field + total / active parameters), model.safetensors
  (fp32, final), milestones/*.safetensors (bf16, as trained);
- tokenizer.json, tokenizer_config.json, special_tokens_map.json, generation_config.json
  (the chat model's tokenizer_config carries the chat template, CHAT_TEMPLATE, and
  stops at <|end|>);
- modeling_quipu_moe.py (standalone: torch, safetensors, tokenizers; no quipu import);
- parity: the standalone model's logits must equal the training model's (loop
  dispatch) within --parity-tol on a fixed input (the first validation tokens, or a
  seeded random input without shards), for the final weights and the last milestone;
- --int4: model-int4.safetensors (quipu/int4.py: group-wise weight-only int4; why not
  torchao is explained there), its own parity check (standalone dequant == training
  model with the dequantized weights), and the measured quality loss: bits per byte
  per evaluation split, fp32 vs int4, on the first --bpb-batches batches of each, file
  sizes, and optionally HumanEval greedy pass@1 on the first --int4-code-eval problems
  -> int4_report.json (in --out and --results-dir);
- README.md from hf/README_moe.md (quipu/model_card.py): every field from a result
  file, "not yet measured" where the file is missing. --card-only rewrites just it.

lm_head.weight is left out of every weights file because it is the embedding matrix
(tied); safetensors refuses to store two names for one storage anyway.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quipu.config import load_config  # noqa: E402
from quipu.data import read_shard  # noqa: E402
from quipu.model import Quipu  # noqa: E402

TIED = "lm_head.weight"
# Spec 13's chat format as a Hugging Face chat template (jinja).
CHAT_TEMPLATE = ("{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}"
                 "<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}")


def _strip_tied(state: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {k: v.detach().to(dtype).contiguous() for k, v in state.items() if k != TIED}


def _import_standalone(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up while building the class
    spec.loader.exec_module(mod)
    return mod


# ---- quipu-114m (dense), unchanged -------------------------------------------------------

def export_dense(cfg, ckpt_dir: Path, out: Path) -> int:
    (out / "milestones").mkdir(parents=True, exist_ok=True)

    pointer = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=True)
    final = torch.load(ckpt_dir / pointer["file"], map_location="cpu", weights_only=False)
    final_step = int(final["step"])
    print(f"final checkpoint: {pointer['file']} (step {final_step})")

    config = dataclasses.asdict(cfg.model) | {
        "architecture": "quipu",
        "tokenizer": "tiktoken:gpt2",
        "tie_word_embeddings": True,
        "trained_steps": final_step,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    save_file(_strip_tied(final["model"], torch.float32), str(out / "model.safetensors"),
              metadata={"step": str(final_step), "dtype": "float32"})

    for p in sorted((ckpt_dir / "milestones").glob("step_*.pt")):
        state = torch.load(p, map_location="cpu", weights_only=True)
        save_file(_strip_tied(state, torch.bfloat16), str(out / "milestones" / f"{p.stem}.safetensors"),
                  metadata={"step": str(int(p.stem.split("_")[1])), "dtype": "bfloat16"})
        print(f"milestone {p.stem}")

    for name in ("modeling_quipu.py", "README.md"):
        shutil.copy2(REPO_ROOT / "hf" / name, out / name)

    # Parity: the standalone loader must reproduce the training model's logits.
    mq = _import_standalone(out / "modeling_quipu.py", "modeling_quipu")
    exported = mq.load(str(out))
    reference = Quipu(cfg.model)
    reference.load_state_dict(final["model"])
    reference.eval()

    val = sorted((REPO_ROOT / cfg.data.shard_dir / "val").glob("*.bin"))[0]
    tokens = torch.from_numpy(read_shard(val)[: 2 * 256 + 1].astype("int64"))
    x = tokens[:-1].view(2, 256)
    y = tokens[1:].view(2, 256)
    with torch.no_grad():
        a, b = reference(x), exported(x)
    diff = (a - b).abs().max().item()
    loss = torch.nn.functional.cross_entropy(b.view(-1, b.shape[-1]), y.reshape(-1)).item()
    print(f"parity: max |logit diff| = {diff:.2e}; loss on 512 val tokens = {loss:.3f}")
    if diff > 1e-4:
        print("FAIL: exported model does not match the training model", file=sys.stderr)
        return 1
    shutil.rmtree(out / "__pycache__", ignore_errors=True)
    print(f"OK: {out}")
    return 0


# ---- quipu-moe --------------------------------------------------------------------------

@dataclasses.dataclass
class MoEExport:
    """Everything export_moe needs; paths absolute or relative to the working directory."""
    cfg: object
    ckpt_dir: Path
    out: Path
    chat: bool = False
    int4: bool = False
    card_only: bool = False
    parity_tol: float = 1e-4
    bpb_batches: int = 4
    int4_code_eval: int = 0
    repo_id: str | None = None
    results_dir: Path | None = None
    card_inputs: object | None = None        # quipu.model_card.CardInputs
    log: object = print


def parity_input(cfg, n_rows: int = 2, width: int = 256) -> torch.Tensor:
    """A fixed input: the first validation tokens if the shard set is there, else a
    seeded random one."""
    from quipu import evalsets

    width = min(width, cfg.model.context)
    split = evalsets.eval_splits(cfg.data.shard_dir).get(evalsets.ENGLISH)
    if split is not None:
        t = evalsets.read_split(split, n_rows * width)
        if len(t) == n_rows * width:
            return torch.from_numpy(t).view(n_rows, width)
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, cfg.model.vocab_size, (n_rows, width), generator=g)


def _logits(model, x: torch.Tensor) -> torch.Tensor:
    from quipu.eval import loop_dispatch
    with torch.no_grad(), loop_dispatch(model):
        return model(x).float()


def tokenizer_files(out: Path, tokenizer_path: Path, context: int, chat: bool) -> None:
    from quipu.bpe import SPECIAL_TOKENS, BPETokenizer

    tok = BPETokenizer(tokenizer_path)
    shutil.copy2(tokenizer_path, out / "tokenizer.json")
    eos = "<|end|>" if chat else "<|endoftext|>"
    tcfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": context,
        "bos_token": None, "eos_token": eos, "pad_token": "<|endoftext|>", "unk_token": None,
        "clean_up_tokenization_spaces": False,
        "additional_special_tokens": list(SPECIAL_TOKENS[1:]),
    }
    if chat:
        tcfg["chat_template"] = CHAT_TEMPLATE
    (out / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=2) + "\n", encoding="utf-8")
    (out / "special_tokens_map.json").write_text(json.dumps(
        {"eos_token": eos, "pad_token": "<|endoftext|>",
         "additional_special_tokens": list(SPECIAL_TOKENS[1:])}, indent=2) + "\n", encoding="utf-8")
    stop = [tok.special_id("<|end|>"), tok.eot] if chat else [tok.eot]
    (out / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": stop if chat else stop[0], "pad_token_id": tok.eot,
         "max_length": context}, indent=2) + "\n", encoding="utf-8")


def _final_state(ckpt_dir: Path) -> tuple[dict, int, str]:
    pointer = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=True)
    final = torch.load(ckpt_dir / pointer["file"], map_location="cpu", weights_only=False)
    return final["model"], int(final["step"]), pointer["file"]


def _int4_quality(e: MoEExport, state: dict, deq_state: dict) -> dict:
    """bpb per evaluation split, fp32 vs int4 (training model, loop dispatch), and
    optionally HumanEval greedy pass@1 on the first problems."""
    from quipu import evalsets
    from quipu.eval import nll_tokens_bytes
    from quipu.tokenizer import make_tokenizer

    cfg = e.cfg
    tok = make_tokenizer(cfg.data.tokenizer)
    lens = tok.token_byte_lengths()
    ref = evalsets.load_model(cfg.model, state=state)
    q = evalsets.load_model(cfg.model, state=deq_state)
    bpb, per_split_tokens = {}, None
    import math
    for label, split in evalsets.eval_splits(cfg.data.shard_dir).items():
        data = evalsets.split_batches(split, cfg.train.micro_batch, cfg.model.context, e.bpb_batches)
        vals = {}
        for key, m in (("fp32", ref), ("int4", q)):
            nll, n_tok, n_bytes = nll_tokens_bytes(m, data, lens, "cpu")
            vals[key] = nll / (n_bytes * math.log(2))
        bpb[label] = vals
        per_split_tokens = n_tok if per_split_tokens is None else min(per_split_tokens, n_tok)
    report = {"bpb": bpb, "bpb_tokens_per_split": per_split_tokens}
    if e.int4_code_eval:
        ce = _import_standalone(REPO_ROOT / "scripts" / "code_eval.py", "code_eval")
        tasks, _ = ce.load_tasks(ce.HUMANEVAL, cfg.data.humaneval_revision)
        tasks = tasks[: e.int4_code_eval]
        r = {k: ce.evaluate(m, tok, tasks, samples=0, log=lambda _m: None)["greedy_pass@1"]
             for k, m in (("fp32", ref), ("int4", q))}
        report["code_eval"] = {"problems": len(tasks), **r}
    return report


def write_card(e: MoEExport, total: int, active: int) -> Path:
    from quipu import model_card

    inputs = e.card_inputs or model_card.CardInputs()
    if inputs.int4_report is None and (e.out / "int4_report.json").is_file():
        inputs = dataclasses.replace(inputs, int4_report=e.out / "int4_report.json")
    fields = model_card.fields(e.cfg, inputs, variant="chat" if e.chat else "base",
                               total=total, active=active, repo_id=e.repo_id)
    template = (REPO_ROOT / "hf" / "README_moe.md").read_text(encoding="utf-8")
    path = e.out / "README.md"
    path.write_text(model_card.render(template, fields), encoding="utf-8")
    return path


def export_moe(e: MoEExport) -> int:
    from quipu import evalsets, model_card
    from quipu.fsio import write_text_atomic
    from quipu.int4 import FORMAT, GROUP_SIZE, dequantize_state, quantize_state

    log, cfg, out = e.log, e.cfg, Path(e.out)
    out.mkdir(parents=True, exist_ok=True)
    total, active = model_card.count_params(cfg.model)
    name = model_card.release_name(total, active) + ("-chat" if e.chat else "")
    if e.repo_id is None:
        e.repo_id = "AneekC/" + name
    if e.card_only:
        log(f"card: {write_card(e, total, active)}")
        return 0

    state, step, fname = _final_state(Path(e.ckpt_dir))
    log(f"final checkpoint: {fname} (step {step}); {name}: {total:,} total / {active:,} active")
    config = dataclasses.asdict(cfg.model) | {
        "architecture": "quipu-moe", "model_name": name, "tokenizer": "tokenizer.json",
        "tie_word_embeddings": True, "trained_steps": step, "total_params": total,
        "active_params": active, "chat": e.chat,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file(_strip_tied(state, torch.float32), str(out / "model.safetensors"),
              metadata={"step": str(step), "dtype": "float32"})
    milestones = sorted((Path(e.ckpt_dir) / "milestones").glob("step_*.pt"))
    if milestones:
        (out / "milestones").mkdir(exist_ok=True)
    for p in milestones:
        ms = torch.load(p, map_location="cpu", weights_only=True)
        save_file(_strip_tied(ms, torch.bfloat16), str(out / "milestones" / f"{p.stem}.safetensors"),
                  metadata={"step": str(int(p.stem.split("_")[1])), "dtype": "bfloat16"})
    tokenizer_files(out, Path(cfg.data.tokenizer), cfg.model.context, e.chat)
    shutil.copy2(REPO_ROOT / "hf" / "modeling_quipu_moe.py", out / "modeling_quipu_moe.py")

    # Parity through the standalone loader.
    mq = _import_standalone(out / "modeling_quipu_moe.py", "modeling_quipu_moe")
    x = parity_input(cfg)
    checks = [("final (fp32)", "model.safetensors", state)]
    if milestones:
        checks.append((f"milestone {milestones[-1].stem} (bf16)",
                       f"milestones/{milestones[-1].stem}.safetensors",
                       torch.load(milestones[-1], map_location="cpu", weights_only=True)))
    report: dict = {"parity": {}}
    if e.int4:
        saved, meta = quantize_state({k: v for k, v in state.items() if k != TIED}, GROUP_SIZE)
        save_file(saved, str(out / "model-int4.safetensors"),
                  metadata={"step": str(step), "format": FORMAT, "int4": json.dumps(meta)})
        deq = dequantize_state(saved, meta)
        deq[TIED] = deq["embed.weight"]
        checks.append(("int4 (dequantized)", "model-int4.safetensors", deq))
    ok = True
    for label, weights, ref_state in checks:
        ref = evalsets.load_model(cfg.model, state={k: v.float() for k, v in ref_state.items()})
        got = mq.load(str(out), weights=weights)
        diff = (_logits(ref, x) - _logits(got, x)).abs().max().item()
        report["parity"][label] = diff
        log(f"parity {label}: max |logit diff| = {diff:.2e}")
        ok &= diff <= e.parity_tol
    if not ok:
        log(f"FAIL: the standalone model does not match the training model (tol {e.parity_tol:g})")
        return 1
    if e.int4:
        full = evalsets.load_model(cfg.model, state=state)
        report["int4_vs_fp32_max_logit_diff"] = (_logits(full, x) - _logits(
            evalsets.load_model(cfg.model, state=deq), x)).abs().max().item()
        report.update({"format": FORMAT, "group_size": GROUP_SIZE,
                       "sizes": {f: (out / f).stat().st_size
                                 for f in ("model.safetensors", "model-int4.safetensors")}})
        bf16 = {k: v.to(torch.bfloat16) for k, v in state.items() if k != TIED}
        save_file(bf16, str(out / "_bf16_size_probe.safetensors"))
        report["sizes"]["model (bf16, for comparison, not shipped)"] = (
            out / "_bf16_size_probe.safetensors").stat().st_size
        (out / "_bf16_size_probe.safetensors").unlink()
        report.update(_int4_quality(e, state, deq))
        payload = json.dumps(report, indent=1)
        write_text_atomic(out / "int4_report.json", payload)
        if e.results_dir is not None:
            Path(e.results_dir).mkdir(parents=True, exist_ok=True)
            write_text_atomic(Path(e.results_dir) / f"{name}-int4.json", payload)
        for k, v in report["bpb"].items():
            log(f"int4 bpb {k}: fp32 {v['fp32']:.4f} -> int4 {v['int4']:.4f} "
                f"({v['int4'] / v['fp32'] - 1:+.2%})")
    log(f"card: {write_card(e, total, active)}")
    shutil.rmtree(out / "__pycache__", ignore_errors=True)
    log(f"OK: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/quipu-114m.toml")
    ap.add_argument("--ckpt-dir", default=None, help="default: the config's train.ckpt_dir "
                    "(quipu-114m: checkpoints)")
    ap.add_argument("--out", default="hf_export")
    moe = ap.add_argument_group("quipu-moe")
    moe.add_argument("--chat", action="store_true", help="the chat model (-chat repo, chat template)")
    moe.add_argument("--int4", action="store_true", help="also write model-int4.safetensors + report")
    moe.add_argument("--card-only", action="store_true", help="only (re)write README.md")
    moe.add_argument("--repo-id", default=None)
    moe.add_argument("--parity-tol", type=float, default=1e-4)
    moe.add_argument("--bpb-batches", type=int, default=4)
    moe.add_argument("--int4-code-eval", type=int, default=0,
                     help="HumanEval problems for the int4 greedy delta (0 = skip)")
    moe.add_argument("--results-dir", default="results/export")
    moe.add_argument("--run-dir", default="results/moe")
    moe.add_argument("--ab-dir", default="results/ab")
    moe.add_argument("--manifest", default=None, help="default: <shard_dir>/manifest.json")
    moe.add_argument("--ledger", default="results/spend.json")
    moe.add_argument("--milestones-dir", default="results/moe/milestones")
    moe.add_argument("--code-eval-dir", default="results/code_eval")
    moe.add_argument("--experts-dir", default="results/experts")
    moe.add_argument("--sft-manifest", default=None)
    moe.add_argument("--hardware", default=None, help='e.g. "1 x RTX 5090 (rented, Vast.ai)"')
    args = ap.parse_args(argv)

    cfg = load_config(REPO_ROOT / args.config)
    if cfg.model.kind != "moe":
        return export_dense(cfg, REPO_ROOT / (args.ckpt_dir or "checkpoints"), REPO_ROOT / args.out)

    from quipu.model_card import CardInputs

    def p(v):
        return None if v is None else Path(v)

    inputs = CardInputs(run_dir=p(args.run_dir), ab_dir=p(args.ab_dir),
                        manifest=p(args.manifest) or Path(cfg.data.shard_dir) / "manifest.json",
                        ledger=p(args.ledger), milestones_dir=p(args.milestones_dir),
                        code_eval_dir=p(args.code_eval_dir), experts_dir=p(args.experts_dir),
                        sft_manifest=p(args.sft_manifest), hardware=args.hardware)
    return export_moe(MoEExport(
        cfg=cfg, ckpt_dir=Path(args.ckpt_dir or cfg.train.ckpt_dir), out=Path(args.out),
        chat=args.chat, int4=args.int4, card_only=args.card_only, parity_tol=args.parity_tol,
        bpb_batches=args.bpb_batches, int4_code_eval=args.int4_code_eval, repo_id=args.repo_id,
        results_dir=p(args.results_dir), card_inputs=inputs))


if __name__ == "__main__":
    raise SystemExit(main())
