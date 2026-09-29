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
  dispatch) within --parity-tol (default 1e-5) on a fixed input (the first validation
  tokens, or a seeded random input without shards), for the final weights and the last
  milestone; the results go to parity.json (in --out, and <name>-parity.json in
  --results-dir) whether they pass or not;
- --int4: model-int4.safetensors (quipu/int4.py: group-wise weight-only int4; why not
  torchao is explained there), its own parity check (standalone dequant == training
  model with the dequantized weights), and the measured quality loss: bits per byte
  per evaluation split with the weights in fp32, rounded to bf16 (spec 12's baseline)
  and int4, fp32 arithmetic for all three, on the first --bpb-batches batches of each
  (token count per split recorded), file sizes, and optionally HumanEval greedy pass@1
  on the first --int4-code-eval problems -> int4_report.json (in --out and
  --results-dir). The quality runs on --device.
- Memory: see export_moe (peak ~3 fp32 copies of the model, ~12 GB for the 1B model);
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
    parity_tol: float = 1e-5
    bpb_batches: int = 4
    int4_code_eval: int = 0
    device: str = "cpu"                      # for the int4 quality measurement
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
    # The FILE markers ("=== FILE: ", "=== END FILE ===", ids 5 and 6) are plain text
    # everywhere (quipu.chat): drop them from added_tokens so an HF tokenizer never
    # matches them in text. They stay in model.vocab, so ids and vocab size are unchanged.
    markers = set(SPECIAL_TOKENS[5:])
    spec = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    spec["added_tokens"] = [t for t in spec.get("added_tokens", [])
                            if t["content"] not in markers]
    (out / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    chat_specials = list(SPECIAL_TOKENS[1:5])          # system, user, assistant, end
    eos = "<|end|>" if chat else "<|endoftext|>"
    tcfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": context,
        "bos_token": None, "eos_token": eos, "pad_token": "<|endoftext|>", "unk_token": None,
        "clean_up_tokenization_spaces": False,
        "additional_special_tokens": chat_specials,
    }
    if chat:
        tcfg["chat_template"] = CHAT_TEMPLATE
    (out / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=2) + "\n", encoding="utf-8")
    (out / "special_tokens_map.json").write_text(json.dumps(
        {"eos_token": eos, "pad_token": "<|endoftext|>",
         "additional_special_tokens": chat_specials}, indent=2) + "\n", encoding="utf-8")
    stop = [tok.special_id("<|end|>"), tok.eot] if chat else [tok.eot]
    (out / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": stop if chat else stop[0], "pad_token_id": tok.eot,
         "max_length": context}, indent=2) + "\n", encoding="utf-8")


def _final_state(ckpt_dir: Path) -> tuple[dict, int, str]:
    """The final checkpoint's model state_dict, memory-mapped: the optimizer states in
    the same file (up to 2x the model for AdamW) are never read into RAM."""
    pointer = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=True)
    final = torch.load(ckpt_dir / pointer["file"], map_location="cpu", weights_only=False,
                       mmap=True)
    return final["model"], int(final["step"]), pointer["file"]


def _free() -> None:
    import gc
    gc.collect()


def _int4_model(cfg, saved: dict, meta: dict):
    """The training model with the int4 file's weights, dequantized to fp32. The
    dequantized state_dict is freed once the model holds its own copy."""
    from quipu import evalsets
    from quipu.int4 import dequantize_state

    deq = dequantize_state(saved, meta)
    deq[TIED] = deq["embed.weight"]
    model = evalsets.load_model(cfg.model, state=deq)
    del deq
    _free()
    return model


def _round_to_bf16_(model) -> None:
    """Round every floating parameter and buffer to bf16 values, in place (still
    stored as fp32): the bf16 baseline without a second copy of the model."""
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            if t.is_floating_point():
                t.copy_(t.to(torch.bfloat16).to(t.dtype))


def _int4_quality(e: MoEExport, ref, q) -> dict:
    """bpb per evaluation split for the trained weights in fp32, rounded to bf16 (spec
    12's baseline), and int4 (dequantized): the training model, loop dispatch, fp32
    arithmetic for all three (amp off), so only the weights differ. Optionally
    HumanEval greedy pass@1 on the first problems. `ref` is rounded to bf16 IN PLACE
    after its fp32 numbers are taken (no third model in memory)."""
    import math

    from quipu import evalsets
    from quipu.eval import nll_tokens_bytes
    from quipu.tokenizer import make_tokenizer

    cfg, device = e.cfg, e.device
    tok = make_tokenizer(cfg.data.tokenizer)
    lens = tok.token_byte_lengths()
    ref.to(device)
    q.to(device)
    splits = {label: evalsets.split_batches(split, cfg.train.micro_batch, cfg.model.context,
                                            e.bpb_batches)
              for label, split in evalsets.eval_splits(cfg.data.shard_dir).items()}
    bpb: dict[str, dict[str, float]] = {label: {} for label in splits}
    tokens: dict[str, int] = {}
    ce = tasks = None
    code: dict[str, float] = {}
    if e.int4_code_eval:
        ce = _import_standalone(REPO_ROOT / "scripts" / "code_eval.py", "code_eval")
        tasks, _ = ce.load_tasks(ce.HUMANEVAL, cfg.data.humaneval_revision)
        tasks = tasks[: e.int4_code_eval]

    def measure(key: str, model) -> None:
        for label, data in splits.items():
            nll, n_tok, n_bytes = nll_tokens_bytes(model, data, lens, device, amp=False)
            bpb[label][key] = nll / (n_bytes * math.log(2))
            tokens[label] = n_tok
        if ce is not None:
            code[key] = ce.evaluate(model, tok, tasks, samples=0, log=lambda _m: None)[
                "greedy_pass@1"]

    measure("fp32", ref)
    measure("int4", q)
    _round_to_bf16_(ref)
    measure("bf16", ref)
    report: dict = {"bpb": bpb, "bpb_tokens": tokens, "bpb_device": str(device),
                    "bpb_arithmetic": "fp32 (amp off) for fp32, bf16 and int4 weights"}
    if ce is not None:
        report["code_eval"] = {"problems": len(tasks), **code,
                               "arithmetic": "bf16 autocast on CUDA, fp32 on CPU"}
    return report


def _write_report(e: MoEExport, name: str, suffix: str, report: dict) -> None:
    from quipu.fsio import write_text_atomic

    payload = json.dumps(report, indent=1)
    write_text_atomic(Path(e.out) / f"{suffix}.json", payload)
    if e.results_dir is not None:
        Path(e.results_dir).mkdir(parents=True, exist_ok=True)
        write_text_atomic(Path(e.results_dir) / f"{name}-{suffix.replace('_report', '')}.json",
                          payload)


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
    """Memory: at most three fp32 copies of the model are alive at once, so the peak
    host RAM is about 3 x 4 bytes x total parameters: ~12 GB for quipu-moe-1B (998M
    parameters, 4.0 GB per fp32 copy), which fits a 16 GB machine with little else
    running. The copies: the training-side reference model (kept throughout; the
    checkpoint itself is memory-mapped, and its optimizer states are never read), and
    at the worst moments either the standalone loader (its state_dict + its model, 2
    copies while loading) or the int4 model plus its dequantized state_dict while it is
    built. Every other model is compared through its logits on the small parity input
    and freed (del + gc) before the next one is built; the int4 quality measurement
    reuses the reference and int4 models, and rounds the reference to bf16 in place for
    the bf16 baseline. With --device cuda the two models move to the GPU for the
    quality measurement (~8 GB of VRAM for the 1B model)."""
    from quipu import evalsets, model_card
    from quipu.int4 import FORMAT, GROUP_SIZE, quantize_state

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
        ms = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        save_file(_strip_tied(ms, torch.bfloat16), str(out / "milestones" / f"{p.stem}.safetensors"),
                  metadata={"step": str(int(p.stem.split("_")[1])), "dtype": "bfloat16"})
        del ms
    tokenizer_files(out, Path(cfg.data.tokenizer), cfg.model.context, e.chat)
    shutil.copy2(REPO_ROOT / "hf" / "modeling_quipu_moe.py", out / "modeling_quipu_moe.py")

    # The reference: the training model with the final weights, kept to the end.
    ref = evalsets.load_model(cfg.model, state=state)
    del state
    _free()

    # Parity through the standalone loader: one model alive next to `ref` at a time.
    mq = _import_standalone(out / "modeling_quipu_moe.py", "modeling_quipu_moe")
    x = parity_input(cfg)
    parity: dict[str, float] = {}

    def check(label: str, weights: str, want: torch.Tensor) -> None:
        got = mq.load(str(out), weights=weights)
        parity[label] = (want - _logits(got, x)).abs().max().item()
        del got
        _free()
        log(f"parity {label}: max |logit diff| = {parity[label]:.2e}")

    ref_logits = _logits(ref, x)
    check("final (fp32)", "model.safetensors", ref_logits)
    if milestones:
        ms_model = evalsets.load_model(cfg.model, state=torch.load(
            milestones[-1], map_location="cpu", weights_only=True, mmap=True))
        ms_logits = _logits(ms_model, x)
        del ms_model
        _free()
        check(f"milestone {milestones[-1].stem} (bf16)",
              f"milestones/{milestones[-1].stem}.safetensors", ms_logits)
    saved = meta = q_logits = None
    if e.int4:
        saved, meta = quantize_state({k: v for k, v in ref.state_dict().items() if k != TIED},
                                     GROUP_SIZE)
        save_file(saved, str(out / "model-int4.safetensors"),
                  metadata={"step": str(step), "format": FORMAT, "int4": json.dumps(meta)})
        q = _int4_model(cfg, saved, meta)
        q_logits = _logits(q, x)
        del q                                  # rebuilt below: the standalone load needs the room
        _free()
        check("int4 (dequantized)", "model-int4.safetensors", q_logits)
    ok = all(v <= e.parity_tol for v in parity.values())
    _write_report(e, name, "parity", {"tolerance": e.parity_tol, "passed": ok,
                                      "max_abs_logit_diff": parity,
                                      "input_shape": list(x.shape)})
    if not ok:
        log(f"FAIL: the standalone model does not match the training model (tol {e.parity_tol:g})")
        return 1
    if e.int4:
        report: dict = {"parity": parity, "parity_tolerance": e.parity_tol,
                        "int4_vs_fp32_max_logit_diff": (ref_logits - q_logits).abs().max().item(),
                        "format": FORMAT, "group_size": GROUP_SIZE,
                        "sizes": {f: (out / f).stat().st_size
                                  for f in ("model.safetensors", "model-int4.safetensors")}}
        probe = out / "_bf16_size_probe.safetensors"
        save_file({k: v.to(torch.bfloat16) for k, v in ref.state_dict().items() if k != TIED},
                  str(probe))
        report["sizes"]["model (bf16, for comparison, not shipped)"] = probe.stat().st_size
        probe.unlink()
        _free()
        q = _int4_model(cfg, saved, meta)
        del saved
        report.update(_int4_quality(e, ref, q))
        del q
        _free()
        _write_report(e, name, "int4_report", report)
        for k, v in report["bpb"].items():
            log(f"int4 bpb {k}: fp32 {v['fp32']:.4f}, bf16 {v['bf16']:.4f} -> int4 "
                f"{v['int4']:.4f} ({v['int4'] / v['bf16'] - 1:+.2%} vs bf16)")
    del ref
    _free()
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
    moe.add_argument("--parity-tol", type=float, default=1e-5)
    moe.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                     help="where the int4 quality is measured (parity always runs on CPU)")
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
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    return export_moe(MoEExport(
        cfg=cfg, ckpt_dir=Path(args.ckpt_dir or cfg.train.ckpt_dir), out=Path(args.out),
        chat=args.chat, int4=args.int4, card_only=args.card_only, parity_tol=args.parity_tol,
        device=device,
        bpb_batches=args.bpb_batches, int4_code_eval=args.int4_code_eval, repo_id=args.repo_id,
        results_dir=p(args.results_dir), card_inputs=inputs))


if __name__ == "__main__":
    raise SystemExit(main())
