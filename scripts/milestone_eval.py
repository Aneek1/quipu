"""Evaluate every milestone snapshot of a run: loss, bits per byte and fixed-prompt samples.

Reads `<ckpt_dir>/milestones/step_NNNNNN.pt` (bf16 model weights only, written by
`quipu.train.Trainer.save_milestone`) plus the final checkpoint via `latest.pt`, and
for each one measures FineWeb val loss and code val loss over a fixed number of
batches from a fixed starting position, and generates from a fixed set of prompts
(greedy + one seeded sample). This is how the run records *when* the model learned
grammar, then coherence, then code.

Both model kinds (quipu-114m dense, quipu-moe): the model is built from the config
(quipu.model_factory) and the tokenizer is the config's (GPT-2 or the BPE
tokenizer.json). Every metric and sample of an MoE model runs with loop dispatch
(quipu.eval.loop_dispatch), so nothing depends on the rest of the batch.

Bits per byte (spec 7 and 11) on every evaluation split the shard set has
(quipu.evalsets.eval_splits: English val, each val_lang/<language>, code_val), over
the first --bpb-batches x micro_batch x context tokens of each, never wrapping. Unlike
loss per token it compares across tokenizers, so quipu-moe and quipu-114m can be put
side by side. metrics.json gets a "bpb" entry per checkpoint and bpb.md a table.

Prompts: quipu-114m's seven (English and code) for every model; a multilingual config
(data.text_language_weights set) adds two per language of spec 11 (LANG_PROMPTS),
greedy only (temperature 0).

Never modifies a checkpoint: everything is opened read-only, ideally with
`weights_only=True`; the only writes are to `--out-dir` (metrics.json, samples.md),
written atomically.

Runs unattended after the ~50-hour training run finishes, so it must not require
any interaction and must produce a clear, non-zero exit if there is nothing to
evaluate.

Run: uv run python scripts/milestone_eval.py --config configs/quipu-114m.toml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import math

import torch
import torch.nn as nn

from quipu import evalsets
from quipu.config import Config, load_config
from quipu.eval import estimate_loss, generate, loop_dispatch, nll_tokens_bytes
from quipu.fsio import replace_with_retry
from quipu.loader import TokenStream
from quipu.tokenizer import make_tokenizer
from quipu.train import LATEST, MILESTONE_DIR

_STEP_RE = re.compile(r"^step_(\d+)\.pt$")

TEXT_PROMPTS = [
    "The history of the printing press",
    "Photosynthesis is the process by which",
    "In mathematics, a prime number",
]
CODE_PROMPTS = [
    "def fibonacci(n):",
    "function debounce(fn, delay) {",
    "<!DOCTYPE html>\n<html>",
    "SELECT customer_id, COUNT(*)",
]
ALL_PROMPTS = TEXT_PROMPTS + CODE_PROMPTS
# Two per language of spec 11 other than English (English and code are covered by
# ALL_PROMPTS above), greedy only. Plain openings of a sentence, nothing to look up.
LANG_PROMPTS: dict[str, tuple[str, str]] = {
    "ind_Latn": ("Sejarah kota Jakarta dimulai", "Fotosintesis adalah proses"),
    "zsm_Latn": ("Malaysia terkenal dengan", "Cara terbaik untuk belajar bahasa baharu ialah"),
    "zho_Hans": ("光合作用是植物", "北京是中国的"),
    "zho_Hant": ("台灣的夜市", "學習程式設計的第一步是"),
    "jpn_Jpan": ("日本の四季は", "東京で一番有名な"),
    "kor_Hang": ("한국의 전통 음식은", "서울은 대한민국의"),
    "tam_Taml": ("தமிழ் மொழி உலகின்", "சென்னை நகரம்"),
    "hin_Deva": ("भारत की राजधानी", "प्रकाश संश्लेषण वह प्रक्रिया है"),
    "hin_Latn": ("Mera naam Rahul hai aur main", "Aaj mausam bahut"),
    "urd_Latn": ("Pakistan ka sab se bara shehar", "Mujhe kitabein parhna"),
}
GREEDY_ONLY_PROMPTS = [p for pair in LANG_PROMPTS.values() for p in pair]


def prompts_for(cfg: Config) -> list[str]:
    """The prompts a run is sampled with: ALL_PROMPTS, plus the language prompts for
    a multilingual config."""
    return ALL_PROMPTS + (GREEDY_ONLY_PROMPTS if cfg.data.text_language_weights else [])

MAX_NEW_TOKENS = 80
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_K = 50
SAMPLE_SEED = 1337


class NoCheckpointsFound(RuntimeError):
    """Nothing under <ckpt_dir>/milestones and no usable latest.pt either."""


def discover_checkpoints(ckpt_dir: Path) -> list[tuple[str, int, Path, dict | None]]:
    """Return (label, step, path, preloaded_final_state) quadruples, sorted by step,
    milestones first then a trailing "final" entry when it adds anything.
    `preloaded_final_state` is the "final" entry's fp32 model state_dict (already
    read off disk while deciding whether to include it), or None for a milestone
    entry; passing it on lets the caller avoid reading the full checkpoint (model +
    optimiser state) a second time.

    The final checkpoint is included unless its step equals the last milestone's
    step AND its weights are identical to that milestone's — a run that finishes
    exactly on a milestone step (e.g. steps == 4000) would otherwise be evaluated
    twice for nothing.
    """
    milestone_dir = ckpt_dir / MILESTONE_DIR
    found: list[tuple[int, Path]] = []
    if milestone_dir.is_dir():
        for p in milestone_dir.glob("step_*.pt"):
            m = _STEP_RE.match(p.name)
            if m and p.is_file():
                found.append((int(m.group(1)), p))
    found.sort()

    out: list[tuple[str, int, Path, dict | None]] = [
        (f"step_{step:06d}", step, path, None) for step, path in found
    ]

    latest_ptr = ckpt_dir / LATEST
    if latest_ptr.exists():
        # weights_only=True can't unpickle an arbitrary dict of Python ints/strings
        # reliably across torch versions for this tiny pointer file in every case,
        # but it is not a model checkpoint (no tensors), so there is nothing a
        # malicious/corrupt file here could do beyond raising; read it plainly.
        pointer = torch.load(latest_ptr, map_location="cpu", weights_only=True)
        final_path = ckpt_dir / pointer["file"]
        m = _STEP_RE.match(final_path.name)
        final_step = int(m.group(1)) if m else None
        # Read the full checkpoint (model + optimiser state) exactly once here, and
        # hand the model state_dict on to the caller, so load_final_model never has
        # to re-read it.
        final_full = torch.load(final_path, map_location="cpu", weights_only=False)
        final_state = final_full["model"]
        include = True
        if found and final_step == found[-1][0]:
            # Same step as the last milestone: include only if the weights differ
            # (the final checkpoint is fp32 + optimiser state; compare model weights).
            last_milestone_state = torch.load(found[-1][1], map_location="cpu", weights_only=True)
            include = not _state_dicts_equal(last_milestone_state, final_state)
        if include:
            out.append(("final", final_step if final_step is not None else -1, final_path, final_state))

    if not out:
        raise NoCheckpointsFound(
            f"no milestone snapshots in {milestone_dir} and no usable {latest_ptr}; "
            "nothing to evaluate"
        )
    return out


def _state_dicts_equal(a: dict, b: dict) -> bool:
    ak, bk = set(a), set(b)
    # Both the milestone and the final checkpoint's "model" state_dict carry
    # "lm_head.weight" and "embed.weight" as separate keys (sharing one storage on
    # disk), so comparing key sets directly is meaningful.
    if ak != bk:
        return False
    for k in ak:
        va, vb = a[k], b[k]
        if va.shape != vb.shape:
            return False
        # `a` (the milestone) is bf16-rounded; `b` (the final checkpoint) is fp32.
        # Round `b` through bf16 too before comparing, or an identical checkpoint
        # saved at both precisions would look "different" from rounding alone.
        if not torch.equal(va.to(torch.bfloat16), vb.to(torch.bfloat16)):
            return False
    return True


def load_milestone_model(cfg: Config, path: Path, device: str) -> nn.Module:
    """Fresh model (the config's kind) with a milestone's bf16 weights loaded (cast to
    fp32), read-only.

    Corrected understanding (verified against actual `_bf16_state_dict` behaviour
    and confirmed by a reviewer): the milestone state_dict DOES contain
    "lm_head.weight" — it and "embed.weight" share a single on-disk storage (that's
    the size saving), but both keys are present. So this loads with the DEFAULT
    strict=True (no dropped keys, no assign=True, which would break the tie): if a
    key is missing, renamed, or extra, `load_state_dict` itself raises loudly. The
    dtype cast bf16 -> fp32 happens for free, since `load_state_dict` copies into
    the destination module's existing fp32 parameters in place.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    try:
        return evalsets.load_model(cfg.model, device=device, state=state)
    except RuntimeError as exc:
        raise AssertionError(f"milestone {path} did not match the model exactly: {exc}") from exc


def load_final_model(cfg: Config, path: Path, device: str, state: dict | None = None) -> nn.Module:
    """The full checkpoint is `{"model": state_dict (fp32), "optimizer": ..., ...}`.
    torch.save writes the optimiser state as a plain (non-tensor) nested dict/list of
    Python numbers and tensors, which weights_only=True's restricted unpickler does
    not accept for every torch version this script may run under, so a from-scratch
    load falls back to weights_only=False. Still read-only: map_location="cpu", never
    written back, and only the "model" key is used.

    `state` lets a caller that already read the checkpoint (discover_checkpoints, to
    decide whether "final" duplicates the last milestone) hand over the model
    state_dict directly, so the ~1.4 GB full checkpoint is never read from disk twice.
    """
    if state is None:
        full = torch.load(path, map_location="cpu", weights_only=False)
        state = full["model"]
    return evalsets.load_model(cfg.model, device=device, state=state)


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"
    return "cpu"


def _rewind_stream(stream: TokenStream) -> None:
    """Every checkpoint must see identical val tokens, so always start from position
    0 regardless of anything else that touched the stream."""
    stream.load_state_dict({"position": 0, "wraps": 0})


def evaluate_losses(
    model: nn.Module, cfg: Config, device: str, eval_batches: int
) -> tuple[float, float | None, str | None]:
    """(text_val_loss, code_val_loss_or_None, note_or_None)."""
    shard_dir = Path(cfg.data.shard_dir)
    text_stream = TokenStream(shard_dir / "val", cfg.train.micro_batch, cfg.model.context)
    _rewind_stream(text_stream)
    text_loss = estimate_loss(model, text_stream, eval_batches, device)

    code_val_dir = shard_dir / "code_val"
    if not code_val_dir.is_dir() or not any(code_val_dir.glob("shard_*.bin")):
        return text_loss, None, f"code_val shards not found at {code_val_dir}"
    code_stream = TokenStream(code_val_dir, cfg.train.micro_batch, cfg.model.context)
    _rewind_stream(code_stream)
    code_loss = estimate_loss(model, code_stream, eval_batches, device)
    return text_loss, code_loss, None


def evaluate_bpb(
    model: nn.Module, cfg: Config, device: str, batches: int, byte_lengths: list[int]
) -> dict[str, dict]:
    """label -> {"bpb", "loss", "tokens", "bytes"} for every evaluation split of the
    shard set (quipu.evalsets.eval_splits), over the first `batches` batches of each.
    A split whose targets hold no bytes gets "bpb": None."""
    out: dict[str, dict] = {}
    for label, split in evalsets.eval_splits(cfg.data.shard_dir).items():
        data = evalsets.split_batches(split, cfg.train.micro_batch, cfg.model.context, batches)
        nll, n_tok, n_bytes = nll_tokens_bytes(model, data, byte_lengths, device)
        out[label] = {
            "bpb": nll / (n_bytes * math.log(2)) if n_bytes else None,
            "loss": nll / n_tok, "tokens": n_tok, "bytes": n_bytes,
        }
    return out


def greedy_generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, device: str) -> torch.Tensor:
    """Argmax decoding. `generate`'s top_k path with top_k=1 still routes through
    multinomial sampling over a one-hot distribution, which is greedy in outcome but
    not in mechanism; this is deliberately argmax so "greedy" means what it says.
    MoE models run with loop dispatch."""
    was_training = model.training
    model.eval()
    idx = idx.to(device)
    with torch.no_grad(), loop_dispatch(model):
        for _ in range(max_new_tokens):
            window = idx[:, -model.cfg.context :]
            logits = model(window)[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
    if was_training:
        model.train()
    return idx


def greedy_for_prompt(model: nn.Module, tok, prompt: str, device: str) -> tuple[str, None]:
    """(greedy_continuation, None): the language prompts are greedy only."""
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = greedy_generate(model, ids, MAX_NEW_TOKENS, device)
    return tok.decode(out[0].tolist()), None


def sample_for_prompt(
    model: nn.Module, tok, prompt: str, device: str
) -> tuple[str, str]:
    """(greedy_continuation, sampled_continuation) as decoded text, prompt included."""
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    greedy_out = greedy_generate(model, ids.clone(), MAX_NEW_TOKENS, device)
    greedy_text = tok.decode(greedy_out[0].tolist())

    # Fixed seed immediately before sampling, so the sampled continuation is
    # reproducible across runs regardless of how many random draws happened before
    # (greedy decoding draws none, but future changes here must not be able to
    # perturb this seed's effect).
    torch.manual_seed(SAMPLE_SEED)
    sampled_out = generate(
        model, ids.clone(), MAX_NEW_TOKENS, device,
        temperature=SAMPLE_TEMPERATURE, top_k=SAMPLE_TOP_K,
    )
    sampled_text = tok.decode(sampled_out[0].tolist())
    return greedy_text, sampled_text


def run(cfg: Config, device: str, eval_batches: int, out_dir: Path,
        bpb_batches: int | None = None) -> bool:
    """Evaluate every checkpoint and write metrics.json + samples.md + bpb.md.
    bpb_batches defaults to eval_batches.

    A single corrupt/unreadable checkpoint must not lose the other checkpoints'
    results after a 50-hour run: each checkpoint's load + eval + sampling is
    isolated in its own try/except, a failure is recorded (metrics gets an
    "error" entry, samples.md notes it) and evaluation continues. Both files are
    always written. Returns True iff every checkpoint succeeded; the caller uses
    this to decide the process exit code.
    """
    ckpt_dir = Path(cfg.train.ckpt_dir)
    checkpoints = discover_checkpoints(ckpt_dir)
    tok = make_tokenizer(cfg.data.tokenizer)
    byte_lengths = tok.token_byte_lengths()
    prompts = prompts_for(cfg)
    bpb_batches = eval_batches if bpb_batches is None else bpb_batches

    metrics: list[dict] = []
    failures: list[tuple[str, int, str]] = []
    # samples[prompt] = list of (label, step, greedy_text, sampled_text or None),
    # appended in checkpoint order (checkpoints is already step-sorted).
    samples: dict[str, list[tuple[str, int, str, str | None]]] = {p: [] for p in prompts}

    for label, step, path, preloaded_state in checkpoints:
        t0 = time.perf_counter()
        # Buffered locally and merged into `samples`/`metrics` only once this
        # checkpoint's whole body has succeeded: if sample_for_prompt (or anything
        # else here) raises partway through the prompt loop, the prompts already
        # sampled for this checkpoint must not survive into samples.md while
        # metrics.json marks the checkpoint as failed and the "No samples were
        # generated" banner is shown for it — that would be self-contradictory.
        checkpoint_samples: dict[str, tuple[str, str | None]] = {}
        try:
            model = (
                load_final_model(cfg, path, device, state=preloaded_state)
                if label == "final"
                else load_milestone_model(cfg, path, device)
            )

            text_loss, code_loss, note = evaluate_losses(model, cfg, device, eval_batches)
            bpb = evaluate_bpb(model, cfg, device, bpb_batches, byte_lengths)

            for prompt in ALL_PROMPTS:
                checkpoint_samples[prompt] = sample_for_prompt(model, tok, prompt, device)
            for prompt in prompts[len(ALL_PROMPTS):]:
                checkpoint_samples[prompt] = greedy_for_prompt(model, tok, prompt, device)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for prompt, (greedy_text, sampled_text) in checkpoint_samples.items():
                samples[prompt].append((label, step, greedy_text, sampled_text))

            elapsed = time.perf_counter() - t0
            record = {
                "label": label,
                "step": step,
                "text_val_loss": text_loss,
                "code_val_loss": code_loss,
                "bpb": bpb,
                "seconds": elapsed,
            }
            if note:
                record["note"] = note
            metrics.append(record)
            print(
                f"{label} (step {step}): text_val_loss={text_loss:.4f} "
                f"code_val_loss={'n/a' if code_loss is None else f'{code_loss:.4f}'} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            error = f"{type(exc).__name__}: {exc}"
            metrics.append({"label": label, "step": step, "error": error, "seconds": elapsed})
            failures.append((label, step, error))
            print(f"error: {label} (step {step}) failed: {error}", file=sys.stderr, flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir / "metrics.json", metrics)
    _write_samples(out_dir / "samples.md", samples, failures)
    _write_text(out_dir / "bpb.md", bpb_table(metrics))
    _write_text(out_dir / "samples.json", json.dumps(
        {p: [{"label": l, "step": s, "greedy": g, "sampled": x} for l, s, g, x in entries]
         for p, entries in samples.items()}, indent=1, ensure_ascii=False))
    return not failures


def bpb_table(metrics: list[dict]) -> str:
    """Markdown: one row per checkpoint, one column per evaluation split."""
    labels: list[str] = []
    for m in metrics:
        for k in m.get("bpb") or {}:
            if k not in labels:
                labels.append(k)
    lines = ["# Bits per byte", "",
             "Summed next-token loss (bits) over the UTF-8 bytes of the targets, on the "
             "first tokens of each evaluation split (never wrapping). Lower is better; "
             "comparable across tokenizers.", ""]
    if not labels:
        return "\n".join(lines + ["No checkpoint was evaluated.", ""])
    names = [evalsets.LANGUAGES.get(k, k) for k in labels]
    lines += ["| checkpoint | step | " + " | ".join(names) + " |",
              "|---|---:|" + "---:|" * len(labels)]
    for m in metrics:
        b = m.get("bpb") or {}
        cells = [("-" if b.get(k, {}).get("bpb") is None else f"{b[k]['bpb']:.4f}")
                 for k in labels]
        lines.append(f"| {m['label']} | {m['step']} | " + " | ".join(cells) + " |")
    last = next((m for m in reversed(metrics) if m.get("bpb")), None)
    if last:
        lines += ["", f"Sample size ({last['label']}): "
                  + "; ".join(f"{evalsets.LANGUAGES.get(k, k)} {v['tokens']:,} tokens / "
                              f"{v['bytes']:,} bytes" for k, v in last["bpb"].items()) + "."]
    return "\n".join(lines) + "\n"


def _write_text(path: Path, payload: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    replace_with_retry(tmp, path)


def _write_metrics(path: Path, metrics: list[dict]) -> None:
    payload = json.dumps({"checkpoints": metrics}, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    replace_with_retry(tmp, path)


def _write_samples(
    path: Path,
    samples: dict[str, list[tuple[str, int, str, str | None]]],
    failures: list[tuple[str, int, str]] | None = None,
) -> None:
    lines = [
        "# Milestone samples",
        "",
        "Fixed-prompt generations from every milestone snapshot of this weekend run, "
        "organised by prompt and then by checkpoint in step order, so the "
        "progression of learning reads top to bottom for each prompt. "
        "\"Greedy\" is argmax decoding (80 new tokens); \"sampled\" is one "
        "temperature=0.8, top_k=50 continuation with a fixed seed, so it is "
        "reproducible across runs. The per-language prompts (multilingual runs) "
        "are greedy only.",
        "",
    ]
    if failures:
        lines.append("## Failed checkpoints")
        lines.append("")
        lines.append(
            "These checkpoints could not be loaded or evaluated; see metrics.json "
            "for the same errors. No samples were generated for them."
        )
        lines.append("")
        for label, step, error in failures:
            lines.append(f"- **{label}** (step {step}): {error}")
        lines.append("")
    for prompt in samples:
        lines.append(f"## Prompt: `{prompt}`")
        lines.append("")
        for label, step, greedy_text, sampled_text in samples[prompt]:
            lines.append(f"### {label} (step {step})")
            lines.append("")
            lines.append("**Greedy:**")
            lines.append("```")
            lines.append(greedy_text)
            lines.append("```")
            lines.append("")
            if sampled_text is None:   # a language prompt: greedy only
                continue
            lines.append("**Sampled (temperature=0.8, top_k=50, seed={}):**".format(SAMPLE_SEED))
            lines.append("```")
            lines.append(sampled_text)
            lines.append("```")
            lines.append("")
    payload = "\n".join(lines) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    replace_with_retry(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quipu-114m.toml")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="auto (cuda if usable), cuda or cpu",
    )
    parser.add_argument("--out-dir", default="results/milestones")
    parser.add_argument("--bpb-batches", type=int, default=None,
                        help="batches per evaluation split for bits per byte "
                             "(default: --eval-batches)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = _pick_device(args.device)

    try:
        ok = run(cfg, device, args.eval_batches, Path(args.out_dir), args.bpb_batches)
    except NoCheckpointsFound as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
