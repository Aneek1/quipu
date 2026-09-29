"""The quipu-moe model card, generated from the run's result files (spec 7, 11-13).

hf/README_moe.md is the template: {{field}} placeholders, filled by render(). Every
field is built from a file the run wrote (run config, shard manifest, A/B summary,
spend ledger, milestone metrics, code_eval results, int4 report, expert usage); a
field whose input is missing renders as NOT_MEASURED, never as a guess. The same
template serves the base model and the chat model (variant="chat").

Release names (spec 13): base AneekC/quipu-moe-1B-A149M, chat
AneekC/quipu-moe-1B-A149M-chat. The card always states total AND active parameters.
"""
from __future__ import annotations

import dataclasses
import json
import re
import tomllib
from pathlib import Path
from typing import Any

from quipu import evalsets

NOT_MEASURED = "_not yet measured_"
BASE_REPO = "AneekC/quipu-moe-1B-A149M"
CHAT_REPO = BASE_REPO + "-chat"
LID_REPO = "AneekC/lid-specialists-9plus1"

DATA_LICENCES = {
    "code": "the permissive subset of codeparrot/github-code-clean (each file keeps its own "
            "licence)",
    "english": "HuggingFaceFW/fineweb-edu, ODC-By 1.0",
    "fineweb2": "HuggingFaceFW/fineweb-2, ODC-By 1.0",
}

# Published numbers of reference models: cited from the source named, NOT re-run.
PUBLISHED: list[dict[str, Any]] = [
    {"model": "CodeGen-350M-mono", "params": "350M",
     "humaneval_pass@1": 12.76, "humaneval_pass@10": 23.11,
     "mbpp_pass@1": None, "mbpp_pass@10": None,
     "source": "Nijkamp et al., CodeGen (ICLR 2023), arXiv:2203.13474, Table 1",
     "note": "pass@k sampled at the best temperature per k; MBPP not in that table"},
    {"model": "SantaCoder-1.1B", "params": "1.1B",
     "humaneval_pass@1": 18.0, "humaneval_pass@10": 29.0,
     "mbpp_pass@1": 35.0, "mbpp_pass@10": 58.0,
     "source": "bigcode/santacoder model card (self-reported; Allal et al., arXiv:2301.03988)",
     "note": "MBPP subset as in the BigCode harness, not necessarily the 257 sanitized"},
    {"model": "SmolLM-135M", "params": "135M",
     "humaneval_pass@1": None, "humaneval_pass@10": None,
     "mbpp_pass@1": None, "mbpp_pass@10": None,
     "source": "huggingface.co/blog/smollm (HumanEval shown as a chart only, no number in the text)",
     "note": "no published number to cite; cheap to re-run on the laptop"},
    {"model": "SmolLM-360M", "params": "360M",
     "humaneval_pass@1": None, "humaneval_pass@10": None,
     "mbpp_pass@1": None, "mbpp_pass@10": None,
     "source": "huggingface.co/blog/smollm (HumanEval shown as a chart only, no number in the text)",
     "note": "no published number to cite; cheap to re-run on the laptop"},
    {"model": "Qwen2.5-Coder-1.5B (base)", "params": "1.5B",
     "humaneval_pass@1": 43.9, "humaneval_pass@10": None,
     "mbpp_pass@1": 69.2, "mbpp_pass@10": None,
     "source": "Hui et al., Qwen2.5-Coder Technical Report, arXiv:2409.12186, Table 5",
     "note": "greedy; MBPP is the EvalPlus MBPP set, not the 257 sanitized"},
]


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def code_results_table(ours: list[dict[str, Any]], temperature: float = 0.8,
                       top_p: float = 0.95) -> str:
    """Markdown: our models (measured) then PUBLISHED (cited, not re-run).
    ours: [{"model", "params", "results": {"humaneval"|"mbpp": summary}}]."""
    lines = [
        "| model | params | HumanEval pass@1 | HumanEval pass@10 | MBPP pass@1 | MBPP pass@10 | source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for m in ours:
        r = m["results"]
        he, mb = r.get("humaneval") or {}, r.get("mbpp") or {}
        n = he.get("samples") or mb.get("samples") or 0
        lines.append(f"| **{m['model']}** | {m['params']} | {_pct(he.get('greedy_pass@1'))} | "
                     f"{_pct(he.get('pass@10'))} | {_pct(mb.get('greedy_pass@1'))} | "
                     f"{_pct(mb.get('pass@10'))} | measured here: greedy pass@1; pass@10 from "
                     f"n={n} at T={temperature}, top-p {top_p} |")
    for p in PUBLISHED:
        lines.append(f"| {p['model']} | {p['params']} | {_pct(p['humaneval_pass@1'])} | "
                     f"{_pct(p['humaneval_pass@10'])} | {_pct(p['mbpp_pass@1'])} | "
                     f"{_pct(p['mbpp_pass@10'])} | published, not re-run: {p['source']}"
                     f" ({p['note']}) |")
    return "\n".join(lines)


# ---- parameters -------------------------------------------------------------------------

def count_params(model_cfg) -> tuple[int, int]:
    """(total, active per token) of the configured model, built on the meta device (no
    memory). Tied embedding counted once; active = everything but the routed experts,
    plus top_k / n_experts of them. Embeddings are included in both."""
    import torch

    from quipu.model_factory import build_model

    with torch.device("meta"):
        model = build_model(model_cfg)
    seen, total, routed = set(), 0, 0
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if ".moe.experts." in name:
            routed += p.numel()
    if model_cfg.kind != "moe":
        return total, total
    active = total - routed + routed * model_cfg.top_k // model_cfg.n_experts
    return total, active


def release_name(total: int, active: int) -> str:
    """quipu-moe-<total>-A<active>, e.g. quipu-moe-1B-A149M for 998.0M / 148.7M."""
    def short(n: int, billions: bool) -> str:
        if billions and n >= 0.95e9:
            return f"{n / 1e9:.0f}B"
        return f"{n / 1e6:.0f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"
    return f"quipu-moe-{short(total, True)}-A{short(active, False)}"


def _fmt_params(n: int) -> str:
    return f"{n / 1e9:.3f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


# ---- inputs ------------------------------------------------------------------------------

@dataclasses.dataclass
class CardInputs:
    """Where the card's inputs live; any may be missing."""
    run_dir: Path | None = None          # results/moe: run_config.toml, plan.json, runs/*.json
    ab_dir: Path | None = None           # results/ab: summary.md, winners.toml
    manifest: Path | None = None         # <shard_dir>/manifest.json
    ledger: Path | None = None           # results/spend.json
    milestones_dir: Path | None = None   # milestone_eval out: metrics.json, samples.json
    code_eval_dir: Path | None = None    # results/code_eval
    int4_report: Path | None = None      # int4_report.json from export_hf --int4
    experts_dir: Path | None = None      # results/experts: usage.json
    sft_manifest: Path | None = None     # the chat SFT data manifest (M12)
    hardware: str | None = None


def _json(path: Path | None) -> Any:
    if path is None or not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    return Path(path).read_text(encoding="utf-8")


def _section(md: str, heading: str) -> str | None:
    """The body of '## heading' in a markdown text, up to the next '## '."""
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", md, re.M | re.S)
    return m.group(1).strip() if m else None


# ---- fields ------------------------------------------------------------------------------

def _architecture(cfg, total: int, active: int, run_cfg: dict | None) -> str:
    m = cfg.model
    train = (run_cfg or {}).get("train", {}) if run_cfg else {}
    optimizer = train.get("optimizer", cfg.train.optimizer)
    opt = ("Muon (Newton-Schulz orthogonalised per attention head) for weight matrices, "
           "AdamW for the rest" if optimizer == "muon" else "AdamW")
    act = ("SiTU-GLU (soft-capped SwiGLU, beta_gate "
           f"{m.situ_beta_gate:g}, beta_up {m.situ_beta_up:g})" if m.activation == "situ_glu"
           else "SwiGLU")
    depth = (f"Block Attention Residuals, {m.attnres_blocks} blocks" if m.attnres_blocks
             else "plain pre-norm residual (Block Attention Residuals not used)")
    precision = train.get("precision", getattr(cfg.train, "precision", "bf16"))
    rows = [
        ("Type", "Decoder-only transformer, sparse mixture of experts"),
        ("Parameters", f"{total:,} total / {active:,} active per token "
                       f"({_fmt_params(total)} / {_fmt_params(active)}; embeddings tied, counted once)"),
        ("Layers / width", f"{m.n_layer} layers, d_model {m.d_model}"),
        ("Attention", f"grouped-query, {m.n_head} query / {m.n_kv_head} key-value heads, "
                      f"head dim {m.head_dim}, RoPE base {m.rope_base:g}"),
        ("Feed-forward", f"{m.shared_experts} shared expert(s) (hidden {m.shared_hidden}) + "
                         f"{m.n_experts} routed experts (hidden {m.expert_hidden}), top-{m.top_k}"),
        ("Routing", "softmax router, Top-k on score + bias, weights renormalised over the "
                    "chosen experts; Quantile Balancing (auxiliary-loss-free bias)"),
        ("Expert activation", act),
        ("Depth mixing", depth),
        ("Optimizer", opt),
        ("Training precision", f"{precision} (router and balancing math in fp32)"),
        ("Context", f"{m.context:,} tokens"),
        ("Tokenizer", f"byte-level BPE, {m.vocab_size:,} tokens, trained for this model "
                      "(code-aware whitespace, ten languages)"),
    ]
    return "| | |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def _training(inp: CardInputs, cfg, run_cfg: dict | None) -> str:
    rows: list[tuple[str, str]] = []
    runs = sorted((inp.run_dir / "runs").glob("*.json")) if inp.run_dir else []
    last = None
    for p in runs:
        rec = _json(p) or {}
        steps = rec.get("steps") or []
        if steps and (last is None or steps[-1].get("step", 0) > last.get("step", 0)):
            last = steps[-1]
    rows.append(("Tokens trained", f"{last['tokens']:,} ({last['step']:,} steps)"
                 if last and "tokens" in last else NOT_MEASURED))
    plan = _json(inp.run_dir / "plan.json") if inp.run_dir else None
    rows.append(("Throughput", f"{plan['tokens_per_s']:,.0f} tokens/s (throughput gate)"
                 if plan and plan.get("tokens_per_s") else NOT_MEASURED))
    rows.append(("Hardware", inp.hardware or NOT_MEASURED))
    cost = NOT_MEASURED
    if inp.ledger is not None and Path(inp.ledger).is_file():
        try:
            from quipu.spend import Ledger
            led = Ledger.load(inp.ledger)
            last_seen = max((s["last_seen"] for s in led.sessions), default=None)
            spent = led.spent_usd(now=last_seen) if last_seen is not None else led.spent_usd()
            cost = (f"${spent:.2f} for the whole rented-box session (setup, shard build, A/B "
                    "runs, the full run, evaluation), from the spend ledger")
        except Exception as exc:  # a corrupt ledger must not break the card
            cost = f"{NOT_MEASURED} (ledger unreadable: {type(exc).__name__})"
    rows.append(("Cost", cost))
    train = (run_cfg or {}).get("train", {})
    rows.append(("Schedule", f"peak LR {train.get('lr', cfg.train.lr):g}, warmup "
                             f"{train.get('warmup_steps', cfg.train.warmup_steps)} steps, cosine "
                             f"to {train.get('lr_min', cfg.train.lr_min):g}; batch "
                             f"{train.get('batch_tokens', cfg.train.batch_tokens):,} tokens"))
    return "| | |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def _ab(inp: CardInputs) -> str:
    md = _text(inp.ab_dir / "summary.md") if inp.ab_dir else None
    if not md:
        return NOT_MEASURED
    parts = []
    for h in ("Runs", "Decisions"):
        body = _section(md, h)
        if body:
            parts.append(f"**{h}** (from results/ab/summary.md, verbatim)\n\n{body}")
    return "\n\n".join(parts) or NOT_MEASURED


def _fp8(inp: CardInputs, run_cfg: dict | None) -> str:
    precision = ((run_cfg or {}).get("train", {}) or {}).get("precision", "bf16")
    md = (_text(inp.ab_dir / "summary.md") if inp.ab_dir else None) or ""
    tried = re.search(r"\bprecision\b", _section(md, "Decisions") or "") is not None
    if precision == "fp8":
        return "FP8 training was used: it won its A/B pair (see the decisions above)."
    if tried:
        return ("FP8 training was tried in A/B pair 4 and lost (see the decisions above: it "
                "needed >= 1.2x tokens/s with loss within seed noise); the run is bf16.")
    return ("FP8 training was not tried in the paid run: the owner decided (2026-09-29) not to "
            "spend A/B time on the FP8 pair, so the run is bf16. The FP8 option exists in the "
            "training code (quipu/fp8.py) and was only checked on the laptop.")


def _data(inp: CardInputs, cfg) -> str:
    man = _json(inp.manifest)
    lines = []
    if man:
        src = man.get("sources") or {}
        lines += ["| source | dataset | revision | licence |", "|---|---|---|---|"]
        for key, s in src.items():
            lines.append(f"| {key} | {s.get('dataset', '-')} | `{str(s.get('revision', '-'))[:12]}` | "
                         f"{DATA_LICENCES.get(key, '-')} |")
        mix = man.get("mix") or {}
        if mix.get("achieved"):
            lines += ["", "Mix achieved (share of training tokens): "
                      + ", ".join(f"{k} {v:.1%}" if isinstance(v, (int, float)) else f"{k} {v}"
                                  for k, v in mix["achieved"].items()) + "."]
        lines += ["", f"Training tokens in the shard set: {man.get('total_tokens', 0):,}."]
    else:
        lines += [f"Planned mix (from the config; the shard manifest is {NOT_MEASURED}): "
                  f"{cfg.data.code_share:.0%} code, the rest text "
                  f"({', '.join(f'{k} {v:.3f}' for k, v in cfg.data.text_language_weights.items())} "
                  "of text)."]
    lines += ["", "Code licences kept: " + ", ".join(cfg.data.code_licenses) + "."]
    return "\n".join(lines)


def _decontam(inp: CardInputs) -> str:
    man = _json(inp.manifest)
    d = (man or {}).get("decontamination")
    if not d or not d.get("dropped"):
        return NOT_MEASURED
    lines = ["| split | " + " | ".join(("humaneval", "mbpp")) + " |", "|---|---:|---:|"]
    for split, counts in d["dropped"].items():
        counts = counts or {}
        lines.append(f"| {split} | {counts.get('humaneval', 0):,} | {counts.get('mbpp', 0):,} |")
    return ("Documents dropped because they contain a HumanEval or MBPP problem "
            "(quipu/decontam.py: whitespace-collapsed substring, or >= 50% of a solution's "
            "13-grams):\n\n" + "\n".join(lines))


def _lid(inp: CardInputs, cfg) -> str:
    man = _json(inp.manifest)
    lid = (man or {}).get("lid")
    head = (f"Language identification: every text document was checked with the fastText "
            f"specialist from [{cfg.data.lid_model}](https://huggingface.co/{cfg.data.lid_model}) "
            f"(MIT, revision `{cfg.data.lid_revision[:12]}`), the owner's own model; a document "
            "whose top label is not its source language was dropped, and FineWeb-2's cmn_Hani "
            "was split into Simplified and Traditional Chinese by its labels.")
    if not lid or not lid.get("by_language"):
        return head + f"\n\nPer-language kept / dropped counts: {NOT_MEASURED}."
    lines = ["| language | kept | dropped | kept while unsure |", "|---|---:|---:|---:|"]
    for x, s in lid["by_language"].items():
        lines.append(f"| {evalsets.LANGUAGES.get(x, x)} | {s.get('kept', 0):,} | "
                     f"{s.get('dropped', 0):,} | {s.get('kept_while_unsure', 0):,} |")
    return head + "\n\n" + "\n".join(lines)


def _bpb(inp: CardInputs) -> str:
    metrics = _json(inp.milestones_dir / "metrics.json") if inp.milestones_dir else None
    cps = [c for c in (metrics or {}).get("checkpoints", []) if c.get("bpb")]
    if not cps:
        return NOT_MEASURED
    labels = list(cps[-1]["bpb"])
    lines = ["| checkpoint | step | " + " | ".join(evalsets.LANGUAGES.get(k, k) for k in labels) + " |",
             "|---|---:|" + "---:|" * len(labels)]
    for c in cps:
        lines.append(f"| {c['label']} | {c['step']:,} | " + " | ".join(
            "-" if (c["bpb"].get(k) or {}).get("bpb") is None else f"{c['bpb'][k]['bpb']:.3f}"
            for k in labels) + " |")
    return ("Bits per byte (lower is better; comparable across tokenizers) on held-out text "
            "per language and held-out code:\n\n" + "\n".join(lines))


def _code_eval(inp: CardInputs, names: list[str], params: str) -> str:
    ours = []
    for name in names:
        data = _json(inp.code_eval_dir / f"{name}.json") if inp.code_eval_dir else None
        if data and data.get("results"):
            partial = any(b.get("evaluated", 0) < b.get("problems", 0)
                          for b in (data.get("benchmarks") or {}).values())
            ours.append({"model": name + (" (partial run)" if partial else ""), "params": params,
                         "results": data["results"]})
    if not ours:
        return NOT_MEASURED + " (published numbers of reference models, for when it is:)\n\n" \
            + code_results_table([])
    return code_results_table(ours)


def _int4(inp: CardInputs) -> str:
    rep = _json(inp.int4_report)
    if not rep:
        return NOT_MEASURED
    lines = [f"Weight-only int4 ({rep.get('format')}, groups of {rep.get('group_size')} along "
             "the last dimension, fp16 scale + minimum per group) for the attention, shared "
             "and routed expert matrices; embedding bf16; router, norms and biases fp32. "
             "Dequantized to fp32 on load (the file is smaller; memory once loaded is not).",
             "", "| file | size |", "|---|---:|"]
    for f, b in (rep.get("sizes") or {}).items():
        lines.append(f"| `{f}` | {b / 1e6:,.1f} MB |")
    q = rep.get("bpb") or {}
    if q:
        lines += ["", "| split | fp32 bpb | int4 bpb | change |", "|---|---:|---:|---:|"]
        for k, v in q.items():
            lines.append(f"| {evalsets.LANGUAGES.get(k, k)} | {v['fp32']:.4f} | {v['int4']:.4f} | "
                         f"{v['int4'] - v['fp32']:+.4f} ({(v['int4'] / v['fp32'] - 1):+.2%}) |")
        lines.append(f"\nMeasured on the first {rep.get('bpb_tokens_per_split', '?'):,} tokens of "
                     "each split." if isinstance(rep.get('bpb_tokens_per_split'), int) else "")
    ce = rep.get("code_eval")
    lines.append("" if not ce else
                 f"\nHumanEval greedy pass@1 on the first {ce['problems']} problems: fp32 "
                 f"{ce['fp32']:.1f}, int4 {ce['int4']:.1f}.")
    lines.append(f"\nStepbuild score, int4 vs bf16: {NOT_MEASURED} (needs the chat/step fine-tune).")
    return "\n".join(l for l in lines if l is not None)


def _experts(inp: CardInputs) -> str:
    usage = _json(inp.experts_dir / "usage.json") if inp.experts_dir else None
    if not usage:
        return NOT_MEASURED
    lines = ["| split | dead experts | overloaded experts |", "|---|---:|---:|"]
    for k, u in usage["splits"].items():
        lines.append(f"| {evalsets.LANGUAGES.get(k, k)} | {len(u['dead'])} | {len(u['overloaded'])} |")
    return (f"Routing frequency per expert, per layer and language ({usage['n_layer']} layers x "
            f"{usage['n_experts']} experts): the full tables and a heatmap are in the repository "
            f"(results/experts). Dead: < {usage['dead_below_x_target']:.0%} of the balanced target; "
            f"overloaded: > {usage['overloaded_above_x_target']:.0%}.\n\n" + "\n".join(lines))


def _samples(inp: CardInputs) -> str:
    data = _json(inp.milestones_dir / "samples.json") if inp.milestones_dir else None
    if not data:
        return NOT_MEASURED
    lines = []
    for prompt, entries in data.items():
        if not entries:
            continue
        e = entries[-1]
        text = e["greedy"].replace("```", "'''")
        if len(text) > 400:
            text = text[:400] + " ..."
        lines += [f"**{prompt.strip()}** ({e['label']}, greedy)", "", "```", text, "```", ""]
    return ("Greedy (temperature 0) continuations of the fixed prompts, final checkpoint, "
            "unedited: good and bad alike.\n\n" + "\n".join(lines)).strip()


def _sft(inp: CardInputs) -> str:
    man = _json(inp.sft_manifest)
    if not man:
        return NOT_MEASURED
    return "```json\n" + json.dumps(man.get("sources", man), indent=1)[:3000] + "\n```"


# ---- assembly ----------------------------------------------------------------------------

def fields(cfg, inp: CardInputs, *, variant: str = "base", total: int | None = None,
           active: int | None = None, repo_id: str | None = None) -> dict[str, str]:
    if variant not in ("base", "chat"):
        raise ValueError(f"variant must be 'base' or 'chat', got {variant!r}")
    if total is None or active is None:
        total, active = count_params(cfg.model)
    name = release_name(total, active) + ("-chat" if variant == "chat" else "")
    repo = repo_id or ("AneekC/" + name)
    run_cfg = None
    if inp.run_dir and (inp.run_dir / "run_config.toml").is_file():
        with open(inp.run_dir / "run_config.toml", "rb") as f:
            run_cfg = tomllib.load(f)
    params = f"{_fmt_params(total)} total / {_fmt_params(active)} active"
    base_name = release_name(total, active)
    f = {
        "repo_id": repo, "model_name": name, "base_repo": "AneekC/" + base_name,
        "variant": variant, "params": params,
        "total_params": f"{total:,}", "active_params": f"{active:,}",
        "architecture": _architecture(cfg, total, active, run_cfg),
        "training": _training(inp, cfg, run_cfg),
        "ab_results": _ab(inp), "fp8": _fp8(inp, run_cfg),
        "data": _data(inp, cfg), "decontamination": _decontam(inp), "lid": _lid(inp, cfg),
        "lid_model": cfg.data.lid_model,
        "bpb": _bpb(inp), "code_eval": _code_eval(inp, [base_name, base_name + "-chat"]
                                                  if variant == "chat" else [base_name], params),
        "int4": _int4(inp), "experts": _experts(inp), "samples": _samples(inp),
        "sft_data": _sft(inp) if variant == "chat" else "",
    }
    if variant == "chat":
        f["kind_line"] = (f"This is the **chat model**: {base_name} after a short supervised "
                          "fine-tune on human-written conversations (spec 13).")
        f["base_model_yaml"] = f"base_model: AneekC/{base_name}\n"
        f["usage_call"] = ('print(mq.chat(model, tok, [{"role": "user", "content": '
                           '"Write a Python function that reverses a string."}]))')
    else:
        f["kind_line"] = ("This is the **base model**: it continues text and code; it does not "
                          "follow instructions (the chat model is "
                          f"AneekC/{base_name}-chat).")
        f["base_model_yaml"] = ""
        f["usage_call"] = 'print(mq.generate(model, tok, "def fibonacci(n):", max_new_tokens=60))'
    return f


def render(template: str, values: dict[str, str]) -> str:
    """Replace every {{name}}; a name without a value becomes NOT_MEASURED."""
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda m: values.get(m.group(1)) or
                  ("" if m.group(1) in ("base_model_yaml", "sft_data") else NOT_MEASURED),
                  template)
