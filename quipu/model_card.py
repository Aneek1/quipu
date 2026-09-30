"""The quipu-moe model card, generated from the run's result files (spec 7, 11-13).

hf/README_moe.md is the template: {{field}} placeholders, filled by render(). Every
field is built from a file the run wrote (run config, shard manifest, A/B summary,
spend ledger, milestone metrics, code_eval results, int4 report, expert usage); a
field whose input is missing renders as NOT_MEASURED, never as a guess. The same
template serves the base model and the chat model (variant="chat"); the chat card
adds the fine-tuning data (datasets YAML, a sources table, the licence credits), from
the SFT manifest when it exists (keys read: see sft_sources) and the planned sources
of spec 13 otherwise. Limitations that depend on a measurement (the language mix, code
correctness) are built from the result files too, never hardcoded.

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
# "setting" is how each number was produced, as far as the source states it (checked
# against the source text on 2026-09-29); what the source does not state is said so.
PUBLISHED: list[dict[str, Any]] = [
    {"model": "CodeGen-350M-mono", "params": "350M",
     "humaneval_pass@1": 12.76, "humaneval_pass@10": 23.11,
     "mbpp_pass@1": None, "mbpp_pass@10": None,
     "setting": "sampled, top-p 0.95, best of T in {0.2, 0.6, 0.8} per k (Table 1 caption); "
                "HumanEval 164; 0-shot; no MBPP in that table",
     "source": "Nijkamp et al., CodeGen (ICLR 2023), arXiv:2203.13474, Table 1"},
    {"model": "SantaCoder-1.1B", "params": "1.1B",
     "humaneval_pass@1": 18.0, "humaneval_pass@10": 29.0,
     "mbpp_pass@1": 35.0, "mbpp_pass@10": 58.0,
     "setting": "sampled n=200, T=0.2 for pass@1 and T=0.8 for pass@10 (paper, sec. on "
                "evaluation); HumanEval and MBPP as MultiPL-E's Python versions, MBPP "
                "derived from the sanitized subset (paper), not necessarily our 257-problem "
                "test split",
     "source": "bigcode/santacoder model card (self-reported; Allal et al., arXiv:2301.03988)"},
    {"model": "SmolLM-135M", "params": "135M",
     "humaneval_pass@1": None, "humaneval_pass@10": None,
     "mbpp_pass@1": None, "mbpp_pass@10": None,
     "setting": "no number in the text (HumanEval shown as a chart only)",
     "source": "huggingface.co/blog/smollm; cheap to re-run on the laptop"},
    {"model": "SmolLM-360M", "params": "360M",
     "humaneval_pass@1": None, "humaneval_pass@10": None,
     "mbpp_pass@1": None, "mbpp_pass@10": None,
     "setting": "no number in the text (HumanEval shown as a chart only)",
     "source": "huggingface.co/blog/smollm; cheap to re-run on the laptop"},
    {"model": "Qwen2.5-Coder-1.5B (base)", "params": "1.5B",
     "humaneval_pass@1": 43.9, "humaneval_pass@10": None,
     "mbpp_pass@1": 69.2, "mbpp_pass@10": None,
     "setting": "unverified setting: the report does not state the decoding for this table "
                "(the word 'greedy' does not appear) nor which MBPP problems or how many "
                "shots this column uses (a separate MBPP 3-shot column exists); HumanEval "
                "(164) and MBPP are reported beside EvalPlus's HumanEval+ / MBPP+",
     "source": "Hui et al., Qwen2.5-Coder Technical Report, arXiv:2409.12186, Table 5"},
]

# Our benchmark variants (scripts/code_eval.py FILES).
OUR_VARIANTS = "HumanEval 164; MBPP sanitized test (257)"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def code_results_table(ours: list[dict[str, Any]]) -> str:
    """Markdown: our models (measured; a greedy row and, with samples, a sampled row)
    then PUBLISHED (cited, not re-run), each row with its metric / setting.
    ours: [{"model", "params", "results": {"humaneval"|"mbpp": summary}, "chat"?}]."""
    lines = [
        "| model | params | metric / setting | HumanEval pass@1 | HumanEval pass@10 | "
        "MBPP pass@1 | MBPP pass@10 | source |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for m in ours:
        r = m["results"]
        he, mb = r.get("humaneval") or {}, r.get("mbpp") or {}
        mode = "chat template" if m.get("chat") else "base, 0-shot completion"
        lines.append(f"| **{m['model']}** | {m['params']} | greedy pass@1 (T=0, one "
                     f"completion); {OUR_VARIANTS}; {mode} | {_pct(he.get('greedy_pass@1'))} | - | "
                     f"{_pct(mb.get('greedy_pass@1'))} | - | measured here |")
        n = he.get("samples") or mb.get("samples") or 0
        if n:
            t = he.get("temperature") or mb.get("temperature")
            p = he.get("top_p") or mb.get("top_p")
            t = "not recorded" if t is None else f"{t:g}"
            p = "not recorded" if p is None else f"{p:g}"
            lines.append(f"| **{m['model']}** | {m['params']} | sampled n={n}, T={t}, top-p "
                         f"{p}; pass@1 and pass@10 from the same samples (unbiased "
                         f"estimator); {OUR_VARIANTS}; {mode} | {_pct(he.get('pass@1'))} | "
                         f"{_pct(he.get('pass@10'))} | {_pct(mb.get('pass@1'))} | "
                         f"{_pct(mb.get('pass@10'))} | measured here |")
    for p in PUBLISHED:
        lines.append(f"| {p['model']} | {p['params']} | {p['setting']} | "
                     f"{_pct(p['humaneval_pass@1'])} | {_pct(p['humaneval_pass@10'])} | "
                     f"{_pct(p['mbpp_pass@1'])} | {_pct(p['mbpp_pass@10'])} | "
                     f"published, not re-run: {p['source']} |")
    return "\n".join(lines)


def protocol_md(p: dict[str, Any]) -> str:
    """The prompt formats, exact stop lists, decoding and sandbox of a code_eval run
    (scripts/code_eval.py protocol())."""
    def stops(xs: list[str]) -> str:
        return ", ".join(f"`{x!r}`" for x in xs) if xs else "none"

    prompts = p.get("prompts") or {}
    seqs = p.get("stop_sequences") or {}
    lines = [f"- Mode: {p.get('mode', '?')}.",
             f"- HumanEval prompt: {prompts.get('humaneval', '?')}.",
             f"- HumanEval stop sequences: {stops(seqs.get('humaneval') or [])}.",
             f"- MBPP prompt: {prompts.get('mbpp', '?')}.",
             f"- MBPP stop sequences: {stops(seqs.get('mbpp') or [])}.",
             f"- Stop tokens: {', '.join(p.get('stop_tokens') or []) or 'none'}; at most "
             f"{p.get('max_new_tokens', '?')} new tokens.",
             f"- Greedy: {p.get('greedy', '?')}."]
    if p.get("samples"):
        lines.append(f"- Sampled: n={p['samples']} per problem, temperature {p['temperature']:g}, "
                     f"top-p {p['top_p']:g}; {p.get('estimator', '')}.")
    else:
        lines.append("- Sampled: not run (greedy only).")
    lines.append(f"- Execution: {p.get('timeout_s', '?')} s timeout per program; "
                 f"{p.get('sandbox', '?')}.")
    return "\n".join(lines)


def revision_mismatches(revisions: dict[str, str], manifest: Path | str | None) -> list[str]:
    """Benchmarks whose evaluated commit differs from the one the shard manifest's
    decontamination used (decontamination.benchmarks[b].revision, else
    decontamination.index.revisions[b]). Empty when the manifest or the entry is absent."""
    man = _json(Path(manifest)) if manifest is not None else None
    d = (man or {}).get("decontamination")
    if not isinstance(d, dict):
        return []
    bench = d.get("benchmarks") if isinstance(d.get("benchmarks"), dict) else {}
    index = ((d.get("index") or {}).get("revisions") or {}) if isinstance(d.get("index"), dict) else {}
    out = []
    for name, rev in revisions.items():
        info = bench.get(name)
        theirs = info.get("revision") if isinstance(info, dict) else None
        theirs = theirs or index.get(name)
        if theirs and rev and theirs != rev:
            out.append(f"{name}: evaluated at {rev[:12]}, decontaminated against {theirs[:12]}")
    return out


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
    sft_dir: Path | None = None          # results/sft: chat_samples.json, val_by_source.json
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
    train = ((run_cfg or {}).get("train") or {}) if run_cfg else {}
    optimizer = train.get("optimizer", cfg.train.optimizer)
    per_head = train.get("muon_per_head", getattr(cfg.train, "muon_per_head", True))
    # The optimizer, activation, depth mixing and precision are A/B pairs: final only
    # once the run config (written after the A/B step) exists.
    decided = ("per the A/B winners, results/moe/run_config.toml" if run_cfg
               else "config default; A/B not yet decided")
    opt = ((f"Muon (Newton-Schulz orthogonalised "
            f"{'per attention head' if per_head else 'per whole matrix'}) for weight "
            "matrices, AdamW for the rest") if optimizer == "muon" else "AdamW")
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
        ("Expert activation", f"{act} ({decided})"),
        ("Depth mixing", f"{depth} ({decided})"),
        ("Optimizer", f"{opt} ({decided})"),
        ("Training precision", f"{precision} (router and balancing math in fp32; {decided})"),
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
            cost = (f"${spent:.2f} from the spend ledger: GPU session (hourly); excludes "
                    "Vast bandwidth charges. The GPU box's hours (setup, A/B runs, the full "
                    "run, the chat fine-tune, evaluation) plus the CPU shard-build box")
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
        train_tokens = ((man.get("splits") or {}).get("train") or {}).get("tokens")
        lines += ["", "Training tokens in the shard set: "
                  + (f"{train_tokens:,}" if isinstance(train_tokens, int) else NOT_MEASURED)
                  + " (the train split; validation and evaluation splits are separate)."]
    else:
        lines += [f"Planned mix (from the config; the shard manifest is {NOT_MEASURED}): "
                  f"{cfg.data.code_share:.0%} code, the rest text "
                  f"({', '.join(f'{k} {v:.3f}' for k, v in cfg.data.text_language_weights.items())} "
                  "of text)."]
    lines += ["", "Code licences kept: " + ", ".join(cfg.data.code_licenses) + "."]
    return "\n".join(lines)


def _benchmark_revisions(cfg) -> dict[str, str]:
    return {"humaneval": getattr(cfg.data, "humaneval_revision", ""),
            "mbpp": getattr(cfg.data, "mbpp_revision", "")}


def _decontam(inp: CardInputs, cfg) -> str:
    man = _json(inp.manifest)
    d = (man or {}).get("decontamination")
    if not isinstance(d, dict) or not d.get("dropped"):
        return NOT_MEASURED
    lines = ["| split | " + " | ".join(("humaneval", "mbpp")) + " |", "|---|---:|---:|"]
    for split, counts in d["dropped"].items():
        counts = counts or {}
        lines.append(f"| {split} | {counts.get('humaneval', 0):,} | {counts.get('mbpp', 0):,} |")
    rule = ((d.get("index") or {}).get("rule") if isinstance(d.get("index"), dict) else None) \
        or "whitespace-collapsed substring or 13-gram overlap, quipu/decontam.py"
    out = ("Documents dropped because they contain a HumanEval or MBPP problem (rule, from "
           f"the shard manifest: {rule}):\n\n" + "\n".join(lines))
    bad = revision_mismatches(_benchmark_revisions(cfg), inp.manifest)
    if bad:
        out = ("**Revision mismatch:** the evaluated benchmark commits differ from the ones "
               "the training data was decontaminated against (" + "; ".join(bad) + "), so "
               "the HumanEval / MBPP numbers may include leaked problems.\n\n" + out)
    return out


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
            "per language and held-out code. Definition: summed next-token loss in bits over "
            "the UTF-8 bytes of the target tokens. Every <|endoftext|> document separator is a "
            "target too: its loss counts but it is 0 bytes of text, so these numbers are "
            "slightly higher (more pessimistic) than a separator-free measure.\n\n"
            + "\n".join(lines))


def _code_json(inp: CardInputs, name: str) -> dict | None:
    data = _json(inp.code_eval_dir / f"{name}.json") if inp.code_eval_dir else None
    return data if data and data.get("results") else None


def _partial(data: dict) -> bool:
    return any(b.get("evaluated", 0) < b.get("problems", 0)
               for b in (data.get("benchmarks") or {}).values())


def _code_eval(inp: CardInputs, names: list[str], params: str) -> str:
    ours, protocols = [], []
    for name in names:
        data = _code_json(inp, name)
        if data:
            ours.append({"model": name + (" (partial run)" if _partial(data) else ""),
                         "params": params, "results": data["results"],
                         "chat": bool(data.get("chat"))})
            if data.get("protocol"):
                protocols.append((name, data["protocol"]))
    if not ours:
        return NOT_MEASURED + " (published numbers of reference models, for when it is:)\n\n" \
            + code_results_table([])
    out = [code_results_table(ours), "",
           "Published numbers are copied from their sources with the setting each source "
           "states; they were produced with other prompts, decoding and harnesses, so compare "
           "them with care (a row's setting says what differs)."]
    for name, p in protocols:
        out += ["", f"**Protocol of {name}** (from results/code_eval/{name}.json)", "",
                protocol_md(p)]
    if not protocols:
        out += ["", f"Prompt format and stop lists: {NOT_MEASURED} (not recorded in the result "
                "file; re-run scripts/code_eval.py)."]
    return "\n".join(out)


def _code_limitation(inp: CardInputs, name: str) -> str:
    data = _code_json(inp, name)
    r = (data or {}).get("results") or {}
    parts = [f"{v:.1f}% of {label}" for label, key in (("HumanEval", "humaneval"), ("MBPP", "mbpp"))
             if isinstance(v := (r.get(key) or {}).get("greedy_pass@1"), (int, float))]
    if not parts:
        return (f"- Code correctness: {NOT_MEASURED} (HumanEval / MBPP above); treat generated "
                "code as untested.")
    partial = " (a partial run)" if data and _partial(data) else ""
    return (f"- Code: greedy decoding solves {' and '.join(parts)} problems{partial} (see "
            "above); treat generated code as untested.")


def _mix_limitation(inp: CardInputs) -> str:
    man = _json(inp.manifest)
    a = ((man or {}).get("mix") or {}).get("achieved") or {}
    other, eng, code = a.get("other_languages"), a.get(evalsets.ENGLISH), a.get("code")
    if all(isinstance(v, (int, float)) for v in (other, eng, code)):
        return (f"- The nine languages other than English together are {other:.1%} of the "
                f"training tokens (English {eng:.1%}, code {code:.1%}; shard manifest, mix "
                "achieved). Expect weaker output in them than in English, and some mixing of "
                "languages; the bits per byte above show how much weaker, per language.")
    return (f"- Language mix: {NOT_MEASURED} (no shard manifest with the achieved mix). The "
            "nine languages other than English are a minority of the training text; expect "
            "weaker output in them than in English.")


EXPERTS_LIMITATION = (
    "- Expert specialisation by programming language (Python vs JavaScript / JSX, spec 7) "
    "was not measured: the code validation split carries no per-document language label, "
    "so code is one split in the expert-usage tables.")


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
    tokens = rep.get("bpb_tokens") or {}

    def num(v, fmt=".4f"):
        return "-" if not isinstance(v, (int, float)) else format(v, fmt)

    if q:
        lines += ["", "| split | tokens | fp32 bpb | bf16 bpb | int4 bpb | int4 vs bf16 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for k, v in q.items():
            base = v.get("bf16")
            delta = ("-" if not isinstance(base, (int, float)) else
                     f"{v['int4'] - base:+.4f} ({v['int4'] / base - 1:+.2%})")
            tok = tokens.get(k, rep.get("bpb_tokens_per_split"))
            lines.append(f"| {evalsets.LANGUAGES.get(k, k)} | {num(tok, ',')} | "
                         f"{num(v.get('fp32'))} | {num(base)} | {num(v.get('int4'))} | {delta} |")
        lines.append("\nThe first tokens of each evaluation split (count per row). bf16 = the "
                     "weights rounded to bf16, arithmetic in fp32 (spec 12's baseline); fp32 = "
                     "the trained weights as saved.")
    ce = rep.get("code_eval")
    if ce:
        lines.append(f"\nHumanEval greedy pass@1 on the first {ce['problems']} problems: "
                     + ", ".join(f"{k} {ce[k]:.1f}" for k in ("fp32", "bf16", "int4") if k in ce)
                     + ".")
    par = rep.get("parity") or {}
    if par:
        lines.append("\nStandalone-loader parity (max |logit difference| against the training "
                     "code): " + "; ".join(f"{k} {v:.1e}" for k, v in par.items()) + ".")
    lines.append(f"\nStepbuild score, int4 vs bf16: {NOT_MEASURED} (needs the chat/step fine-tune).")
    return "\n".join(lines)


def _experts(inp: CardInputs) -> str:
    usage = _json(inp.experts_dir / "usage.json") if inp.experts_dir else None
    if not usage:
        return NOT_MEASURED
    lines = ["| split | dead experts | overloaded experts |", "|---|---:|---:|"]
    for k, u in usage["splits"].items():
        lines.append(f"| {evalsets.LANGUAGES.get(k, k)} | {len(u['dead'])} | {len(u['overloaded'])} |")
    return (f"Routing frequency per expert, per layer and language ({usage['n_layer']} layers x "
            f"{usage['n_experts']} experts): the full tables and a heatmap are in the repository "
            f"(results/experts). Here, dead: < {usage['dead_below_x_target']:.0%} of the balanced "
            f"target over the evaluated tokens; overloaded: > "
            f"{usage['overloaded_above_x_target']:.0%}. These are the trainer's health-alert "
            "bounds (quipu.train MOE_HEALTH_LOW / MOE_HEALTH_HIGH), applied to one evaluation "
            "pass; the training log's own \"dead\" count is stricter (experts with zero "
            "assignments in a logging interval), and its health alert needs a sustained window "
            "of steps out of bounds.\n\n" + "\n".join(lines)
            + "\n\n" + EXPERTS_LIMITATION[2:])


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


def _held_out_table(held: Any) -> str | None:
    """The SFT's held-out chat loss per source (quipu.train write_val_by_source)."""
    sources = (held or {}).get("sources") if isinstance(held, dict) else None
    if not isinstance(sources, dict) or not sources:
        return None
    lines = ["| source | held-out loss (nats per assistant token) | assistant tokens |",
             "|---|---:|---:|"]
    for name, v in sources.items():
        if not isinstance(v, dict) or not isinstance(v.get("loss"), (int, float)):
            continue
        n = v.get("assistant_tokens")
        lines.append(f"| {name} | {v['loss']:.4f} | "
                     f"{f'{n:,}' if isinstance(n, int) else '-'} |")
    return "\n".join(lines) if len(lines) > 2 else None


def _chat_samples(inp: CardInputs) -> str:
    """The chat card's samples: scripts/chat_eval.py's chat_samples.json (two fixed
    prompts per language, the chat template, temperature 0, from the CHAT model) and
    the SFT's held-out loss per source (from that file, else val_by_source.json).
    Never the base model's pretraining samples; NOT_MEASURED for whatever is missing."""
    sft = Path(inp.sft_dir) if inp.sft_dir else None
    data = _json(sft / "chat_samples.json") if sft else None
    held = (data or {}).get("val_by_source") if isinstance(data, dict) else None
    if not held and sft:
        held = _json(sft / "val_by_source.json")
    table = _held_out_table(held)
    parts = ["Held-out chat loss per source (the SFT's validation conversations, "
             "micro-batch 1): " + ("" if table else NOT_MEASURED)]
    if table:
        parts[0] = parts[0].rstrip(": ") + ":\n\n" + table
    samples = (data or {}).get("samples") if isinstance(data, dict) else None
    if not isinstance(samples, dict) or not samples:
        parts.append(f"Chat samples: {NOT_MEASURED} (scripts/chat_eval.py writes them).")
        text = "\n\n".join(parts)
        return text if table else NOT_MEASURED + " " + text
    dec = (data or {}).get("decoding") or {}
    cap = dec.get("max_new_tokens")
    lines = ["Replies of the chat model to two fixed prompts per language, one user turn in "
             "the chat template, temperature 0 (greedy), unedited: good and bad alike."
             + (f" A reply marked (cut) hit the {cap}-token cap." if cap else ""), ""]
    for lang, rows in samples.items():
        for r in rows or []:
            reply = str(r.get("reply", "")).replace("```", "'''")
            if len(reply) > 600:
                reply = reply[:600] + " ..."
            cut = r.get("finish") == "length"
            lines += [f"**{lang}: {str(r.get('prompt', '')).strip()}**"
                      + (f" (cut at {cap} tokens)" if cut and cap else " (cut)" if cut else ""),
                      "", "```", reply, "```", ""]
    parts.append("\n".join(lines).strip())
    return "\n\n".join(parts)


# The chat fine-tune's sources (spec 13) when the SFT manifest is not there yet:
# (key, dataset, licence, the config field holding its pinned revision).
SFT_DEFAULTS = [
    ("aya", "CohereForAI/aya_dataset", "Apache-2.0", "aya_revision"),
    ("oasst2", "OpenAssistant/oasst2", "Apache-2.0", "oasst2_revision"),
    ("stepbuild", "stepbuild train split (this project)",
     "per repository (permissive; each example keeps its repository's licence)", None),
]
STEPBUILD_CREDIT = ("the stepbuild train split (FILE-block replies mined from permissively "
                    "licensed public repositories, each keeping its repository's licence; "
                    "built for this project, its test split and benchmark apps held out)")


def sft_sources(inp: CardInputs, cfg) -> list[dict[str, Any]]:
    """The chat fine-tune's sources, from the SFT manifest (M12, scripts/build_chat_data.py)
    when there is one, else SFT_DEFAULTS. Manifest keys read, all optional:
        sources: {name: {...}} or [{"name": ..., ...}], each with
            dataset (or name), revision, license / licence,
            examples (or kept.examples / kept.conversations), tokens (or kept.tokens).
    Returns [{"name", "dataset", "revision", "licence", "examples", "tokens"}]."""
    man = _json(inp.sft_manifest)
    raw = (man or {}).get("sources") if isinstance(man, dict) else None
    items: list[tuple[str, dict]] = []
    if isinstance(raw, dict):
        items = [(str(k), v if isinstance(v, dict) else {}) for k, v in raw.items()]
    elif isinstance(raw, list):
        items = [(str(v.get("name", "?")), v) for v in raw if isinstance(v, dict)]
    if not items:
        return [{"name": key, "dataset": ds, "licence": lic,
                 "revision": getattr(cfg.data, rev, "") if rev else "",
                 "examples": None, "tokens": None, "from_manifest": False}
                for key, ds, lic, rev in SFT_DEFAULTS]
    defaults = {k: (ds, lic) for k, ds, lic, _ in SFT_DEFAULTS}
    out = []
    for name, s in items:
        kept = s.get("kept") if isinstance(s.get("kept"), dict) else {}
        examples = s.get("examples", kept.get("examples", kept.get("conversations")))
        tokens = s.get("tokens", kept.get("tokens"))
        out.append({"name": name,
                    "dataset": s.get("dataset") or defaults.get(name, ("-",))[0],
                    "revision": s.get("revision") or "",
                    "licence": s.get("license") or s.get("licence")
                    or defaults.get(name, (None, "-"))[1],
                    "examples": examples if isinstance(examples, int) else None,
                    "tokens": tokens if isinstance(tokens, int) else None,
                    "from_manifest": True})
    return out


def _sft(inp: CardInputs, cfg) -> str:
    srcs = sft_sources(inp, cfg)
    have = any(s["from_manifest"] for s in srcs)
    lines = ["### Chat fine-tuning data", "",
             "Human-written conversations only (spec 13)"
             + ("; from the SFT data manifest:" if have else
                f"; planned sources (the SFT data manifest is {NOT_MEASURED}):"), "",
             "| source | dataset | revision | licence | examples | tokens |",
             "|---|---|---|---|---:|---:|"]

    def n(v):
        return "-" if v is None else f"{v:,}"

    for s in srcs:
        rev = f"`{s['revision'][:12]}`" if s["revision"] else "-"
        lines.append(f"| {s['name']} | {s['dataset']} | {rev} | {s['licence']} | "
                     f"{n(s['examples'])} | {n(s['tokens'])} |")
    return "\n".join(lines)


def _hf_datasets(srcs: list[dict[str, Any]]) -> list[str]:
    return [s["dataset"] for s in srcs if re.fullmatch(r"[\w.-]+/[\w.-]+", s["dataset"] or "")]


def _sft_credits(srcs: list[dict[str, Any]]) -> str:
    hf = [s for s in srcs if s["dataset"] in _hf_datasets(srcs)]
    parts = [f"[{s['dataset']}](https://huggingface.co/datasets/{s['dataset']}) ({s['licence']})"
             for s in hf]
    if any(s["name"] == "stepbuild" for s in srcs):
        parts.append(STEPBUILD_CREDIT)
    return (" Chat fine-tuning data: " + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "")
            + parts[-1] + ".") if parts else ""


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
        "data": _data(inp, cfg), "decontamination": _decontam(inp, cfg), "lid": _lid(inp, cfg),
        "lid_model": cfg.data.lid_model,
        "bpb": _bpb(inp), "code_eval": _code_eval(inp, [base_name, base_name + "-chat"]
                                                  if variant == "chat" else [base_name], params),
        "int4": _int4(inp), "experts": _experts(inp),
        # The chat card shows the chat model's replies, never the base model's
        # pretraining samples.
        "samples": _chat_samples(inp) if variant == "chat" else _samples(inp),
        "samples_title": "Chat samples and held-out chat loss" if variant == "chat"
        else "Samples",
        "mix_limitation": _mix_limitation(inp),
        "code_limitation": _code_limitation(inp, name),
        "experts_limitation": EXPERTS_LIMITATION,
        "sft_data": "", "datasets_yaml": "", "sft_credits": "", "chat_limitation": "",
    }
    if variant == "chat":
        srcs = sft_sources(inp, cfg)
        f["sft_data"] = _sft(inp, cfg)
        f["datasets_yaml"] = "".join(f"- {d}\n" for d in _hf_datasets(srcs))
        f["sft_credits"] = _sft_credits(srcs)
        f["chat_limitation"] = (
            "- This chat model had one short supervised fine-tune (data above). How well it "
            "follows instructions was not measured beyond the held-out chat loss per source, "
            "the fixed-prompt replies and the chat-mode HumanEval / MBPP numbers above.")
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
                  ("" if m.group(1) in OPTIONAL_FIELDS else NOT_MEASURED),
                  template)


# Fields that are empty on purpose for one variant (the chat-only parts).
OPTIONAL_FIELDS = ("base_model_yaml", "sft_data", "datasets_yaml", "sft_credits",
                   "chat_limitation")
