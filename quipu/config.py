"""One TOML file fully determines a run. Everything derived is computed here, once,
so no two call sites can disagree about how many steps there are.

steps floors total_tokens / batch_tokens, so up to batch_tokens-1 tokens are unused
(24,064 for the shipped config: 3,000,000,000 - 5,722 x 524,288)."""
from __future__ import annotations

import dataclasses
import math
import tomllib
from pathlib import Path
from typing import Any

TOP_LEVEL_KEYS = {"name", "model", "data", "train"}


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    ffn_hidden: int
    context: int
    rope_base: float
    norm_eps: float
    # quipu-moe (spec section 3). Every default below is the dense quipu-114m model,
    # so configs written before these fields load unchanged. For kind "moe" the
    # feed-forward is shared_experts always-on experts of shared_hidden plus
    # n_experts routed experts of expert_hidden, top_k of them per token; ffn_hidden
    # is then unused. The MoE-only int fields may be 0 (they must be, for "dense").
    kind: str = "dense"                  # "dense" | "moe"
    n_experts: int = 0
    top_k: int = 0
    expert_hidden: int = 0
    shared_experts: int = 0
    shared_hidden: int = 0
    activation: str = "swiglu"           # "swiglu" | "situ_glu" (situ_glu: moe only)
    situ_beta_gate: float = 4.0          # SiTU-GLU beta_1 (gate tanh bound)
    situ_beta_up: float = 25.0           # SiTU-GLU beta_2 (up tanh bound)
    attnres_blocks: int = 0              # Block Attention Residuals; 0 = plain residual
    attnres_checkpoint: bool = False     # recompute the AttnRes depth mix in backward (CPU: +14% time, ~no memory saved)
    balance_update_rate: float = 0.3     # Quantile Balancing bias EMA rate
    # Routed-expert dispatch: "loop" runs each expert on its own contiguous slice;
    # "padded" pads every slice to capacity ceil(capacity_factor * T * top_k / n)
    # and runs all experts in one batched matmul, dropping overflow tokens.
    moe_dispatch: str = "loop"           # "loop" | "padded"
    capacity_factor: float = 1.5         # "padded" only

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclasses.dataclass(frozen=True)
class DataConfig:
    dataset: str
    subset: str
    shard_dir: str
    shard_tokens: int
    val_tokens: int
    # Code mix (weekend-run spec section 3). Train reads code parquet files
    # [0, code_heldout_first_file); code val reads [code_heldout_first_file,
    # code_files_total), which train never opens.
    code_dataset: str
    code_share: float
    code_languages: tuple[str, ...]
    code_licenses: tuple[str, ...]
    html_cap: float
    code_val_tokens: int
    code_heldout_first_file: int
    code_files_total: int
    # Code documents longer than this many tokens are skipped (vendored bundles,
    # data blobs, generated files). Defaulted so configs written before it load.
    code_max_doc_tokens: int = 16_000
    # Tokenizer: "gpt2" (tiktoken, quipu-114m) or a path to a tokenizer.json from
    # scripts/train_tokenizer.py, relative to the working directory like shard_dir.
    # When the file exists at load time its vocabulary must equal model.vocab_size.
    tokenizer: str = "gpt2"
    # Share of code tokens per language, keyed by github-code-clean's labels; empty
    # means the old behaviour (every code_languages entry as it comes). When given,
    # the keys must be exactly code_languages and the weights sum to 1.
    code_language_weights: dict[str, float] = dataclasses.field(default_factory=dict)
    # Share of text tokens per language (spec section 11). "eng_Latn" is the
    # English source (dataset/subset above, FineWeb-Edu); every other key is a
    # FineWeb-2 subset name. cmn_Hani is one source here: the zh-Hans/zh-Hant split
    # happens when shards are built. Empty means English-only text, as before.
    text_language_weights: dict[str, float] = dataclasses.field(default_factory=dict)
    # The language-ID model that filters text documents at shard-building time
    # (Hugging Face repo id), and the revision used; "" until a build pins it.
    lid_model: str = "AneekC/lid-specialists-9plus1"
    lid_revision: str = ""
    # Data v2: the dataset commits the shard build reads (code_dataset, the English
    # dataset above, FineWeb-2), full 40-hex shas. "" = whatever the Hub's main
    # branch is at build time (the build prints a warning); the quipu-moe configs pin
    # the commits the tokenizer and its gate were built from.
    code_revision: str = ""
    text_revision: str = ""
    fineweb2_revision: str = ""
    # Data v2: the benchmark commits the shard build decontaminates against
    # (quipu/decontam.py: openai/openai_humaneval, google-research-datasets/mbpp
    # "sanitized"); "" = the Hub's current commit (the build prints a warning).
    humaneval_revision: str = ""
    mbpp_revision: str = ""


ENGLISH_TEXT_KEY = "eng_Latn"
MODEL_KINDS = ("dense", "moe")
ACTIVATIONS = ("swiglu", "situ_glu")
MOE_DISPATCHES = ("loop", "padded")
OPTIMIZERS = ("adamw", "muon")
PRECISIONS = ("bf16", "fp8")
GPT2_TOKENIZER = "gpt2"
# ModelConfig int fields that are 0 for a dense model.
MOE_INT_FIELDS = ("n_experts", "top_k", "expert_hidden", "shared_experts", "shared_hidden",
                  "attnres_blocks")
WEIGHT_SUM_TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    total_tokens: int
    batch_tokens: int
    micro_batch: int
    context: int
    lr: float
    lr_min: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    seed: int
    ckpt_dir: str
    ckpt_every: int
    ckpt_keep: int
    eval_every: int
    eval_batches: int
    # Steps at which bf16 weights are kept for post-run evaluation (the final step
    # is always kept too). Separate from ckpt_*: never pruned, never resumed from.
    milestones: tuple[int, ...] = ()
    # Optimizer (spec section 6.1). "muon": per-head Muon for 2-D weight matrices
    # except embeddings, AdamW (lr, betas, weight_decay above) for the rest.
    optimizer: str = "adamw"             # "adamw" | "muon"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_per_head: bool = True
    # Decoupled weight decay for the Muon groups only (AdamW keeps weight_decay).
    # The per-step shrink is lr * wd: AdamW at lr 6e-4, wd 0.1 shrinks by 6e-5 a
    # step, while Muon at muon_lr 0.02 with that same 0.1 would shrink by 2e-3,
    # 33x as much, and the Muon-vs-AdamW A/B would partly measure decay strength.
    # 0.01 gives 2e-4 a step, ~3x AdamW's rather than 33x: not matched exactly,
    # only the same order of magnitude; nothing here was tuned. For reference,
    # Keller Jordan's Muon (Keller-scale lr ~0.02) shipped with no weight decay;
    # Moonlight's wd 0.1 goes with an update rescaled to AdamW's RMS, i.e. a much
    # smaller effective lr, so its 0.1 does not carry over to this lr.
    muon_weight_decay: float = 0.01
    compile: bool = False                # torch.compile the model
    # Matmul precision (spec section 12). "bf16": bf16 autocast, as quipu-114m.
    # "fp8": additionally, attention q/k/v/o and the shared-expert linears run their
    # matmuls in FP8 (quipu.fp8); kind "moe" on CUDA only. Kept for the full run only
    # if A/B pair 4 shows >= 1.2x tokens/s with loss within seed noise.
    precision: str = "bf16"              # "bf16" | "fp8"
    # Spend guard (spec section 6.4): stop cleanly once elapsed hours x usd_per_hour
    # reaches budget_usd. budget_usd 0 = no guard.
    budget_usd: float = 0.0
    usd_per_hour: float = 0.0

    @property
    def steps(self) -> int:
        return self.total_tokens // self.batch_tokens

    @property
    def grad_accum(self) -> int:
        return self.batch_tokens // (self.micro_batch * self.context)


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig


# Tables that are one value, not a section: an override replaces the whole table.
# Merged key by key, an override could never drop a language, and a partial
# override would silently keep the file's other weights.
WHOLE_VALUE_TABLES = frozenset({"code_language_weights", "text_language_weights"})


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if key in WHOLE_VALUE_TABLES:
            out[key] = value
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _validate_int_fields(instance: Any, allow_zero: frozenset[str] = frozenset()) -> None:
    """Every int-annotated field must actually be an int (not bool, not float) and
    strictly positive, so a typo or a 0 doesn't silently reach a division below.
    Fields named in allow_zero may additionally be 0 (e.g. seed)."""
    for f in dataclasses.fields(instance):
        # f.type is the raw annotation; it's the string "int" under `from __future__
        # import annotations` and the type int otherwise, so cover both.
        if f.type not in (int, "int"):
            continue
        value = getattr(instance, f.name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{f.name} must be an int, got {value!r}")
        minimum = 0 if f.name in allow_zero else 1
        if value < minimum:
            raise ValueError(f"{f.name} must be positive, got {value!r}")


def _check_fraction(name: str, value: Any, *, allow_one: bool) -> None:
    """A share strictly inside (0, 1), or (0, 1] when allow_one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    upper_ok = value <= 1 if allow_one else value < 1
    if not (value > 0 and upper_ok):
        interval = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{name} must be in {interval}, got {value!r}")


def _check_str_list(name: str, value: Any) -> None:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list, got {value!r}")
    if not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{name} entries must be non-empty strings, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} has duplicate entries: {value!r}")


def _check_milestones(value: Any, steps: int) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"milestones must be a list of ints, got {value!r}")
    for m in value:
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError(f"milestones must be positive ints, got {m!r} in {value!r}")
    if any(a >= b for a, b in zip(value, value[1:])):
        raise ValueError(f"milestones must be strictly increasing, got {list(value)!r}")
    if value and value[-1] >= steps:
        raise ValueError(
            f"milestones must all be less than steps ({steps}), got {list(value)!r}; "
            "the final step is always kept as a milestone anyway"
        )


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _check_choice(name: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def _check_positive(name: str, value: Any, *, allow_zero: bool = False) -> None:
    if not _is_number(value) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}")


def _check_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")


def _check_str(name: str, value: Any, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a {'' if allow_empty else 'non-empty '}string, "
                         f"got {value!r}")


def _check_weights(name: str, value: Any) -> dict[str, float]:
    """A {language: weight} table: string keys, weights in (0, 1], summing to 1
    within WEIGHT_SUM_TOLERANCE. Empty is allowed (the caller decides what it means).
    Returns a copy, so the config never shares the parsed TOML's dict."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table of language = weight, got {value!r}")
    for key, weight in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings, got {key!r}")
        _check_fraction(f"{name}[{key!r}]", weight, allow_one=True)
    if value:
        total = sum(value.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"{name} must sum to 1, got {total!r}")
    return dict(value)


def _check_moe(model: ModelConfig) -> None:
    _check_choice("kind", model.kind, MODEL_KINDS)
    _check_choice("activation", model.activation, ACTIVATIONS)
    _check_positive("situ_beta_gate", model.situ_beta_gate)
    _check_positive("situ_beta_up", model.situ_beta_up)
    _check_fraction("balance_update_rate", model.balance_update_rate, allow_one=True)
    _check_choice("moe_dispatch", model.moe_dispatch, MOE_DISPATCHES)
    _check_bool("attnres_checkpoint", model.attnres_checkpoint)
    _check_positive("capacity_factor", model.capacity_factor)
    if model.capacity_factor < 1:
        # Below 1 tokens are dropped even under perfect balance.
        raise ValueError(f"capacity_factor must be >= 1, got {model.capacity_factor!r}")
    if model.kind == "dense":
        # The dense model has no experts and no AttnRes; a non-zero value here
        # would be silently ignored, so it is refused instead.
        set_fields = [f for f in MOE_INT_FIELDS if getattr(model, f) != 0]
        if set_fields:
            raise ValueError(f"kind 'dense' does not use {set_fields}; set them to 0 "
                             "or use kind 'moe'")
        if model.activation != "swiglu":
            raise ValueError(f"kind 'dense' is SwiGLU only; activation "
                             f"{model.activation!r} applies to experts (kind 'moe')")
        if model.moe_dispatch != "loop":
            raise ValueError(f"kind 'dense' has no experts to dispatch; moe_dispatch "
                             f"{model.moe_dispatch!r} applies to kind 'moe'")
        return
    # top_k >= 2: routing weights are renormalised over the chosen experts, so with
    # one expert the weight is identically 1 and the router never gets a gradient.
    if not model.n_experts >= model.top_k >= 2:
        raise ValueError(f"kind 'moe' needs n_experts >= top_k >= 2, got n_experts "
                         f"{model.n_experts}, top_k {model.top_k}")
    if model.expert_hidden < 1:
        raise ValueError(f"kind 'moe' needs expert_hidden > 0, got {model.expert_hidden}")
    if (model.shared_experts > 0) != (model.shared_hidden > 0):
        raise ValueError(f"shared_experts ({model.shared_experts}) and shared_hidden "
                         f"({model.shared_hidden}) must both be 0 or both be positive")
    if model.attnres_blocks and model.n_layer % model.attnres_blocks != 0:
        raise ValueError(f"n_layer ({model.n_layer}) must divide evenly into "
                         f"attnres_blocks ({model.attnres_blocks})")


def _check_tokenizer(data: DataConfig, model: ModelConfig) -> None:
    """"gpt2" is taken as given. A path is checked against model.vocab_size only
    when the file exists: configs are written before their tokenizer is trained."""
    _check_str("tokenizer", data.tokenizer)
    if data.tokenizer == GPT2_TOKENIZER or not Path(data.tokenizer).is_file():
        return
    from quipu.bpe import BPETokenizer  # the `tokenizers` import only when needed

    actual = BPETokenizer(data.tokenizer).vocab_size
    if actual != model.vocab_size:
        raise ValueError(f"vocab_size ({model.vocab_size}) does not match the tokenizer "
                         f"{data.tokenizer} ({actual} tokens)")


def _check_data_mix(data: DataConfig) -> None:
    weights = data.code_language_weights
    if weights:
        if set(weights) != set(data.code_languages):
            raise ValueError(
                f"code_language_weights keys must be exactly code_languages; missing "
                f"{sorted(set(data.code_languages) - set(weights))}, extra "
                f"{sorted(set(weights) - set(data.code_languages))}")
        if weights.get("HTML", 0.0) > data.html_cap:
            raise ValueError(f"code_language_weights['HTML'] ({weights['HTML']}) is above "
                             f"html_cap ({data.html_cap})")
    if data.text_language_weights and ENGLISH_TEXT_KEY not in data.text_language_weights:
        raise ValueError(f"text_language_weights must include {ENGLISH_TEXT_KEY!r} "
                         f"(the {data.dataset} share)")
    _check_str("lid_model", data.lid_model)
    _check_str("lid_revision", data.lid_revision, allow_empty=True)
    for name in ("code_revision", "text_revision", "fineweb2_revision", "humaneval_revision",
                 "mbpp_revision"):
        _check_str(name, getattr(data, name), allow_empty=True)


def _check_train_extras(train: TrainConfig) -> None:
    _check_choice("optimizer", train.optimizer, OPTIMIZERS)
    _check_positive("muon_lr", train.muon_lr)
    if not (_is_number(train.muon_momentum) and 0 <= train.muon_momentum < 1):
        raise ValueError(f"muon_momentum must be in [0, 1), got {train.muon_momentum!r}")
    _check_bool("muon_per_head", train.muon_per_head)
    _check_positive("muon_weight_decay", train.muon_weight_decay, allow_zero=True)
    _check_bool("compile", train.compile)
    _check_choice("precision", train.precision, PRECISIONS)
    _check_positive("budget_usd", train.budget_usd, allow_zero=True)
    _check_positive("usd_per_hour", train.usd_per_hour, allow_zero=True)
    if train.budget_usd > 0 and train.usd_per_hour == 0:
        raise ValueError("budget_usd needs usd_per_hour > 0 to be enforced")


OVERRIDE_SECTIONS = {"model": ModelConfig, "data": DataConfig, "train": TrainConfig}


def _override_value(key: str, annotation: str, text: str) -> Any:
    """One --override value, typed by its dataclass field's annotation (a string
    under `from __future__ import annotations`). str fields take the text as it is
    (so a Windows path needs no quoting; a TOML-quoted string is unquoted); every
    other type is parsed as a TOML value and must have that type."""
    base = annotation.split("[", 1)[0].strip()
    if base == "str":
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            return tomllib.loads(f"v = {text}")["v"]
        return text
    try:
        value = tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"override {key}: {text!r} is not a valid {base} value") from exc
    ok = {
        "int": not isinstance(value, bool) and isinstance(value, int),
        "float": _is_number(value),
        "bool": isinstance(value, bool),
        "tuple": isinstance(value, list),
        "list": isinstance(value, list),
        "dict": isinstance(value, dict),
    }.get(base)
    if not ok:
        raise ValueError(f"override {key}: {text!r} is not a valid {base} value")
    return float(value) if base == "float" else value


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """`--override` flags ("section.key=value", or "name=value") → the nested dict
    load_config takes. Each value is typed by its dataclass field (ModelConfig,
    DataConfig or TrainConfig): "train.lr=1.2e-3" is a float, "train.seed=7" an int,
    "train.compile=false" a bool, "train.milestones=[]" a list, "model.activation=
    situ_glu" a string. An unknown section or key, or a value of the wrong type, is
    a ValueError; the rest of the validation is load_config's, as for the TOML."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, text = item.partition("=")
        key, text = key.strip(), text.strip()
        if not sep or not key:
            raise ValueError(f"override {item!r} must look like section.key=value")
        if key == "name":
            out["name"] = _override_value(key, "str", text)
            continue
        parts = key.split(".")
        if len(parts) != 2 or parts[0] not in OVERRIDE_SECTIONS:
            raise ValueError(
                f"override {key!r} must be section.key with section one of "
                f"{sorted(OVERRIDE_SECTIONS)} (or 'name')"
            )
        section, field = parts
        fields = {f.name: f.type for f in dataclasses.fields(OVERRIDE_SECTIONS[section])}
        if field not in fields:
            raise ValueError(f"override {key!r}: [{section}] has no field {field!r}")
        annotation = fields[field] if isinstance(fields[field], str) else fields[field].__name__
        out.setdefault(section, {})[field] = _override_value(key, annotation, text)
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if overrides:
        raw = _merge(raw, overrides)

    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown top-level key(s): {sorted(unknown)}")

    if "context" in raw.get("train", {}):
        raise ValueError("context belongs in [model]; [train] inherits it")

    model = ModelConfig(**raw["model"])
    data_raw = dict(raw["data"])
    for key in ("code_languages", "code_licenses"):
        if key in data_raw:
            _check_str_list(key, data_raw[key])
            data_raw[key] = tuple(data_raw[key])  # frozen config, immutable lists
    for key in ("code_language_weights", "text_language_weights"):
        if key in data_raw:
            data_raw[key] = _check_weights(key, data_raw[key])
    data = DataConfig(**data_raw)
    train_raw = dict(raw["train"])
    if "milestones" in train_raw:
        # Shape/type first, so the tuple() below cannot fail on a non-list; the
        # comparison with steps waits until steps is known to be valid.
        milestones = train_raw["milestones"]
        if not isinstance(milestones, (list, tuple)):
            raise ValueError(f"milestones must be a list of ints, got {milestones!r}")
        train_raw["milestones"] = tuple(milestones)
    train = TrainConfig(context=model.context, **train_raw)

    # Type/positivity checks must run before any division below (e.g. grad_accum's
    # micro_batch * context), so a bad value raises a clear ValueError instead of a
    # ZeroDivisionError or a silently-wrong result. seed may be 0; everything else in
    # TrainConfig, including warmup_steps (Task 10's lr_at divides by it), must stay
    # strictly positive.
    _validate_int_fields(model, allow_zero=frozenset(MOE_INT_FIELDS))
    _validate_int_fields(data)
    _validate_int_fields(train, allow_zero=frozenset({"seed"}))

    _check_fraction("code_share", data.code_share, allow_one=False)
    _check_fraction("html_cap", data.html_cap, allow_one=True)
    if not data.code_heldout_first_file < data.code_files_total:
        raise ValueError(
            f"code_heldout_first_file ({data.code_heldout_first_file}) must be less than "
            f"code_files_total ({data.code_files_total}): train needs files 0..first-1 "
            f"and code val needs at least one held-out file"
        )

    if model.d_model % model.n_head != 0:
        raise ValueError("d_model must divide evenly by n_head")
    if model.n_head % model.n_kv_head != 0:
        raise ValueError("n_head must be a multiple of n_kv_head for GQA")
    if model.head_dim % 2 != 0:
        raise ValueError(f"head_dim ({model.head_dim}) must be even for RoPE")
    _check_moe(model)
    _check_data_mix(data)
    _check_train_extras(train)
    if train.precision == "fp8" and model.kind != "moe":
        # Spec section 12: dense quipu-114m is unaffected by the FP8 option.
        raise ValueError(f"precision 'fp8' applies to kind 'moe', got kind {model.kind!r}")
    _check_tokenizer(data, model)

    if train.batch_tokens % (train.micro_batch * model.context) != 0:
        raise ValueError(
            f"batch_tokens ({train.batch_tokens}) must be a multiple of "
            f"micro_batch x context ({train.micro_batch} x {model.context})"
        )

    if train.lr <= 0:
        raise ValueError(f"lr must be positive, got {train.lr!r}")
    if not (0 <= train.lr_min <= train.lr):
        raise ValueError(f"lr_min ({train.lr_min}) must be between 0 and lr ({train.lr})")

    if train.steps < 1:
        raise ValueError(
            f"steps ({train.steps}) must be at least 1; total_tokens "
            f"({train.total_tokens}) must be >= batch_tokens ({train.batch_tokens})"
        )
    if train.warmup_steps >= train.steps:
        raise ValueError(
            f"warmup_steps ({train.warmup_steps}) must be less than steps ({train.steps})"
        )
    _check_milestones(train.milestones, train.steps)

    return Config(name=raw["name"], model=model, data=data, train=train)
