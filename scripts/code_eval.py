"""HumanEval and MBPP for quipu-moe (plan M10 amendment): functional correctness of
generated Python, run on the laptop (no paid time).

    python -m uv run python scripts/code_eval.py --config results/moe/run_config.toml \
        [--checkpoint latest|PATH] [--chat] [--samples 20] [--limit N] \
        [--benchmarks humaneval,mbpp] [--model-name NAME] [--out results/code_eval]

--model-name defaults to the release name the config's model gets
(quipu.model_card.release_name, e.g. quipu-moe-1B-A149M, plus -chat), which is the
file name the model card looks for.

Benchmarks, at the commits pinned in the config (data.humaneval_revision /
data.mbpp_revision, the same ones the shard builder decontaminated against; main()
refuses to run when the shard manifest's decontamination.benchmarks names another
commit):
- HumanEval: openai/openai_humaneval, test split, 164 problems (MIT);
- MBPP: google-research-datasets/mbpp, "sanitized" config, test split, 257 problems
  (CC-BY-4.0).

Prompts (also written, with the exact stop lists, into every report and the card):
- base (default): plain completion, zero-shot.
  HumanEval: the problem's prompt (signature + docstring) with trailing whitespace
  removed, prompt.rstrip() (the BigCode harness convention). The byte-level
  pre-tokeniser merges a newline with the next line's indentation into one token, so
  a prompt ending in a newline ends on a token that the solution never contains (every one
  of the 164 prompts broke this way with the real tokenizer); the model writes the
  newline and indentation itself. The program is the stripped prompt + completion.
  MBPP: the task text and its first test in a docstring, the model writes the
  function (BigCode style); the prompt ends on the closing docstring quotes and a
  newline, followed by "def", which tokenises as a prefix, so it is not stripped.
  Generation stops at the earliest stop sequence (STOP_HUMANEVAL / STOP_MBPP, listed
  exactly in the reports). MBPP keeps a second def (helper functions are common there).
- chat (--chat, a chat fine-tune): the spec 13 template <|user|>...<|end|><|assistant|>,
  generation stops at <|end|>; the code is the reply's first ```python block (or the
  whole reply). A reply that defines the HumanEval entry point replaces the prompt's
  function, otherwise it is taken as the body.

Decoding: greedy (temperature 0) gives pass@1; --samples n (default 20) samples at
temperature 0.8, top-p 0.95 give pass@1 and pass@10 by the unbiased estimator of
Chen et al. 2021 (pass_at_k); both k from the same T=0.8 samples (published models
often use T=0.2 for pass@1, so the reports label every row's setting). --samples 0
runs greedy only. Every MoE forward uses loop dispatch; each problem's samples are one
batch with a seeded generator.

Execution, best effort and NOT a security boundary: each candidate program is written
to its own temporary directory and run by a fresh `python -I` subprocess with a
timeout (10 s; the process tree is killed on timeout). A prelude (GUARD_PRELUDE, after
human-eval's reliability_guard) runs first: it disables deleting / renaming files,
os.system / fork / kill, subprocess.Popen, shutil.rmtree / move and builtins
exit / quit; refuses opening a file for writing outside the temporary directory;
replaces socket.socket with a subclass whose connect / connect_ex / sendto raise
(so ssl, asyncio and urllib still import) and blocks getaddrinfo / create_connection;
and on Linux / POSIX sets resource limits (address space 4 GB, CPU time, file size
16 MB, no core dumps). Proxy variables point at an unreachable address. Run it only
on a machine where running untrusted model output is acceptable.

Writes results/code_eval/<model-name>.json (per problem: greedy status and
completion, samples passed) and <model-name>.md (the table, with published numbers of
reference models, cited and not re-run).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quipu import chat as chat_format  # noqa: E402
from quipu import model_card  # noqa: E402
from quipu.eval import loop_dispatch  # noqa: E402
from quipu.fsio import write_text_atomic  # noqa: E402

HUMANEVAL, MBPP = "humaneval", "mbpp"
FILES = {
    HUMANEVAL: ("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet",
                "MIT", 164),
    MBPP: ("google-research-datasets/mbpp", "sanitized/test-00000-of-00001.parquet",
           "CC-BY-4.0", 257),
}
STOP_HUMANEVAL = ("\ndef", "\nclass", "\nif __name__", "\nprint(", "\n#")
STOP_MBPP = ("\nclass", "\nif __name__", "\nprint(", "\n#", "\nassert", '\n"""')
TIMEOUT_S = 10.0
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 384
SEED = 1234
UNREACHABLE_PROXY = "http://127.0.0.1:9"
SANDBOX_NOTE = ("best effort, not a security boundary: run it only where executing "
                "untrusted model output is acceptable")
# Runs before every candidate (human-eval's reliability_guard, plus a write guard and a
# socket that cannot connect). Everything lives inside one function that deletes itself,
# so the candidate's namespace is untouched. Limits are set here, not in preexec_fn
# (which is not safe with threads, and the harness runs candidates from a pool).
GUARD_PRELUDE = r'''
def _quipu_guard(timeout):
    import builtins, io, os, shutil, socket, subprocess, sys
    if os.name == "posix":
        try:
            import resource
            gb = 1 << 30
            for res, val in ((resource.RLIMIT_AS, 4 * gb), (resource.RLIMIT_DATA, 4 * gb),
                             (resource.RLIMIT_FSIZE, 16 << 20), (resource.RLIMIT_CORE, 0),
                             (resource.RLIMIT_CPU, timeout)):
                try:
                    resource.setrlimit(res, (val, val))
                except (ValueError, OSError):
                    pass
        except ImportError:
            pass
    try:
        import faulthandler
        faulthandler.disable()
    except Exception:
        pass

    def blocked(name):
        def refuse(*a, **k):
            raise PermissionError(f"{name} is disabled by code_eval")
        return refuse

    root = os.path.realpath(os.getcwd())

    def inside(path):
        try:
            p = os.path.realpath(os.fspath(path))
        except TypeError:        # a file descriptor: already open, allowed
            return True
        return p == root or p.startswith(root + os.sep)

    real_open, real_os_open = io.open, os.open
    writing = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def guarded_open(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+") and not inside(file):
            raise PermissionError(f"code_eval: writing outside the temp dir: {file!r}")
        return real_open(file, mode, *a, **k)

    def guarded_os_open(path, flags, *a, **k):
        if flags & writing and not inside(path):
            raise PermissionError(f"code_eval: writing outside the temp dir: {path!r}")
        return real_os_open(path, flags, *a, **k)

    builtins.open = io.open = guarded_open
    os.open = guarded_os_open
    for name in ("kill", "killpg", "system", "putenv", "remove", "removedirs", "rmdir",
                 "fchdir", "setuid", "fork", "forkpty", "rename", "renames", "truncate",
                 "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
                 "lchflags", "lchmod", "lchown", "link", "symlink", "startfile",
                 "execv", "execve", "execl", "execle", "execlp", "execvp", "execvpe",
                 "spawnl", "spawnle", "spawnv", "spawnve", "popen", "posix_spawn",
                 "posix_spawnp"):
        if hasattr(os, name):
            setattr(os, name, blocked("os." + name))
    for name in ("rmtree", "move", "chown"):     # copies go through the guarded open
        setattr(shutil, name, blocked("shutil." + name))
    class NoPopen(subprocess.Popen):     # a class: asyncio subclasses it at import
        def __init__(self, *a, **k):
            raise PermissionError("subprocess.Popen is disabled by code_eval")

        def __del__(self):
            pass

    subprocess.Popen = NoPopen
    builtins.exit = builtins.quit = None

    def no_network(*a, **k):
        raise OSError("network disabled by code_eval")

    class NoNetSocket(socket.socket):
        connect = connect_ex = sendto = no_network

    socket.socket = NoNetSocket
    socket.create_connection = socket.getaddrinfo = no_network
    for name in ("ipdb", "joblib", "psutil", "tkinter"):
        sys.modules[name] = None


_quipu_guard(%d)
del _quipu_guard
'''


def guard_prelude(timeout: float) -> str:
    """GUARD_PRELUDE with the CPU-time limit for this timeout."""
    return GUARD_PRELUDE % (int(math.ceil(timeout)) + 1)

# Published numbers of reference models (cited, not re-run) live in quipu.model_card,
# shared with the model card.
PUBLISHED = model_card.PUBLISHED


# ---- problems ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Task:
    benchmark: str
    task_id: str
    prompt: str            # the base (completion) prompt; HumanEval's is rstrip()ped
    instruction: str       # the chat user turn
    test: str              # appended after the candidate code
    entry_point: str | None
    stop: tuple[str, ...]
    setup: str = ""        # prepended to chat programs (MBPP's test imports)


def humaneval_tasks(rows: Sequence[dict]) -> list[Task]:
    out = []
    for r in rows:
        instruction = ("Complete the following Python function. Reply with the whole "
                       "function in a ```python code block.\n\n```python\n"
                       + r["prompt"] + "```")
        # rstrip: the prompt must end where a token boundary of prompt + solution is
        # (see the module docstring); the model generates the newline + indentation.
        out.append(Task(HUMANEVAL, r["task_id"], r["prompt"].rstrip(), instruction,
                        r["test"] + f"\n\ncheck({r['entry_point']})\n", r["entry_point"],
                        STOP_HUMANEVAL))
    return out


def mbpp_tasks(rows: Sequence[dict]) -> list[Task]:
    out = []
    for r in rows:
        tests = list(r["test_list"])
        imports = "\n".join(r.get("test_imports") or [])
        prompt = f'"""\n{r["prompt"]}\n{tests[0]}\n"""\n'
        instruction = (f"{r['prompt']}\nYour code should pass this test:\n{tests[0]}\n"
                       "Reply with the Python code in a ```python code block.")
        test = (imports + "\n" if imports else "") + "\n".join(tests) + "\n"
        out.append(Task(MBPP, str(r["task_id"]), prompt, instruction, test, None, STOP_MBPP,
                        setup=imports + "\n" if imports else ""))
    return out


def load_tasks(benchmark: str, revision: str,
               download: Callable[..., str] | None = None) -> tuple[list[Task], dict]:
    """The benchmark's problems at a pinned commit, and their provenance."""
    import pyarrow.parquet as pq

    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError(f"the {benchmark} revision must be a pinned 40-hex commit; "
                         f"got {revision!r}")
    if download is None:
        from huggingface_hub import hf_hub_download as download
    dataset, file, licence, expected = FILES[benchmark]
    path = download(dataset, file, repo_type="dataset", revision=revision)
    rows = pq.read_table(path).to_pylist()
    tasks = (humaneval_tasks if benchmark == HUMANEVAL else mbpp_tasks)(rows)
    if len(tasks) != expected:
        raise ValueError(f"{benchmark}: {len(tasks)} problems, expected {expected}")
    return tasks, {"dataset": dataset, "file": file, "revision": revision,
                   "license": licence, "problems": len(tasks)}


# ---- text handling ----------------------------------------------------------------------

def cut_at_stop(text: str, stops: Sequence[str]) -> str:
    """text up to the earliest stop sequence (exclusive), or all of it."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)(?:```|\Z)", re.S)


def extract_code(reply: str) -> str:
    """The first fenced code block of a chat reply, or the whole reply."""
    m = _FENCE.search(reply)
    return m.group(1) if m else reply


def build_program(task: Task, completion: str, chat: bool) -> str:
    """The program that passes iff the candidate is correct (exit status 0).
    Base HumanEval: the stripped prompt + the completion verbatim, so the completion
    must start with the newline and indentation the model generated (as it does when
    the model continues the docstring's closing line)."""
    if not chat:
        return task.prompt + completion + "\n\n" + task.test
    code = extract_code(completion)
    if task.benchmark == HUMANEVAL:
        if re.search(rf"^\s*def\s+{re.escape(task.entry_point)}\s*\(", code, re.M):
            return task.prompt + "\n    pass\n\n" + code + "\n\n" + task.test
        return task.prompt + "\n" + code + "\n\n" + task.test
    return task.setup + code + "\n\n" + task.test


def protocol(chat: bool, samples: int, *, temperature: float = TEMPERATURE,
             top_p: float = TOP_P, max_new_tokens: int = MAX_NEW_TOKENS,
             timeout: float = TIMEOUT_S) -> dict[str, Any]:
    """How the numbers were produced, exactly; saved with the results and rendered into
    the report and the model card (quipu.model_card.protocol_md)."""
    if chat:
        prompts = {
            HUMANEVAL: "chat template <|user|>...<|end|><|assistant|>; the user turn asks "
                       "to complete the function and shows the published prompt in a "
                       "```python fence; the reply's first code block (or the whole reply) "
                       "is the code",
            MBPP: "chat template; the user turn is the task text, 'Your code should pass "
                  "this test:' and the first assert; the reply's first code block is the code",
        }
        stops: dict[str, list[str]] = {HUMANEVAL: [], MBPP: []}
        stop_tokens = ["<|end|>", "<|endoftext|>"]
    else:
        prompts = {
            HUMANEVAL: "zero-shot completion of prompt.rstrip() (the published signature + "
                       "docstring, trailing whitespace removed, BigCode harness convention); "
                       "the program is that prompt + the completion + the tests",
            MBPP: "zero-shot completion of '\"\"\"\\n{task text}\\n{first assert}\\n\"\"\"\\n' "
                  "(the task text and the first assert in a docstring, BigCode style); the "
                  "program is the completion + the test imports and all asserts",
        }
        stops = {HUMANEVAL: list(STOP_HUMANEVAL), MBPP: list(STOP_MBPP)}
        stop_tokens = ["<|endoftext|>"]
    return {
        "mode": "chat" if chat else "base", "prompts": prompts, "stop_sequences": stops,
        "stop_tokens": stop_tokens, "max_new_tokens": max_new_tokens,
        "greedy": "temperature 0 (argmax), one completion per problem -> greedy pass@1",
        "samples": samples, "temperature": temperature if samples else None,
        "top_p": top_p if samples else None,
        "estimator": "unbiased pass@k (Chen et al. 2021), pass@1 and pass@10 from the "
                     "same samples",
        "timeout_s": timeout,
        "sandbox": "subprocess `python -I` in a temporary directory with a reliability "
                   "guard (no deleting / renaming files, no writing outside the temp dir, "
                   "no subprocesses, sockets cannot connect, resource limits on POSIX); "
                   + SANDBOX_NOTE,
    }


def check_revisions(revisions: dict[str, str], manifest: str | Path | None) -> None:
    """Raise if the shard manifest decontaminated against other benchmark commits than
    the ones evaluated here (manifest.decontamination.benchmarks[b].revision). No
    manifest, or one without decontamination info, is nothing to check."""
    problems = model_card.revision_mismatches(revisions, manifest)
    if problems:
        raise ValueError("benchmark revision differs from the shard manifest's "
                         "decontamination: " + "; ".join(problems))


def default_model_name(cfg, chat: bool) -> str:
    """The release name of the configured model (+ -chat): what the model card reads."""
    return model_card.release_name(*model_card.count_params(cfg.model)) + ("-chat" if chat else "")


# ---- pass@k ------------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021):
    1 - C(n - c, k) / C(n, k), as a numerically stable product."""
    if not 0 <= c <= n or k < 1 or k > n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n; got n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


# ---- execution ---------------------------------------------------------------------------

def _sandbox_env(tmp: str) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
           "TMP": tmp, "TEMP": tmp, "TMPDIR": tmp, "HOME": tmp, "NO_PROXY": "", "no_proxy": ""}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[k] = UNREACHABLE_PROXY
    if os.name == "nt":
        for k in ("SYSTEMROOT", "SystemRoot", "WINDIR"):
            if k in os.environ:
                env[k] = os.environ[k]
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the candidate and anything it started (its own process group / tree)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run_program(program: str, timeout: float = TIMEOUT_S) -> tuple[str, str]:
    """("passed" | "failed" | "timeout", the tail of stderr). Runs in a fresh temp dir."""
    with tempfile.TemporaryDirectory(prefix="quipu_code_eval_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(guard_prelude(timeout) + program, encoding="utf-8")
        kw: dict[str, Any] = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        proc = subprocess.Popen([sys.executable, "-I", "-B", str(path)], cwd=tmp,
                                env=_sandbox_env(tmp), stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **kw)
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            return "timeout", f"killed after {timeout:g} s"
        tail = err.decode("utf-8", "replace")[-500:]
        return ("passed" if proc.returncode == 0 else "failed"), tail


# ---- generation --------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, prompt_ids: list[int], n: int, *, temperature: float,
                   top_p: float = TOP_P, max_new_tokens: int = MAX_NEW_TOKENS,
                   decode: Callable[[list[int]], str], stops: Sequence[str] = (),
                   stop_ids: Sequence[int] = (), seed: int = SEED) -> list[str]:
    """n continuations of one prompt as one batch (loop dispatch for MoE), each cut at
    its first stop sequence or stop token. temperature 0 is greedy."""
    device = next(model.parameters()).device
    context = model.cfg.context
    idx = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    start = idx.shape[1]
    gen = torch.Generator(device=device).manual_seed(seed)
    done = [False] * n
    stop_ids = set(stop_ids)
    was_training = model.training
    model.eval()
    try:
        with loop_dispatch(model), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for step in range(max_new_tokens):
                logits = model(idx[:, -context:])[:, -1, :].float()
                if temperature <= 0:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    sp, si = probs.sort(dim=-1, descending=True)
                    keep = sp.cumsum(-1) - sp < top_p          # nucleus, top token always kept
                    sp = sp * keep
                    choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)
                    nxt = si.gather(-1, choice)
                idx = torch.cat([idx, nxt], dim=1)
                if (step + 1) % 8 == 0 or step + 1 == max_new_tokens or stop_ids:
                    for r in range(n):
                        if done[r]:
                            continue
                        new = idx[r, start:].tolist()
                        if stop_ids and new[-1] in stop_ids:
                            done[r] = True
                        elif stops and (step + 1) % 8 == 0 and \
                                cut_at_stop(decode(new), stops) != decode(new):
                            done[r] = True
                if all(done):
                    break
    finally:
        if was_training:
            model.train()
    out = []
    for r in range(n):
        new = idx[r, start:].tolist()
        for j, t in enumerate(new):
            if t in stop_ids:
                new = new[:j]
                break
        out.append(cut_at_stop(decode(new), stops))
    return out


def _problem_seed(task_id: str) -> int:
    return SEED + zlib.crc32(task_id.encode("utf-8"))


def evaluate(model, tok, tasks: Sequence[Task], *, chat: bool = False, samples: int = 20,
             temperature: float = TEMPERATURE, max_new_tokens: int = MAX_NEW_TOKENS,
             workers: int = 4, timeout: float = TIMEOUT_S,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Greedy + `samples` sampled candidates per task, executed; per-task and summary."""
    per: list[dict[str, Any]] = []
    decode = tok.decode
    stop_ids: list[int] = []
    if chat:
        stop_ids = [tok.special_id("<|end|>"), tok.eot]
    else:
        stop_ids = [tok.eot]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, task in enumerate(tasks):
            if chat:
                # Spec 13: the instruction is plain text (quipu.chat), so a FILE
                # marker in it stays text, as in the SFT data.
                ids = chat_format.encode(tok, [{"role": "user", "content": task.instruction}],
                                         add_generation_prompt=True)[0]
            else:
                ids = tok.encode(task.prompt)
            stops = () if chat else task.stop
            greedy = generate_batch(model, ids, 1, temperature=0.0, max_new_tokens=max_new_tokens,
                                    decode=decode, stops=stops, stop_ids=stop_ids)[0]
            sampled = (generate_batch(model, ids, samples, temperature=temperature,
                                      max_new_tokens=max_new_tokens, decode=decode, stops=stops,
                                      stop_ids=stop_ids, seed=_problem_seed(task.task_id))
                       if samples else [])
            progs = [build_program(task, c, chat) for c in [greedy] + sampled]
            results = list(pool.map(lambda p: run_program(p, timeout), progs))
            c = sum(r[0] == "passed" for r in results[1:])
            per.append({"task_id": task.task_id, "greedy": results[0][0],
                        "greedy_error": results[0][1][-200:] if results[0][0] != "passed" else "",
                        "greedy_completion": greedy, "n": len(sampled), "c": c,
                        "timeouts": sum(r[0] == "timeout" for r in results)})
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                done = sum(p["greedy"] == "passed" for p in per)
                log(f"  {i + 1}/{len(tasks)} problems, greedy {done} passed "
                    f"({time.perf_counter() - t0:.0f} s)")
    return {"per_problem": per, **summarise(per, samples, temperature)}


def summarise(per: Sequence[dict], samples: int, temperature: float = TEMPERATURE,
              top_p: float = TOP_P) -> dict[str, Any]:
    n_prob = len(per)
    out: dict[str, Any] = {
        "problems": n_prob,
        "greedy_pass@1": 100.0 * sum(p["greedy"] == "passed" for p in per) / max(n_prob, 1),
        "samples": samples, "temperature": temperature if samples else None,
        "top_p": top_p if samples else None,
    }
    for k in (1, 10):
        if samples >= k and n_prob:
            out[f"pass@{k}"] = 100.0 * sum(pass_at_k(p["n"], p["c"], k) for p in per) / n_prob
        else:
            out[f"pass@{k}"] = None
    return out


# ---- report ------------------------------------------------------------------------------

def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def results_table(ours: Sequence[dict[str, Any]]) -> str:
    """Markdown table: our models (measured here) then PUBLISHED (cited, not re-run).
    `ours` items: {"model", "params", "results": {benchmark: summary}}."""
    return model_card.code_results_table(list(ours))


def report_md(name: str, payload: dict[str, Any]) -> str:
    lines = [f"# Code evaluation: {name}", "",
             f"Mode: {'chat template' if payload['chat'] else 'base (plain completion)'}; "
             f"checkpoint `{payload['checkpoint']}`.", ""]
    for b, info in payload["benchmarks"].items():
        lines.append(f"- {b}: {info['dataset']} `{info['file']}` at {info['revision'][:12]} "
                     f"({info['license']}), {info['evaluated']} of {info['problems']} problems.")
    lines += ["", results_table([{"model": name, "params": payload.get("params", "-"),
                                  "results": payload["results"],
                                  "chat": payload["chat"]}]), ""]
    if payload.get("protocol"):
        lines += ["## Protocol", "", model_card.protocol_md(payload["protocol"]), ""]
    if any(info["evaluated"] < info["problems"] for info in payload["benchmarks"].values()):
        lines += ["", "**Partial run (--limit): not comparable with the full-benchmark numbers.**"]
    return "\n".join(lines) + "\n"


# ---- CLI ---------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--chat", action="store_true", help="use the chat template (chat model)")
    ap.add_argument("--benchmarks", default="humaneval,mbpp")
    ap.add_argument("--samples", type=int, default=20, help="samples per problem (0 = greedy only)")
    ap.add_argument("--limit", type=int, default=None, help="first N problems per benchmark")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model-name", default=None,
                    help="default: the release name, e.g. quipu-moe-1B-A149M(-chat)")
    ap.add_argument("--manifest", default=None,
                    help="shard manifest to cross-check the benchmark revisions against "
                         "(default: <shard_dir>/manifest.json, skipped when absent)")
    ap.add_argument("--out", default="results/code_eval")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args(argv)

    from quipu import evalsets
    from quipu.config import load_config
    from quipu.tokenizer import make_tokenizer

    cfg = load_config(args.config)
    revisions = {HUMANEVAL: cfg.data.humaneval_revision, MBPP: cfg.data.mbpp_revision}
    check_revisions(revisions, args.manifest or Path(cfg.data.shard_dir) / "manifest.json")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    path = (evalsets.latest_checkpoint(cfg.train.ckpt_dir) if args.checkpoint == "latest"
            else Path(args.checkpoint))
    model = evalsets.load_model(cfg.model, path, device)
    tok = make_tokenizer(cfg.data.tokenizer)
    if args.chat and not hasattr(tok, "special_id"):
        print("error: --chat needs the BPE tokenizer with chat tokens", file=sys.stderr)
        return 2
    name = args.model_name or default_model_name(cfg, args.chat)
    payload: dict[str, Any] = {"model": name, "checkpoint": str(path), "chat": args.chat,
                               "benchmarks": {}, "results": {},
                               "protocol": protocol(args.chat, args.samples,
                                                    max_new_tokens=args.max_new_tokens)}
    for b in [x.strip() for x in args.benchmarks.split(",") if x.strip()]:
        if b not in FILES:
            print(f"error: unknown benchmark {b!r}", file=sys.stderr)
            return 2
        tasks, info = load_tasks(b, revisions[b])
        if args.limit:
            tasks = tasks[: args.limit]
        print(f"{b}: {len(tasks)} problems", flush=True)
        res = evaluate(model, tok, tasks, chat=args.chat, samples=args.samples,
                       max_new_tokens=args.max_new_tokens, workers=args.workers)
        payload["benchmarks"][b] = {**info, "evaluated": len(tasks)}
        payload["results"][b] = res
        print(f"{b}: greedy pass@1 {res['greedy_pass@1']:.1f}; sampled pass@1 "
              f"{_pct(res['pass@1'])}, pass@10 {_pct(res['pass@10'])}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / f"{name}.json", json.dumps(payload, indent=1, ensure_ascii=False))
    write_text_atomic(out / f"{name}.md", report_md(name, payload))
    print(f"wrote {out / (name + '.json')} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
