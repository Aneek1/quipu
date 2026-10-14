"""Source adapters for the data v3 pool (quipu/pool.py): each turns one configured source
into a stream of pool.Row (plain text + provenance + a resumable cursor).

Kinds (configs/data-v3.toml, [sources.*] kind):

- hf_parquet   parquet files of a Hub dataset at a pinned commit, read through
               HfFileSystem one row group at a time (range requests: nothing is
               downloaded whole). Cursor {"file", "row"}: a resumed build skips whole
               row groups by their metadata and starts inside the right one.
- hf_jsonl_gz  gzipped JSON-lines files of a Hub dataset (peS2o's data files, which
               its loading script would read; common-pile arXiv). Cursor {"file", "row"};
               a resume decompresses the file from its start and skips `row` lines.
- http_jsonl   one JSON-lines file over HTTP (RedPajama-1T's Stack Exchange file).
               Its size, ETag and Last-Modified are checked against the pinned values
               and the Stack Exchange cutoff before any row is read. Cursor {"offset"}:
               a resume is a Range request from that byte.
- local_jsonl  local JSON-lines files (data/distill-v2/kept). Cursor {"file", "row"}.
- se_posts_xml a Stack Exchange Posts.xml (an archive.org dump <= 2024-04, extracted):
               each question rendered with its answers (accepted first, then by score),
               HTML stripped. Grouping holds the site in memory: small sites only.

Every read is retried from the same position (retry_wait backoff, ATTEMPTS in all)
unless the error cannot be fixed by a retry (missing file or repo, no access, a
refused source, a programming error).
"""
from __future__ import annotations

import ast
import fnmatch
import gzip
import hashlib
import io
import json
import re
import sys
import time
from html.parser import HTMLParser
from typing import Any, Callable, Iterator

from quipu.pool import Row, SourceSpec

ATTEMPTS = 12
SE_LICENCE = "CC BY-SA"


class SourceRefused(RuntimeError):
    """The source is not what was pinned (size, ETag, date) or is dated after the
    Stack Exchange cutoff. Never retried."""


def retry_wait(attempt: int) -> float:
    """Seconds to wait after failed attempt `attempt` (1-based): 5, 10, 20, ... <= 300."""
    return min(300.0, 5.0 * 2 ** (attempt - 1))


def is_permanent(exc: BaseException) -> bool:
    if isinstance(exc, (SourceRefused, FileNotFoundError, PermissionError, IsADirectoryError,
                        TypeError, AttributeError, KeyError, NameError, AssertionError,
                        NotImplementedError, ImportError, MemoryError)):
        return True
    try:
        from huggingface_hub import errors as hub
        if isinstance(exc, (hub.RepositoryNotFoundError, hub.RevisionNotFoundError,
                            hub.EntryNotFoundError, hub.GatedRepoError)):
            return True
    except ImportError:  # pragma: no cover
        pass
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)  # urllib.error.HTTPError
    if isinstance(status, int):
        return 400 <= status < 500 and status not in (408, 429)
    return False


def _retry(exc: BaseException, attempt: int, what: str) -> None:
    """Wait before the next attempt, or re-raise when retrying cannot help."""
    if is_permanent(exc) or attempt >= ATTEMPTS:
        raise exc
    wait = retry_wait(attempt)
    print(f"\n{what} failed: {exc!r}; retry {attempt}/{ATTEMPTS - 1} in {wait:.0f}s",
          file=sys.stderr, flush=True)
    time.sleep(wait)


def _sha256_lines(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ HTML

class _Strip(HTMLParser):
    BLOCK = {"p", "div", "blockquote", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5",
             "h6", "hr", "pre"}
    LINE = {"li", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.pre = 0

    def _tail(self) -> str:
        return "".join(self.out[-3:])

    def _break(self, n: int) -> None:
        if not self.out:
            return
        have = len(self._tail()) - len(self._tail().rstrip("\n"))
        if have < n:
            self.out.append("\n" * (n - have))

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.BLOCK:
            self._break(2)
        elif tag in self.LINE:
            self._break(1)
        elif tag == "br":
            self.out.append("\n")
        if tag == "pre":
            self.pre += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.pre = max(0, self.pre - 1)
        if tag in self.BLOCK:
            self._break(2)

    def handle_data(self, data: str) -> None:
        if not self.pre and not data.strip() and (not self.out or self._tail().endswith("\n")):
            return
        self.out.append(data)


def strip_html(html: str) -> str:
    """Stack Exchange post HTML as plain text: paragraphs and blocks separated by a
    blank line, list items and <br> by a newline, code and <pre> kept verbatim,
    entities decoded."""
    p = _Strip()
    p.feed(html or "")
    p.close()
    text = "".join(p.out)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def render_question(title: str, body: str, answers: list[dict[str, Any]],
                    accepted: str | None) -> str:
    """"Q: title\n\nbody" then each answer as "A: body": the accepted one first, then
    by score (highest first), ties by id."""
    def key(a: dict[str, Any]) -> tuple:
        return (a["id"] != accepted, -int(a.get("score") or 0), int(a["id"]))
    parts = [f"Q: {title.strip()}\n\n{strip_html(body)}".rstrip()]
    for a in sorted(answers, key=key):
        parts.append(f"A: {strip_html(a['body'])}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ rendering

def _date10(value: Any) -> str:
    return str(value)[:10]


class Renderer:
    """Turns one record (a dict) into a Row: the source's keep / split / held-out
    filters, its text (render), its provenance (meta_fields) and attribution key."""

    def __init__(self, spec: SourceSpec, cutoff: str, exclude: frozenset[str] = frozenset()):
        self.spec = spec
        self.text_field = spec.get("text_field", "text")
        self.meta_fields = list(spec.get("meta_fields", []))
        self.keep = {k: frozenset(v) for k, v in (spec.get("keep") or {}).items()}
        self.split = spec.get("split")
        self.exclude = exclude
        self.id_field = spec.get("id_field", "id")
        self.render = spec.get("render", "text")
        self.attrib = spec.get("attribution_field")
        self.date_field = spec.get("date_field")
        self.cutoff = cutoff

    def columns(self) -> list[str]:
        cols = [self.text_field, *self.meta_fields, *self.keep]
        if self.exclude:
            cols.append(self.id_field)
        if self.spec.get("path_field"):
            cols.append(self.spec.get("path_field"))
        if self.render == "title_text":
            cols.append("title")
        if self.render == "arxiv":
            cols.append("metadata")
        return list(dict.fromkeys(cols))

    def __call__(self, rec: dict[str, Any], cursor: dict[str, Any]) -> Row:
        for f, allowed in self.keep.items():
            if rec.get(f) not in allowed:
                return Row(None, {"_drop": f"dropped_{f}"}, cursor)
        if self.split is not None and rec.get("split") != self.split:
            return Row(None, {"_drop": "dropped_split"}, cursor)
        if self.exclude and rec.get(self.id_field) in self.exclude:
            return Row(None, {"_drop": "dropped_heldout"}, cursor)
        r = self.render
        extra: dict[str, Any] = {}
        if r == "stackexchange_redpajama":
            meta = rec.get("meta") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            if self.date_field:
                d = _date10(meta.get(self.date_field, ""))
                if not d or d > self.cutoff:
                    raise SourceRefused(f"{self.spec.name}: a row is dated {d or 'nothing'}, "
                                        f"after the Stack Exchange cutoff {self.cutoff}")
            src = meta
            url = str(meta.get("url") or "")
            m = re.match(r"https?://([^/]+)/questions/(\d+)", url)
            extra = {"site": m.group(1) if m else None, "question_id": m.group(2) if m else None,
                     "licence": SE_LICENCE}
            text = rec.get(self.text_field)
        elif r == "arxiv":
            md = rec.get("metadata") or {}
            if isinstance(md, str):
                try:
                    md = ast.literal_eval(md)
                except (ValueError, SyntaxError):
                    md = {}
            src = {**md, **{k: v for k, v in rec.items() if k != "metadata"}}
            text = rec.get(self.text_field)
        else:
            src = rec
            if r == "title_text":
                title = (rec.get("title") or "").strip()
                body = rec.get(self.text_field) or ""
                text = f"{title}\n\n{body}" if title else body
            elif r == "prompt_response":
                text = f"{(rec.get('prompt') or '').strip()}\n\n{(rec.get('response') or '').strip()}"
            elif r == "text":
                text = rec.get(self.text_field)
            else:
                raise ValueError(f"source {self.spec.name}: unknown render {r!r}")
        meta = {f: src.get(f) for f in self.meta_fields}
        meta.update(extra)
        if self.spec.get("path_field") and self.spec.get("path_field") not in meta:
            meta[self.spec.get("path_field")] = rec.get(self.spec.get("path_field"))
        if self.attrib:
            meta["_attrib"] = meta.get(self.attrib, src.get(self.attrib))
        return Row(text if isinstance(text, str) else None,
                   meta if isinstance(text, str) else {**meta, "_drop": "dropped_no_text"},
                   cursor)


# ------------------------------------------------------------------ readers

def iter_parquet(fs: Any, base: str, files: list[str], cursor: dict[str, Any] | None,
                 columns: list[str]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """(record, cursor after it) for every row of `files` from `cursor` on."""
    import pyarrow.parquet as pq

    fi = int((cursor or {}).get("file", 0))
    row = int((cursor or {}).get("row", 0))
    attempt = 0
    while fi < len(files):
        path = f"{base}/{files[fi]}"
        try:
            with fs.open(path, "rb", block_size=16 * 1024 * 1024) as f:
                pf = pq.ParquetFile(f)
                n = pf.metadata.num_rows
                if row >= n:
                    fi, row = fi + 1, row - n
                    continue
                names = set(pf.schema_arrow.names)
                cols = [c for c in columns if c in names]
                start = 0
                for rg in range(pf.metadata.num_row_groups):
                    rn = pf.metadata.row_group(rg).num_rows
                    if row >= start + rn:
                        start += rn
                        continue
                    data = pf.read_row_group(rg, columns=cols).to_pylist()
                    for j in range(row - start, rn):
                        row = start + j + 1
                        attempt = 0
                        yield data[j], {"file": fi, "row": row}
                    start += rn
            fi, row = fi + 1, 0
        except Exception as exc:
            attempt += 1
            _retry(exc, attempt, f"read {path} (row {row})")


def _iter_lines(open_file: Callable[[int], Any], files: list[str], cursor: dict[str, Any] | None,
                what: str) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    fi = int((cursor or {}).get("file", 0))
    row = int((cursor or {}).get("row", 0))
    attempt = 0
    while fi < len(files):
        try:
            with open_file(fi) as f:
                for i, line in enumerate(f):
                    if i < row:
                        continue
                    row = i + 1
                    attempt = 0
                    if not line.strip():
                        continue
                    yield json.loads(line), {"file": fi, "row": row}
            fi, row = fi + 1, 0
        except Exception as exc:
            attempt += 1
            _retry(exc, attempt, f"read {what} {files[fi]} (line {row})")


def iter_jsonl_gz(fs: Any, base: str, files: list[str], cursor: dict[str, Any] | None
                  ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    def open_file(fi: int) -> Any:
        raw = fs.open(f"{base}/{files[fi]}", "rb", block_size=8 * 1024 * 1024)
        return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8")
    return _iter_lines(open_file, files, cursor, "json.gz")


def iter_local_jsonl(directory: str, files: list[str], cursor: dict[str, Any] | None
                     ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    def open_file(fi: int) -> Any:
        return open(f"{directory}/{files[fi]}", encoding="utf-8")
    return _iter_lines(open_file, files, cursor, "jsonl")


USER_AGENT = "quipu-pool/1.0 (data v3 builder; +https://github.com/Aneek1/quipu)"


class UrllibHttp:
    """HEAD and ranged GET with the standard library. A named User-Agent: the
    RedPajama host (Cloudflare) answers urllib's default agent with 403."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout

    def head(self, url: str) -> dict[str, str]:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return dict(r.headers.items())

    def open(self, url: str, offset: int) -> Any:
        import urllib.request
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(url, headers=headers)
        r = urllib.request.urlopen(req, timeout=self.timeout)
        if offset and r.status != 206:
            r.close()
            raise SourceRefused(f"{url}: the server ignored the Range request (status "
                                f"{r.status}); cannot resume at byte {offset}")
        return r


def check_http_source(spec: SourceSpec, headers: dict[str, str], cutoff: str) -> dict[str, Any]:
    """The pinned file must be the one served: same size and ETag, Last-Modified on
    or before the cutoff (and the pinned day). Returns what was checked."""
    from email.utils import parsedate_to_datetime

    h = {k.lower(): v for k, v in headers.items()}
    size = int(h["content-length"]) if "content-length" in h else None
    if spec.get("content_length") is not None and size != int(spec.get("content_length")):
        raise SourceRefused(f"{spec.name}: served size {size} != pinned size "
                            f"{spec.get('content_length')}")
    if spec.get("etag") is not None and h.get("etag") != spec.get("etag"):
        raise SourceRefused(f"{spec.name}: served ETag {h.get('etag')} != pinned "
                            f"{spec.get('etag')}")
    lm = h.get("last-modified")
    day = parsedate_to_datetime(lm).date().isoformat() if lm else None
    if day is None or day > cutoff:
        raise SourceRefused(f"{spec.name}: Last-Modified {day} is after the Stack Exchange "
                            f"cutoff {cutoff} (or missing)")
    if spec.get("last_modified") and day != spec.get("last_modified"):
        raise SourceRefused(f"{spec.name}: Last-Modified {day} != pinned "
                            f"{spec.get('last_modified')}")
    return {"content_length": size, "etag": h.get("etag"), "last_modified": day}


def iter_http_jsonl(http: Any, url: str, cursor: dict[str, Any] | None, size: int | None = None
                    ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Rows of one JSON-lines file over HTTP; cursor {"offset"}. A line without its
    newline before the end of the file (`size`) is a cut connection: read again from
    the same byte, never skipped."""
    offset = int((cursor or {}).get("offset", 0))
    attempt = 0
    while True:
        try:
            r = http.open(url, offset)
            try:
                while True:
                    line = r.readline()
                    # without a known size a missing newline can only be taken as the end
                    cut = size is not None and offset + len(line) < size
                    if not line and not cut:
                        return
                    if cut and not line.endswith(NL):
                        raise ConnectionError(f"short read at byte {offset} (size {size})")
                    rec = json.loads(line) if line.strip() else None
                    offset += len(line)
                    attempt = 0
                    if rec is not None:
                        yield rec, {"offset": offset}
            finally:
                r.close()
        except Exception as exc:
            attempt += 1
            _retry(exc, attempt, f"read {url} (byte {offset})")


NL = b"\n"


def stripe_starts(http: Any, url: str, size: int, stripes: int) -> list[list[int]]:
    """[start, end) byte ranges of `stripes` equal parts of the file, each start moved
    to the beginning of the next line (the first stripe starts at 0). Empty stripes
    are dropped."""
    starts = [0]
    for k in range(1, stripes):
        s = k * size // stripes
        attempt = 0
        while True:
            try:
                r = http.open(url, s)
                try:
                    s += len(r.readline())
                finally:
                    r.close()
                break
            except Exception as exc:
                attempt += 1
                _retry(exc, attempt, f"find stripe {k} of {url}")
        starts.append(min(max(s, starts[-1]), size))
    ends = starts[1:] + [size]
    return [[a, b] for a, b in zip(starts, ends) if b > a]


def iter_http_jsonl_striped(http: Any, url: str, cursor: dict[str, Any] | None, size: int,
                            stripes: int) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Rows of a JSON-lines file read as `stripes` byte ranges, one line from each in
    turn (a file sorted by site gives every chunk rows from all over it). Cursor
    {"stripes": [[offset, end], ...], "turn": k}: each range's next byte and the range
    whose turn is next."""
    if cursor is None:
        state = {"stripes": stripe_starts(http, url, size, stripes), "turn": 0}
    else:
        state = {"stripes": [list(x) for x in cursor["stripes"]], "turn": int(cursor["turn"])}
    open_: dict[int, Any] = {}
    attempt = 0
    st = state["stripes"]
    n = len(st)

    def advance(k: int) -> int:
        for j in range(1, n + 1):
            if st[(k + j) % n][0] < st[(k + j) % n][1]:
                return (k + j) % n
        return k

    try:
        while any(off < end for off, end in st):
            k = state["turn"]
            if st[k][0] >= st[k][1]:
                state["turn"] = advance(k)
                continue
            try:
                r = open_.get(k)
                if r is None:
                    r = open_[k] = http.open(url, st[k][0])
                line = r.readline()
                if not line.endswith(NL) and st[k][0] + len(line) < st[k][1]:
                    raise ConnectionError(f"stripe {k}: short read at byte {st[k][0]} (ends "
                                          f"{st[k][1]})")
                rec = json.loads(line) if line.strip() else None
            except Exception as exc:
                bad = open_.pop(k, None)
                if bad is not None:
                    bad.close()
                attempt += 1
                _retry(exc, attempt, f"read {url} (stripe {k}, byte {st[k][0]})")
                continue
            attempt = 0
            st[k][0] += len(line)
            if st[k][0] >= st[k][1]:
                done = open_.pop(k, None)
                if done is not None:
                    done.close()
            state["turn"] = advance(k)
            if rec is not None:
                yield rec, {"stripes": [list(x) for x in st], "turn": state["turn"]}
    finally:
        for r in open_.values():
            r.close()


def iter_posts_xml(path: str, cutoff: str, site: str
                   ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Every question of a Posts.xml with its answers, in file order; cursor
    {"question": n} (questions done). Refuses a dump with posts after the cutoff."""
    import xml.etree.ElementTree as ET

    questions: dict[str, dict[str, Any]] = {}
    answers: dict[str, list[dict[str, Any]]] = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "row":
            continue
        a = el.attrib
        day = _date10(a.get("CreationDate", ""))
        if day > cutoff:
            raise SourceRefused(f"{path}: a post is dated {day}, after the Stack Exchange "
                                f"cutoff {cutoff}: this is not a dump <= {cutoff}")
        if a.get("PostTypeId") == "1":
            questions[a["Id"]] = {"id": a["Id"], "title": a.get("Title", ""),
                                  "body": a.get("Body", ""), "score": a.get("Score"),
                                  "accepted": a.get("AcceptedAnswerId"), "date": day}
        elif a.get("PostTypeId") == "2" and a.get("ParentId"):
            answers.setdefault(a["ParentId"], []).append(
                {"id": a["Id"], "body": a.get("Body", ""), "score": a.get("Score")})
        el.clear()
    for n, (qid, q) in enumerate(questions.items()):
        rec = {"text": render_question(q["title"], q["body"], answers.get(qid, []),
                                       q["accepted"]),
               "url": f"https://{site}/questions/{qid}", "question_id": qid, "site": site,
               "question_score": q["score"], "licence": SE_LICENCE, "date": q["date"]}
        yield rec, {"question": n + 1}


# ------------------------------------------------------------------ sources

class Source:
    """An opened source: open_rows(cursor) -> Rows; identity (what a resume must
    match); files (the resolved file list, for file-based kinds)."""

    def __init__(self, spec: SourceSpec, open_rows: Callable[[Any], Iterator[Row]],
                 identity: dict[str, Any], files: list[str] | None = None) -> None:
        self.spec, self.open_rows, self.identity, self.files = spec, open_rows, identity, files


def list_files(fs: Any, base: str, patterns: list[str], order: str = "asc",
               file_range: list[int] | None = None) -> list[str]:
    """Each pattern's matches (repo-relative), sorted, the patterns' lists in order;
    order "dir_desc": directories newest (last) first, files ascending inside each."""
    norm = base.replace("\\", "/").rstrip("/")
    out: list[str] = []
    for pat in patterns:
        found = []
        for p in fs.glob(f"{base}/{pat}"):
            p = str(p).replace("\\", "/")
            i = p.lower().find(norm.lower().split("://")[-1])
            rel = p[i + len(norm.split("://")[-1]):].lstrip("/") if i >= 0 else p
            if fnmatch.fnmatch(rel, pat):
                found.append(rel)
        if not found:
            raise FileNotFoundError(f"no files match {base}/{pat}")
        found.sort()
        if order == "dir_desc":
            dirs = sorted({f.rsplit("/", 1)[0] for f in found}, reverse=True)
            found = [f for d in dirs for f in sorted(x for x in found if x.rsplit("/", 1)[0] == d)]
        out += [f for f in found if f not in out]
    if file_range is not None:
        out = out[file_range[0]:file_range[1]]
    return out


def _load_ids(fs: Any, base: str, files: list[str], field: str) -> frozenset[str]:
    import pyarrow.parquet as pq
    ids: set[str] = set()
    for f in files:
        attempt = 0
        while True:
            try:
                with fs.open(f"{base}/{f}", "rb") as fh:
                    ids.update(pq.read_table(fh, columns=[field]).column(field).to_pylist())
                break
            except Exception as exc:
                attempt += 1
                _retry(exc, attempt, f"read held-out ids {f}")
    return frozenset(ids)


def open_source(spec: SourceSpec, *, fs: Any = None, base: str | None = None,
                http: Any = None, cutoff: str | None = None) -> Source:
    """The adapter for spec.kind. fs/base default to HfFileSystem and
    datasets/<repo>@<revision>; http to UrllibHttp; cutoff to the spec's or 2024-04-30."""
    from pathlib import Path

    cutoff = str(cutoff or spec.get("cutoff") or "2024-04-30")
    kind = spec.kind
    if kind in ("hf_parquet", "hf_jsonl_gz"):
        if fs is None:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
        base = base or f"datasets/{spec.get('repo')}@{spec.get('revision')}"
        files = list_files(fs, base, list(spec.get("files")), spec.get("order", "asc"),
                           spec.get("file_range"))
        exclude: frozenset[str] = frozenset()
        identity: dict[str, Any] = {"kind": kind, "repo": spec.get("repo"),
                                    "revision": spec.get("revision"), "files": len(files),
                                    "files_sha256": _sha256_lines(files),
                                    "first_file": files[0], "last_file": files[-1]}
        if spec.get("exclude_ids"):
            exclude = _load_ids(fs, base, list(spec.get("exclude_ids")),
                                spec.get("id_field", "id"))
            identity.update(exclude_ids=len(exclude),
                            exclude_ids_sha256=_sha256_lines(sorted(map(str, exclude))))
        if spec.get("skip_rows"):
            identity["skip_rows"] = int(spec.get("skip_rows"))
        render = Renderer(spec, cutoff, exclude)
        columns = render.columns()
        skip = int(spec.get("skip_rows") or 0)

        def open_rows(cursor: Any) -> Iterator[Row]:
            start = cursor if cursor is not None else {"file": 0, "row": skip}
            it = (iter_parquet(fs, base, files, start, columns) if kind == "hf_parquet"
                  else iter_jsonl_gz(fs, base, files, start))
            for rec, cur in it:
                yield render(rec, cur)
        return Source(spec, open_rows, identity, files)

    if kind == "http_jsonl":
        http = http or UrllibHttp()
        url = spec.get("url")
        render = Renderer(spec, cutoff)
        identity = {"kind": kind, "url": url, "via": spec.get("via"),
                    "revision": spec.get("revision"), "etag": spec.get("etag"),
                    "content_length": spec.get("content_length"),
                    "last_modified": spec.get("last_modified"),
                    "dump_date": spec.get("dump_date"), "sha256": spec.get("sha256"),
                    "stripes": int(spec.get("stripes", 1))}
        stripes = int(spec.get("stripes", 1))

        def open_rows(cursor: Any) -> Iterator[Row]:
            checked = check_http_source(spec, http.head(url), cutoff)
            if stripes > 1:
                if not checked["content_length"]:
                    raise SourceRefused(f"{spec.name}: striped reads need the file size")
                it = iter_http_jsonl_striped(http, url, cursor, checked["content_length"],
                                             stripes)
            else:
                it = iter_http_jsonl(http, url, cursor, checked["content_length"])
            for rec, cur in it:
                yield render(rec, cur)
        return Source(spec, open_rows, identity)

    if kind == "local_jsonl":
        directory = str(spec.get("path"))
        files = []
        for pat in spec.get("files", ["*.jsonl"]):
            files += sorted(p.name for p in Path(directory).glob(pat))
        if not files:
            raise FileNotFoundError(f"no files match {directory}/{spec.get('files')}")
        sizes = [f"{f}:{(Path(directory) / f).stat().st_size}" for f in files]
        identity = {"kind": kind, "path": directory, "files": len(files),
                    "files_sha256": _sha256_lines(sizes)}
        render = Renderer(spec, cutoff)

        def open_rows(cursor: Any) -> Iterator[Row]:
            for rec, cur in iter_local_jsonl(directory, files, cursor):
                yield render(rec, cur)
        return Source(spec, open_rows, identity, files)

    if kind == "se_posts_xml":
        path = str(spec.get("path"))
        site = str(spec.get("site"))
        sub = SourceSpec(spec.name, kind, spec.slice, spec.weight, spec.share,
                         {**spec.opts, "render": "text", "text_field": "text",
                          "meta_fields": spec.get("meta_fields") or
                          ["url", "question_id", "site", "question_score", "licence"],
                          "attribution_field": spec.get("attribution_field", "site")})
        render = Renderer(sub, cutoff)
        identity = {"kind": kind, "path": path, "site": site,
                    "dump_date": spec.get("dump_date"),
                    "size": Path(path).stat().st_size}

        def open_rows(cursor: Any) -> Iterator[Row]:
            done = int((cursor or {}).get("question", 0))
            for rec, cur in iter_posts_xml(path, cutoff, site):
                if cur["question"] <= done:
                    continue
                yield render(rec, cur)
        return Source(spec, open_rows, identity)

    raise ValueError(f"source {spec.name}: unknown kind {kind!r}")
