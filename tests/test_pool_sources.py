"""Data v3 source adapters (quipu/pool_sources.py), offline: local parquet / json.gz /
jsonl files stand in for the Hub (an fsspec local filesystem), a fake opener for HTTP."""
import gzip
import importlib.util
import io
import json
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quipu import pool
from quipu import pool_sources as ps

ROOT = Path(__file__).resolve().parents[1]


def spec(kind="hf_parquet", name="s", **opts):
    opts.setdefault("text_field", "text")
    return pool.SourceSpec(name, kind, "a", 1.0, 1.0, opts)


def write_parquet(path, rows, row_group_size=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


@pytest.fixture
def repo(tmp_path):
    """Two directories of parquet files, 7 + 5 rows, 3 rows per row group."""
    base = tmp_path / "repo"
    write_parquet(base / "d1" / "train-0.parquet",
                  [{"text": f"one {i}", "id": f"1-{i}", "url": f"u1/{i}"} for i in range(7)])
    write_parquet(base / "d2" / "train-0.parquet",
                  [{"text": f"two {i}", "id": f"2-{i}", "url": f"u2/{i}"} for i in range(5)])
    return base


def texts(open_rows, cursor=None):
    return [(r.text, r.cursor) for r in open_rows(cursor)]


def test_parquet_rows_in_file_order_with_resumable_cursors(repo):
    fs = fsspec.filesystem("file")
    s = spec(files=["d1/*.parquet", "d2/*.parquet"], meta_fields=["id", "url"])
    src = ps.open_source(s, fs=fs, base=str(repo))
    rows = list(src.open_rows(None))
    assert [r.text for r in rows] == [f"one {i}" for i in range(7)] + [f"two {i}" for i in range(5)]
    assert rows[0].meta == {"id": "1-0", "url": "u1/0"}
    assert src.identity["files"] == 2 and len(src.identity["files_sha256"]) == 64
    # resume from every position gives exactly the rest
    for k, r in enumerate(rows):
        assert [x.text for x in src.open_rows(r.cursor)] == [x.text for x in rows[k + 1:]]


def test_skip_rows_starts_past_rows_an_earlier_build_read(repo):
    fs = fsspec.filesystem("file")
    s = spec(files=["d1/*.parquet", "d2/*.parquet"], skip_rows=8)
    src = ps.open_source(s, fs=fs, base=str(repo))
    assert [r.text for r in src.open_rows(None)] == ["two 1", "two 2", "two 3", "two 4"]


def test_dir_desc_puts_the_newest_directory_first_and_file_range_slices(repo):
    fs = fsspec.filesystem("file")
    s = spec(files=["d*/*.parquet"], order="dir_desc")
    assert [p.split("/")[0] for p in ps.open_source(s, fs=fs, base=str(repo)).files] == ["d2", "d1"]
    s = spec(files=["d*/*.parquet"], file_range=[1, 2])
    assert ps.open_source(s, fs=fs, base=str(repo)).files == ["d2/train-0.parquet"]


def test_keep_filters_and_held_out_ids_drop_rows_with_a_reason(repo):
    fs = fsspec.filesystem("file")
    write_parquet(repo / "held" / "val.parquet", [{"id": "1-2"}, {"id": "2-0"}])
    s = spec(files=["d1/*.parquet", "d2/*.parquet"], keep={"url": ["u1/0", "u1/2", "u2/0", "u2/1"]},
             exclude_ids=["held/val.parquet"], id_field="id")
    src = ps.open_source(s, fs=fs, base=str(repo))
    rows = list(src.open_rows(None))
    kept = [r.text for r in rows if r.text is not None]
    assert kept == ["one 0", "two 1"]
    drops = [r.meta.get("_drop") for r in rows if r.text is None]
    assert drops.count("dropped_heldout") == 2 and drops.count("dropped_url") == 8
    assert src.identity["exclude_ids"] == 2


def test_a_failed_read_is_retried_from_the_same_row(repo, monkeypatch):
    real = fsspec.filesystem("file")

    class Flaky:
        def __init__(self):
            self.fails = 1

        def glob(self, p):
            return real.glob(p)

        def open(self, path, mode="rb", **kw):
            f = real.open(path, mode)
            if self.fails and path.endswith("d2/train-0.parquet"):
                self.fails -= 1
                raise ConnectionError("reset by peer")
            return f

    monkeypatch.setattr(ps, "retry_wait", lambda attempt: 0)
    s = spec(files=["d1/*.parquet", "d2/*.parquet"])
    src = ps.open_source(s, fs=Flaky(), base=str(repo))
    assert len([r for r in src.open_rows(None)]) == 12


def test_jsonl_gz_rows_resume_by_file_and_line(tmp_path):
    base = tmp_path / "gz"
    base.mkdir()
    for f in range(2):
        with gzip.open(base / f"p-{f}.json.gz", "wt", encoding="utf-8") as g:
            for i in range(4):
                g.write(json.dumps({"text": f"paper {f}.{i}", "id": f"{f}{i}",
                                    "source": "s2orc/train"}) + "\n")
    s = spec("hf_jsonl_gz", files=["p-*.json.gz"], meta_fields=["id", "source"],
             attribution_field="source")
    src = ps.open_source(s, fs=fsspec.filesystem("file"), base=str(base))
    rows = list(src.open_rows(None))
    assert [r.text for r in rows] == [f"paper {f}.{i}" for f in range(2) for i in range(4)]
    assert rows[0].meta["_attrib"] == "s2orc/train"
    assert [r.text for r in src.open_rows(rows[5].cursor)] == ["paper 1.2", "paper 1.3"]


def test_arxiv_metadata_gives_the_paper_licence(tmp_path):
    base = tmp_path / "ax"
    base.mkdir()
    meta = "{'license': 'CC BY 4.0', 'url': 'https://arxiv.org/abs/0704.3395', 'authors': 'X'}"
    with gzip.open(base / "arxiv-papers-0000.json.gz", "wt", encoding="utf-8") as g:
        g.write(json.dumps({"text": "# A paper", "id": "0704.3395", "metadata": meta}) + "\n")
    s = spec("hf_jsonl_gz", files=["arxiv-*.json.gz"], render="arxiv",
             meta_fields=["id", "license", "url"], attribution_field="license")
    r = next(ps.open_source(s, fs=fsspec.filesystem("file"), base=str(base)).open_rows(None))
    assert r.meta == {"id": "0704.3395", "license": "CC BY 4.0",
                      "url": "https://arxiv.org/abs/0704.3395", "_attrib": "CC BY 4.0"}


def test_wikipedia_documents_start_with_their_title(repo):
    write_parquet(repo / "wiki" / "t.parquet", [{"title": "Quipu", "text": "Knotted cords.",
                                                 "id": "1", "url": "https://w/Quipu"}])
    s = spec(files=["wiki/*.parquet"], render="title_text", meta_fields=["id", "url", "title"])
    r = next(ps.open_source(s, fs=fsspec.filesystem("file"), base=str(repo)).open_rows(None))
    assert r.text == "Quipu\n\nKnotted cords." and r.meta["title"] == "Quipu"


def test_verified_answers_render_prompt_and_response_train_split_only(tmp_path):
    d = tmp_path / "kept"
    d.mkdir()
    rows = [{"id": "dv2-1", "split": "train", "prompt": "Add two numbers.",
             "response": "```python\ndef add(a, b):\n    return a + b\n```",
             "teacher": "Qwen/Qwen2.5-Coder-7B-Instruct", "revision": "c03e6d3"},
            {"id": "dv2-2", "split": "val", "prompt": "p", "response": "r",
             "teacher": "Qwen/Qwen2.5-Coder-7B-Instruct", "revision": "c03e6d3"}]
    (d / "part-0000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                       encoding="utf-8")
    s = spec("local_jsonl", path=str(d), files=["part-*.jsonl"], render="prompt_response",
             split="train", meta_fields=["id", "teacher", "revision"], attribution_field="teacher")
    got = list(ps.open_source(s).open_rows(None))
    assert got[0].text == ("Add two numbers.\n\n```python\ndef add(a, b):\n    return a + b\n```")
    assert got[0].meta["teacher"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert got[0].meta["revision"] == "c03e6d3"
    assert got[1].text is None and got[1].meta["_drop"] == "dropped_split"


# ------------------------------------------------------------------ Stack Exchange

SE_ROWS = [
    {"text": "Q: Why? Body.\n\nA: Because.", "meta": {
        "url": "https://apple.stackexchange.com/questions/1", "timestamp": "2023-03-29",
        "question_score": "16", "source": "stackexchange", "language": "en"}},
    {"text": "Q: How? Body.\n\nA: Like so.", "meta": {
        "url": "https://math.stackexchange.com/questions/7", "timestamp": "2023-03-29",
        "question_score": "3", "source": "stackexchange", "language": "en"}},
]
SE_SPEC = dict(url="https://example/stackexchange.jsonl", content_length=None,
               etag='"e1"', last_modified="2023-04-17", dump_date="2023-03",
               date_field="timestamp", render="stackexchange_redpajama",
               meta_fields=["url", "question_score"], attribution_field="site",
               cutoff="2024-04-30")


class FakeHttp:
    """Serves bytes with Range support; .heads are the HEAD headers."""

    def __init__(self, data, headers=None, fail_at=None):
        self.data = data
        self.headers = headers or {"Content-Length": str(len(data)), "ETag": '"e1"',
                                   "Last-Modified": "Mon, 17 Apr 2023 03:15:54 GMT"}
        self.fail_at = fail_at
        self.opened = []

    def head(self, url):
        return self.headers

    def open(self, url, offset):
        self.opened.append(offset)
        data = self.data[offset:]
        if self.fail_at is not None and offset < self.fail_at:
            cut = self.fail_at - offset
            self.fail_at = None

            class Broken(io.BytesIO):
                def readline(self_inner, *a):
                    line = io.BytesIO.readline(self_inner, *a)
                    if self_inner.tell() >= cut:
                        raise ConnectionError("dropped")
                    return line
            return Broken(data)
        return io.BytesIO(data)


def se_bytes(rows=SE_ROWS):
    return "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")


def test_stack_exchange_rows_keep_site_and_question_for_attribution():
    data = se_bytes()
    s = spec("http_jsonl", **{**SE_SPEC, "content_length": len(data)})
    src = ps.open_source(s, http=FakeHttp(data))
    rows = list(src.open_rows(None))
    assert rows[0].text == "Q: Why? Body.\n\nA: Because."
    assert rows[0].meta == {"url": "https://apple.stackexchange.com/questions/1",
                            "question_score": "16", "site": "apple.stackexchange.com",
                            "question_id": "1", "licence": "CC BY-SA",
                            "_attrib": "apple.stackexchange.com"}
    assert rows[1].cursor == {"offset": len(data)}
    assert src.identity["dump_date"] == "2023-03" and src.identity["etag"] == '"e1"'
    # resume by byte offset
    assert [r.text for r in src.open_rows(rows[0].cursor)] == [rows[1].text]


def test_stack_exchange_reads_resume_at_the_byte_after_a_dropped_connection(monkeypatch):
    monkeypatch.setattr(ps, "retry_wait", lambda attempt: 0)
    data = se_bytes(SE_ROWS * 3)
    http = FakeHttp(data, fail_at=len(data) // 2)
    s = spec("http_jsonl", **{**SE_SPEC, "content_length": len(data)})
    rows = list(ps.open_source(s, http=http).open_rows(None))
    assert len(rows) == 6 and len(http.opened) == 2 and http.opened[1] > 0


@pytest.mark.parametrize("headers, match", [
    ({"Content-Length": "5", "ETag": '"e1"', "Last-Modified": "Mon, 17 Apr 2023 03:15:54 GMT"},
     "size"),
    ({"ETag": '"other"', "Last-Modified": "Mon, 17 Apr 2023 03:15:54 GMT"}, "ETag"),
    ({"ETag": '"e1"', "Last-Modified": "Tue, 02 Jul 2024 00:00:00 GMT"}, "cutoff"),
])
def test_a_changed_or_later_stack_exchange_file_is_refused(headers, match):
    data = se_bytes()
    headers = {"Content-Length": str(len(data)), **headers}
    s = spec("http_jsonl", **{**SE_SPEC, "content_length": len(data)})
    with pytest.raises(ps.SourceRefused, match=match):
        list(ps.open_source(s, http=FakeHttp(data, headers=headers)).open_rows(None))


def test_a_stack_exchange_row_dated_after_the_cutoff_is_refused():
    late = [{**SE_ROWS[0], "meta": {**SE_ROWS[0]["meta"], "timestamp": "2024-06-30"}}]
    data = se_bytes(late)
    s = spec("http_jsonl", **{**SE_SPEC, "content_length": len(data)})
    with pytest.raises(ps.SourceRefused, match="2024-06-30"):
        list(ps.open_source(s, http=FakeHttp(data)).open_rows(None))


POSTS = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" AcceptedAnswerId="4" CreationDate="2015-01-01T00:00:00.000"
       Score="5" Title="How do I sort a list?" Body="&lt;p&gt;I have &lt;code&gt;[3, 1]&lt;/code&gt;.&lt;/p&gt;" />
  <row Id="2" PostTypeId="2" ParentId="1" CreationDate="2015-01-02T00:00:00.000" Score="9"
       Body="&lt;p&gt;Use &lt;code&gt;sorted()&lt;/code&gt;.&lt;/p&gt;" />
  <row Id="3" PostTypeId="1" CreationDate="2016-01-01T00:00:00.000" Score="1"
       Title="Unanswered" Body="&lt;p&gt;Nobody knows.&lt;/p&gt;" />
  <row Id="4" PostTypeId="2" ParentId="1" CreationDate="2015-01-03T00:00:00.000" Score="2"
       Body="&lt;pre&gt;&lt;code&gt;xs.sort()&#xA;print(xs)&lt;/code&gt;&lt;/pre&gt;" />
  <row Id="5" PostTypeId="2" ParentId="1" CreationDate="2015-01-04T00:00:00.000" Score="4"
       Body="&lt;p&gt;Or &lt;b&gt;heapq&lt;/b&gt; &amp;amp; friends.&lt;/p&gt;" />
</posts>
"""


def test_posts_xml_renders_question_then_accepted_then_answers_by_score(tmp_path):
    p = tmp_path / "Posts.xml"
    p.write_text(POSTS, encoding="utf-8")
    s = spec("se_posts_xml", path=str(p), site="stackoverflow.com", dump_date="2024-04",
             last_modified="2024-04-11", cutoff="2024-04-30")
    rows = list(ps.open_source(s).open_rows(None))
    assert len(rows) == 2
    assert rows[0].text == ("Q: How do I sort a list?\n\nI have [3, 1].\n\n"
                            "A: xs.sort()\nprint(xs)\n\n"
                            "A: Use sorted().\n\n"
                            "A: Or heapq & friends.")
    assert rows[0].meta["url"] == "https://stackoverflow.com/questions/1"
    assert rows[0].meta["licence"] == "CC BY-SA"
    assert rows[1].text == "Q: Unanswered\n\nNobody knows."
    assert [r.text for r in ps.open_source(s).open_rows(rows[0].cursor)] == [rows[1].text]


def test_strip_html_keeps_code_and_paragraph_breaks():
    assert ps.strip_html("<p>a &lt;b&gt;</p><p>c<br>d</p><ul><li>x</li><li>y</li></ul>") == \
        "a <b>\n\nc\nd\n\nx\ny"


# ------------------------------------------------------------------ code rules agree

def test_the_code_path_rules_match_data_v2():
    spec_ = importlib.util.spec_from_file_location("build_shards",
                                                   ROOT / "scripts" / "build_shards.py")
    bs = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(bs)
    for path in ["src/a.py", "dist/app.js", "x/node_modules/y.js", "a.min.js", "b.MIN.css",
                 "m.js.map", "src/vendor_utils.py", "lib/bundle.js", "a/b.pack.js",
                 "third_party/z.c", "", None, "src/build.py", "build/out.py"]:
        assert pool.code_path_reason(path) == bs.path_skip_reason(path), path
