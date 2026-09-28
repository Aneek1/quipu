"""FILE blocks are the only way model output reaches the filesystem, so the parser
is tested as a safety boundary: every way a path could escape the project, or a
reply could be silently misread, must raise BlockError with a message the model
can act on."""
import pytest

from stepbuild.harness.blocks import BlockError, FileBlock, parse_blocks, render_blocks

TWO = [
    FileBlock("backend/models.py", "import os\n\n\nclass A:\n    pass\n"),
    FileBlock("frontend/src/App.jsx", "// === not a marker ===\n=== FILE: nope\n\nx = 1\n"),
]


def test_round_trip():
    assert parse_blocks(render_blocks(TWO)) == TWO


def test_content_trailing_newline_normalised_to_one():
    assert FileBlock("a.py", "x = 1").content == "x = 1\n"
    assert FileBlock("a.py", "x = 1\n\n\n").content == "x = 1\n"
    assert parse_blocks("=== FILE: a.py ===\nx = 1\n\n\n=== END FILE ===\n") == [
        FileBlock("a.py", "x = 1\n")
    ]


def test_empty_file_round_trips():
    bs = [FileBlock("backend/__init__.py", "")]
    assert parse_blocks(render_blocks(bs)) == bs


def test_chatter_outside_blocks_is_ignored():
    text = (
        "Sure! Here are the files.\n\n"
        + "=== FILE: backend/models.py ===\nimport os\n\n\nclass A:\n    pass\n=== END FILE ===\n"
        + "And now the frontend:\n"
        + "=== FILE: frontend/src/App.jsx ===\n// === not a marker ===\n=== FILE: nope\n\nx = 1\n"
        + "=== END FILE ===\nHope this helps."
    )
    assert parse_blocks(text) == TWO


def test_windows_line_endings():
    text = render_blocks(TWO).replace("\n", "\r\n")
    assert parse_blocks(text) == TWO


def test_marker_must_be_the_whole_line():
    text = "=== FILE: a.py ===\nx = '=== END FILE ==='\n=== END FILE ===\n"
    assert parse_blocks(text) == [FileBlock("a.py", "x = '=== END FILE ==='\n")]


def test_no_blocks():
    with pytest.raises(BlockError, match="no FILE block"):
        parse_blocks("I think you should edit models.py.")


def test_unterminated_block():
    with pytest.raises(BlockError, match="unterminated.*a.py"):
        parse_blocks("=== FILE: a.py ===\nx = 1\n")


def test_new_file_marker_before_end_is_unterminated():
    with pytest.raises(BlockError, match="unterminated.*a.py"):
        parse_blocks("=== FILE: a.py ===\nx = 1\n=== FILE: b.py ===\ny\n=== END FILE ===\n")


def test_duplicate_path():
    text = render_blocks([FileBlock("a.py", "1")]) + render_blocks([FileBlock("a.py", "2")])
    with pytest.raises(BlockError, match="a.py.*more than once"):
        parse_blocks(text)


def _one(path):
    return f"=== FILE: {path} ===\nx\n=== END FILE ===\n"


@pytest.mark.parametrize("path", ["/etc/x", "C:\\x", "C:/x", "c:x", "//server/share/x"])
def test_absolute_path_rejected(path):
    with pytest.raises(BlockError):
        parse_blocks(_one(path))


@pytest.mark.parametrize("path", ["/etc/x", "C:/x"])
def test_absolute_path_message(path):
    with pytest.raises(BlockError, match="absolute"):
        parse_blocks(_one(path))


@pytest.mark.parametrize("path", ["..", "../x", "a/../../x", "a/..", "backend/../../etc/passwd"])
def test_traversal_rejected(path):
    with pytest.raises(BlockError, match=r"'\.\.' in path .*inside the project"):
        parse_blocks(_one(path))


def test_backslash_rejected_with_hint():
    with pytest.raises(BlockError, match="use '/'"):
        parse_blocks(_one("backend\\models.py"))


def test_empty_path():
    with pytest.raises(BlockError, match="empty"):
        parse_blocks("=== FILE:  ===\nx\n=== END FILE ===\n")


@pytest.mark.parametrize(
    "path", ["a//b.py", "./a.py", "a/./b.py", "a/", "a/b. /c", "a/ b.py", "nul", "backend/con.py", "a:b"]
)
def test_non_canonical_or_windows_hostile_paths_rejected(path):
    with pytest.raises(BlockError):
        parse_blocks(_one(path))


@pytest.mark.parametrize("path", ["../x", "a.py ", "a.", ""])
def test_fileblock_constructor_validates(path):
    with pytest.raises(BlockError):
        FileBlock(path, "")


def test_allowed_accepts_listed():
    assert parse_blocks(_one("backend/models.py"), allowed={"backend/models.py"}) == [
        FileBlock("backend/models.py", "x\n")
    ]


def test_path_not_in_allowed_lists_allowed_files():
    with pytest.raises(BlockError) as e:
        parse_blocks(_one("backend/app.py"), allowed=["backend/routes.py", "backend/models.py"])
    msg = str(e.value)
    assert "backend/app.py" in msg
    assert "backend/models.py" in msg and "backend/routes.py" in msg


def test_render_rejects_marker_lines_in_content():
    with pytest.raises(BlockError, match="marker"):
        render_blocks([FileBlock("a.py", "x\n=== END FILE ===\ny\n")])


def test_render_rejects_duplicates():
    with pytest.raises(BlockError, match="more than once"):
        render_blocks([FileBlock("a.py", "1"), FileBlock("a.py", "2")])


@pytest.mark.parametrize("content", ["x\r\ny\r\n", "x\ry\r", "x\r\ny", "x\r"])
def test_carriage_returns_normalised_so_round_trip_holds(content):
    b = FileBlock("a.py", content)
    assert b == FileBlock("a.py", content.replace("\r\n", "\n").replace("\r", "\n"))
    assert parse_blocks(render_blocks([b])) == [b]


def test_lone_cr_line_endings_in_reply():
    assert parse_blocks("=== FILE: a.py ===\rx\r=== END FILE ===\r") == [FileBlock("a.py", "x\n")]


@pytest.mark.parametrize(
    "path",
    [
        ".gitkeep",
        ".env.example",
        "frontend/vite.config.js",
        "src/components/List.jsx",
        "backend/tests/conftest.py",
        ".github/workflows/ci.yml",
        "a..b.py",
    ],
)
def test_legitimate_paths_accepted(path):
    b = FileBlock(path, "x\n")
    assert b.path == path
    assert parse_blocks(render_blocks([b])) == [b]
    assert parse_blocks(render_blocks([b]), allowed={path}) == [b]


def test_allowed_as_bare_string_is_a_type_error():
    with pytest.raises(TypeError):
        parse_blocks(_one("a.py"), allowed="backend/a.py")


def test_allowed_empty_says_no_files_may_be_written():
    with pytest.raises(BlockError, match="no files may be written in this step"):
        parse_blocks(_one("a.py"), allowed=[])


def test_duplicate_detection_ignores_case():
    text = _one("frontend/src/App.jsx") + _one("frontend/src/app.jsx")
    with pytest.raises(BlockError, match="more than once"):
        parse_blocks(text)
    with pytest.raises(BlockError, match="more than once"):
        render_blocks([FileBlock("App.jsx", "1"), FileBlock("app.jsx", "2")])


def test_trailing_whitespace_after_markers_tolerated():
    text = "=== FILE: a.py === \t\nx = 1\n=== END FILE ===\t \n"
    assert parse_blocks(text) == [FileBlock("a.py", "x = 1\n")]


def test_colon_in_path_has_specific_message():
    with pytest.raises(BlockError, match="':' in path"):
        parse_blocks(_one("backend/a:b.py"))
