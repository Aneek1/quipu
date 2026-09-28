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
