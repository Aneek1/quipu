"""Hidden acceptance tests for the bookmarks app (never shown to the model).

They import create_app from the finished project's backend (on PYTHONPATH) and
define their own client, so nothing the model wrote in backend/tests can affect
them. Delete may answer 200 or 204; errors are checked for an `error` key only.
"""
import pytest

from app import create_app


FIELDS = ("id", "title", "url", "note")


def view(item):
    """Only the spec's fields: an app may return extra keys (such as created_at)."""
    return {key: item[key] for key in FIELDS}

URL = "https://example.com/page"


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create(client, **body):
    body.setdefault("url", URL)
    response = client.post("/api/bookmarks", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return view(response.get_json())


def test_create_returns_201_and_the_bookmark_with_an_integer_id(client):
    bookmark = create(client, title="Docs", url="http://docs.example.org", note="later")
    assert isinstance(bookmark["id"], int)
    assert bookmark["title"] == "Docs"
    assert bookmark["url"] == "http://docs.example.org"
    assert bookmark["note"] == "later"


def test_note_defaults_to_an_empty_string(client):
    assert create(client, title="No note")["note"] == ""


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/bookmarks")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, title="One")
    second = create(client, title="Two")
    items = client.get("/api/bookmarks").get_json()
    assert isinstance(items, list)
    assert sorted(b["id"] for b in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, title="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/bookmarks").get_json() == []


def test_get_one(client):
    bookmark = create(client, title="Docs", note="n")
    response = client.get(f"/api/bookmarks/{bookmark['id']}")
    assert response.status_code == 200
    got = response.get_json()
    assert got["title"] == "Docs" and got["url"] == URL and got["note"] == "n"


def test_update(client):
    bookmark = create(client, title="Draft")
    body = {"title": "Final", "url": "https://final.example", "note": "x"}
    response = client.put(f"/api/bookmarks/{bookmark['id']}", json=body)
    assert response.status_code == 200
    assert view(response.get_json()) == {"id": bookmark["id"], **body}
    assert view(client.get(f"/api/bookmarks/{bookmark['id']}").get_json()) == {"id": bookmark["id"], **body}


def test_partial_update_keeps_the_fields_left_out(client):
    bookmark = create(client, title="Keep me", note="old")
    response = client.put(f"/api/bookmarks/{bookmark['id']}", json={"note": "new"})
    assert response.status_code == 200
    expected = {"id": bookmark["id"], "title": "Keep me", "url": URL, "note": "new"}
    assert view(response.get_json()) == expected
    assert view(client.get(f"/api/bookmarks/{bookmark['id']}").get_json()) == expected


@pytest.mark.parametrize("body", [{"title": ""}, {"url": "ftp://example.com"}, {"note": 5}])
def test_update_that_breaks_a_rule_is_400(client, body):
    bookmark = create(client, title="Valid")
    response = client.put(f"/api/bookmarks/{bookmark['id']}", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert view(client.get(f"/api/bookmarks/{bookmark['id']}").get_json()) == bookmark


def test_ids_are_never_reused(client):
    create(client, title="A")
    b = create(client, title="B")
    assert client.delete(f"/api/bookmarks/{b['id']}").status_code in (200, 204)
    c = create(client, title="C")
    assert c["id"] != b["id"]


def test_delete_then_404(client):
    bookmark = create(client, title="Temporary")
    response = client.delete(f"/api/bookmarks/{bookmark['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/bookmarks/{bookmark['id']}").status_code == 404
    assert client.get("/api/bookmarks").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/bookmarks/999").status_code == 404
    assert client.put("/api/bookmarks/999", json={"title": "x"}).status_code == 404
    assert client.delete("/api/bookmarks/999").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"url": URL},
        {"title": "", "url": URL},
        {"title": "   ", "url": URL},
        {"title": 7, "url": URL},
        {"title": "x"},
        {"title": "x", "url": ""},
        {"title": "x", "url": "example.com"},
        {"title": "x", "url": "ftp://example.com"},
        {"title": "x", "url": 42},
        {"title": "x", "url": URL, "note": 42},
        {"title": "x", "url": URL, "note": None},
        {"title": None, "url": URL},
    ],
)
def test_invalid_bookmark_is_400_with_an_error(client, body):
    response = client.post("/api/bookmarks", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get("/api/bookmarks").get_json() == []
