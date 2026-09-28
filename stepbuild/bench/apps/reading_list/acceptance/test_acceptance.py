"""Hidden acceptance tests for the reading_list app (never shown to the model).

They import create_app from the finished project's backend (on PYTHONPATH) and
define their own client, so nothing the model wrote in backend/tests can affect
them. Delete may answer 200 or 204; errors are checked for an `error` key only.
"""
import pytest

from app import create_app


FIELDS = ("id", "title", "author", "status")


def view(item):
    """Only the spec's fields: an app may return extra keys (such as created_at)."""
    return {key: item[key] for key in FIELDS}


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create(client, **body):
    body.setdefault("author", "Octavia Butler")
    response = client.post("/api/books", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return view(response.get_json())


def test_create_returns_201_and_the_book_with_an_integer_id(client):
    book = create(client, title="Kindred", author="Octavia E. Butler", status="reading")
    assert isinstance(book["id"], int)
    assert book["title"] == "Kindred"
    assert book["author"] == "Octavia E. Butler"
    assert book["status"] == "reading"


def test_status_defaults_to_to_read(client):
    assert create(client, title="Dawn")["status"] == "to-read"


@pytest.mark.parametrize("status", ["to-read", "reading", "done"])
def test_every_status_is_accepted(client, status):
    assert create(client, title="x", status=status)["status"] == status


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/books")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, title="One")
    second = create(client, title="Two")
    items = client.get("/api/books").get_json()
    assert isinstance(items, list)
    assert sorted(b["id"] for b in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, title="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/books").get_json() == []


def test_get_one(client):
    book = create(client, title="Parable of the Sower")
    response = client.get(f"/api/books/{book['id']}")
    assert response.status_code == 200
    assert view(response.get_json()) == book


def test_update(client):
    book = create(client, title="Draft")
    body = {"title": "Final", "author": "Someone Else", "status": "done"}
    response = client.put(f"/api/books/{book['id']}", json=body)
    assert response.status_code == 200
    assert view(response.get_json()) == {"id": book["id"], **body}
    assert view(client.get(f"/api/books/{book['id']}").get_json()) == {"id": book["id"], **body}


def test_partial_update_keeps_the_fields_left_out(client):
    book = create(client, title="Keep me", author="Kept")
    response = client.put(f"/api/books/{book['id']}", json={"status": "reading"})
    assert response.status_code == 200
    expected = {"id": book["id"], "title": "Keep me", "author": "Kept", "status": "reading"}
    assert view(response.get_json()) == expected
    assert view(client.get(f"/api/books/{book['id']}").get_json()) == expected


@pytest.mark.parametrize(
    "body", [{"title": ""}, {"author": "  "}, {"status": "finished"}, {"status": None}]
)
def test_update_that_breaks_a_rule_is_400(client, body):
    book = create(client, title="Valid")
    response = client.put(f"/api/books/{book['id']}", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert view(client.get(f"/api/books/{book['id']}").get_json()) == book


def test_ids_are_never_reused(client):
    create(client, title="A")
    b = create(client, title="B")
    assert client.delete(f"/api/books/{b['id']}").status_code in (200, 204)
    c = create(client, title="C")
    assert c["id"] != b["id"]


def test_delete_then_404(client):
    book = create(client, title="Temporary")
    response = client.delete(f"/api/books/{book['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/books/{book['id']}").status_code == 404
    assert client.get("/api/books").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/books/999").status_code == 404
    assert client.put("/api/books/999", json={"title": "x"}).status_code == 404
    assert client.delete("/api/books/999").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"author": "A"},
        {"title": "", "author": "A"},
        {"title": "   ", "author": "A"},
        {"title": 5, "author": "A"},
        {"title": "x"},
        {"title": "x", "author": ""},
        {"title": "x", "author": ["A"]},
        {"title": "x", "author": "A", "status": "finished"},
        {"title": "x", "author": "A", "status": "Done"},
        {"title": "x", "author": "A", "status": 1},
        {"title": "x", "author": "A", "status": None},
    ],
)
def test_invalid_book_is_400_with_an_error(client, body):
    response = client.post("/api/books", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get("/api/books").get_json() == []
