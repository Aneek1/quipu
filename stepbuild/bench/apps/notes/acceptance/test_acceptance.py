"""Hidden acceptance tests for the notes app (never shown to the model).

They import create_app from the finished project's backend (on PYTHONPATH) and
define their own client, so nothing the model wrote in backend/tests can affect
them. Delete may answer 200 or 204; errors are checked for an `error` key only.
"""
import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create(client, **body):
    response = client.post("/api/notes", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return response.get_json()


def test_create_returns_201_and_the_note_with_an_integer_id(client):
    note = create(client, title="Groceries", body="eggs, milk")
    assert isinstance(note["id"], int)
    assert note["title"] == "Groceries"
    assert note["body"] == "eggs, milk"


def test_body_defaults_to_an_empty_string(client):
    assert create(client, title="Just a title")["body"] == ""


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/notes")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, title="One")
    second = create(client, title="Two")
    items = client.get("/api/notes").get_json()
    assert isinstance(items, list)
    assert sorted(n["id"] for n in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, title="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/notes").get_json() == []


def test_get_one(client):
    note = create(client, title="Ideas", body="a, b")
    response = client.get(f"/api/notes/{note['id']}")
    assert response.status_code == 200
    got = response.get_json()
    assert got["title"] == "Ideas" and got["body"] == "a, b"


def test_update(client):
    note = create(client, title="Draft", body="old")
    response = client.put(f"/api/notes/{note['id']}", json={"title": "Final", "body": "new"})
    assert response.status_code == 200
    updated = response.get_json()
    assert updated["id"] == note["id"]
    assert updated["title"] == "Final"
    assert updated["body"] == "new"
    again = client.get(f"/api/notes/{note['id']}").get_json()
    assert again["title"] == "Final" and again["body"] == "new"


def test_delete_then_404(client):
    note = create(client, title="Temporary")
    response = client.delete(f"/api/notes/{note['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/notes/{note['id']}").status_code == 404
    assert client.get("/api/notes").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/notes/999").status_code == 404
    assert client.put("/api/notes/999", json={"title": "x", "body": ""}).status_code == 404
    assert client.delete("/api/notes/999").status_code == 404


def test_missing_title_is_400_with_an_error(client):
    response = client.post("/api/notes", json={"body": "no title"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_empty_title_is_400(client):
    response = client.post("/api/notes", json={"title": ""})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_string_body_is_400(client):
    response = client.post("/api/notes", json={"title": "x", "body": 42})
    assert response.status_code == 400
    assert "error" in response.get_json()
