"""Hidden acceptance tests for the todo app (never shown to the model).

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
    response = client.post("/api/todos", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return response.get_json()


def test_create_returns_201_and_the_todo_with_an_integer_id(client):
    todo = create(client, title="Buy milk")
    assert isinstance(todo["id"], int)
    assert todo["title"] == "Buy milk"
    assert todo["done"] is False


def test_create_keeps_done_when_given(client):
    assert create(client, title="Walk dog", done=True)["done"] is True


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/todos")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, title="One")
    second = create(client, title="Two")
    items = client.get("/api/todos").get_json()
    assert isinstance(items, list)
    assert sorted(t["id"] for t in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, title="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/todos").get_json() == []


def test_get_one(client):
    todo = create(client, title="Read book")
    response = client.get(f"/api/todos/{todo['id']}")
    assert response.status_code == 200
    assert response.get_json()["title"] == "Read book"


def test_update(client):
    todo = create(client, title="Draft")
    response = client.put(f"/api/todos/{todo['id']}", json={"title": "Final", "done": True})
    assert response.status_code == 200
    updated = response.get_json()
    assert updated["id"] == todo["id"]
    assert updated["title"] == "Final"
    assert updated["done"] is True
    again = client.get(f"/api/todos/{todo['id']}").get_json()
    assert again["title"] == "Final" and again["done"] is True


def test_partial_update_keeps_the_fields_left_out(client):
    todo = create(client, title="Keep me")
    response = client.put(f"/api/todos/{todo['id']}", json={"done": True})
    assert response.status_code == 200
    assert response.get_json()["title"] == "Keep me"
    assert response.get_json()["done"] is True
    again = client.get(f"/api/todos/{todo['id']}").get_json()
    assert again["title"] == "Keep me" and again["done"] is True


def test_update_with_an_empty_title_is_400(client):
    todo = create(client, title="Valid")
    response = client.put(f"/api/todos/{todo['id']}", json={"title": ""})
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get(f"/api/todos/{todo['id']}").get_json()["title"] == "Valid"


def test_ids_are_never_reused(client):
    create(client, title="A")
    b = create(client, title="B")
    assert client.delete(f"/api/todos/{b['id']}").status_code in (200, 204)
    c = create(client, title="C")
    assert c["id"] != b["id"]


def test_delete_then_404(client):
    todo = create(client, title="Temporary")
    response = client.delete(f"/api/todos/{todo['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/todos/{todo['id']}").status_code == 404
    assert client.get("/api/todos").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/todos/999").status_code == 404
    assert client.put("/api/todos/999", json={"title": "x", "done": False}).status_code == 404
    assert client.delete("/api/todos/999").status_code == 404


def test_missing_title_is_400_with_an_error(client):
    response = client.post("/api/todos", json={"done": False})
    assert response.status_code == 400
    assert "error" in response.get_json()


@pytest.mark.parametrize("title", ["", "   "])
def test_empty_title_is_400(client, title):
    response = client.post("/api/todos", json={"title": title})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_boolean_done_is_400(client):
    response = client.post("/api/todos", json={"title": "x", "done": "yes"})
    assert response.status_code == 400
    assert "error" in response.get_json()
