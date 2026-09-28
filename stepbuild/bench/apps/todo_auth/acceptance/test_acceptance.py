"""Hidden acceptance tests for the todo_auth app (never shown to the model).

They import create_app from the finished project's backend (on PYTHONPATH) and
define their own clients, so nothing the model wrote in backend/tests can affect
them. Delete may answer 200 or 204; errors are checked for an `error` key only.
Each test client keeps its own session cookie, so two clients on one app are two
people using the same server. Todos are compared field by field, so an app may
return extra keys (such as the owner) without failing.
"""
import pytest

from app import create_app

FIELDS = ("id", "title", "done")


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app


def signed_in(app, username="ada", password="s3cret"):
    client = app.test_client()
    body = {"username": username, "password": password}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/login", json=body).status_code == 200
    return client


@pytest.fixture
def client(app):
    return signed_in(app)


def view(todo):
    return {key: todo[key] for key in FIELDS}


def create(client, **body):
    response = client.post("/api/todos", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return view(response.get_json())


def get(client, todo):
    response = client.get(f"/api/todos/{todo['id']}")
    assert response.status_code == 200
    return view(response.get_json())


def test_register_returns_201_with_the_username(app):
    response = app.test_client().post("/api/auth/register", json={"username": "ada", "password": "pw"})
    assert response.status_code == 201
    assert response.get_json()["username"] == "ada"


def test_a_taken_username_is_409(app):
    signed_in(app, "ada")
    response = app.test_client().post("/api/auth/register", json={"username": "ada", "password": "x"})
    assert response.status_code == 409
    assert "error" in response.get_json()


@pytest.mark.parametrize(
    "body",
    [{"password": "pw"}, {"username": "", "password": "pw"}, {"username": "  ", "password": "pw"},
     {"username": 5, "password": "pw"}, {"username": "ada"}, {"username": "ada", "password": ""},
     {"username": "ada", "password": 1234}],
)
def test_invalid_registration_is_400(app, body):
    client = app.test_client()
    response = client.post("/api/auth/register", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.post("/api/auth/login", json={"username": "ada", "password": "pw"}).status_code == 401


def test_registering_does_not_log_in(app):
    client = app.test_client()
    client.post("/api/auth/register", json={"username": "ada", "password": "pw"})
    assert client.get("/api/todos").status_code == 401


@pytest.mark.parametrize(
    "body",
    [{"username": "ada", "password": "wrong"}, {"username": "nobody", "password": "s3cret"},
     {"username": "ada"}, {}],
)
def test_bad_login_is_401(app, body):
    signed_in(app, "ada")
    client = app.test_client()
    response = client.post("/api/auth/login", json=body)
    assert response.status_code == 401
    assert "error" in response.get_json()
    assert client.get("/api/todos").status_code == 401


def test_login_returns_the_username(app):
    signed_in(app, "ada")
    client = app.test_client()
    response = client.post("/api/auth/login", json={"username": "ada", "password": "s3cret"})
    assert response.status_code == 200
    assert response.get_json()["username"] == "ada"


def test_logged_out_todo_endpoints_are_401(app):
    client = app.test_client()
    for method, path, body in [
        ("get", "/api/todos", None), ("post", "/api/todos", {"title": "x"}),
        ("get", "/api/todos/1", None), ("put", "/api/todos/1", {"title": "x"}),
        ("delete", "/api/todos/1", None),
    ]:
        response = getattr(client, method)(path, json=body)
        assert response.status_code == 401, (method, path)
        assert "error" in response.get_json()


def test_logout_ends_the_session(client):
    todo = create(client, title="Mine")
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/todos").status_code == 401
    assert client.get(f"/api/todos/{todo['id']}").status_code == 401
    assert client.post("/api/auth/logout").status_code == 200


def test_create_returns_201_and_the_todo_with_an_integer_id(client):
    todo = create(client, title="Buy milk")
    assert isinstance(todo["id"], int)
    assert todo["title"] == "Buy milk"
    assert todo["done"] is False
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


def test_each_app_starts_empty_with_no_users(client):
    create(client, title="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    fresh = other.test_client()
    assert fresh.post("/api/auth/login", json={"username": "ada", "password": "s3cret"}).status_code == 401
    assert signed_in(other, "ada").get("/api/todos").get_json() == []


def test_get_one(client):
    todo = create(client, title="Read book")
    assert get(client, todo) == todo


def test_update(client):
    todo = create(client, title="Draft")
    response = client.put(f"/api/todos/{todo['id']}", json={"title": "Final", "done": True})
    assert response.status_code == 200
    expected = {"id": todo["id"], "title": "Final", "done": True}
    assert view(response.get_json()) == expected
    assert get(client, todo) == expected


def test_partial_update_keeps_the_fields_left_out(client):
    todo = create(client, title="Keep me")
    response = client.put(f"/api/todos/{todo['id']}", json={"done": True})
    assert response.status_code == 200
    expected = {"id": todo["id"], "title": "Keep me", "done": True}
    assert view(response.get_json()) == expected
    assert get(client, todo) == expected


@pytest.mark.parametrize("body", [{"title": ""}, {"title": "  "}, {"done": "yes"}])
def test_update_that_breaks_a_rule_is_400(client, body):
    todo = create(client, title="Valid")
    response = client.put(f"/api/todos/{todo['id']}", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert get(client, todo) == todo


def test_ids_are_never_reused(client):
    create(client, title="A")
    b = create(client, title="B")
    assert client.delete(f"/api/todos/{b['id']}").status_code in (200, 204)
    assert create(client, title="C")["id"] != b["id"]


def test_delete_then_404(client):
    todo = create(client, title="Temporary")
    assert client.delete(f"/api/todos/{todo['id']}").status_code in (200, 204)
    assert client.get(f"/api/todos/{todo['id']}").status_code == 404
    assert client.get("/api/todos").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/todos/999").status_code == 404
    assert client.put("/api/todos/999", json={"title": "x"}).status_code == 404
    assert client.delete("/api/todos/999").status_code == 404


@pytest.mark.parametrize("body", [{"done": False}, {"title": ""}, {"title": "   "}, {"title": 3}, {"title": "x", "done": "yes"}])
def test_invalid_todo_is_400_with_an_error(client, body):
    response = client.post("/api/todos", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get("/api/todos").get_json() == []


def test_users_cannot_see_or_change_each_others_todos(app):
    ada = signed_in(app, "ada")
    bob = signed_in(app, "bob")
    todo = create(ada, title="Ada's secret")
    assert bob.get("/api/todos").get_json() == []
    assert bob.get(f"/api/todos/{todo['id']}").status_code == 404
    assert bob.put(f"/api/todos/{todo['id']}", json={"title": "hacked"}).status_code == 404
    assert bob.delete(f"/api/todos/{todo['id']}").status_code == 404
    assert get(ada, todo) == todo
    mine = create(bob, title="Bob's")
    assert [t["id"] for t in ada.get("/api/todos").get_json()] == [todo["id"]]
    assert [t["id"] for t in bob.get("/api/todos").get_json()] == [mine["id"]]
