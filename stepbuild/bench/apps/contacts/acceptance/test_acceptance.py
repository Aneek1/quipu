"""Hidden acceptance tests for the contacts app (never shown to the model).

They import create_app from the finished project's backend (on PYTHONPATH) and
define their own client, so nothing the model wrote in backend/tests can affect
them. Delete may answer 200 or 204; errors are checked for an `error` key only.
"""
import pytest

from app import create_app

EMAIL = "grace@example.org"


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create(client, **body):
    body.setdefault("email", EMAIL)
    response = client.post("/api/contacts", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return response.get_json()


def test_create_returns_201_and_the_contact_with_an_integer_id(client):
    contact = create(client, name="Grace Hopper", email="grace@navy.mil", phone="555-0199")
    assert isinstance(contact["id"], int)
    assert contact["name"] == "Grace Hopper"
    assert contact["email"] == "grace@navy.mil"
    assert contact["phone"] == "555-0199"


def test_phone_defaults_to_an_empty_string(client):
    assert create(client, name="No phone")["phone"] == ""


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/contacts")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, name="One")
    second = create(client, name="Two")
    items = client.get("/api/contacts").get_json()
    assert isinstance(items, list)
    assert sorted(c["id"] for c in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, name="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/contacts").get_json() == []


def test_get_one(client):
    contact = create(client, name="Alan", phone="1")
    response = client.get(f"/api/contacts/{contact['id']}")
    assert response.status_code == 200
    assert response.get_json() == contact


def test_update(client):
    contact = create(client, name="Draft")
    body = {"name": "Final", "email": "final@example.com", "phone": "2"}
    response = client.put(f"/api/contacts/{contact['id']}", json=body)
    assert response.status_code == 200
    assert response.get_json() == {"id": contact["id"], **body}
    assert client.get(f"/api/contacts/{contact['id']}").get_json() == {"id": contact["id"], **body}


def test_partial_update_keeps_the_fields_left_out(client):
    contact = create(client, name="Keep me", phone="old")
    response = client.put(f"/api/contacts/{contact['id']}", json={"phone": "new"})
    assert response.status_code == 200
    expected = {"id": contact["id"], "name": "Keep me", "email": EMAIL, "phone": "new"}
    assert response.get_json() == expected
    assert client.get(f"/api/contacts/{contact['id']}").get_json() == expected


@pytest.mark.parametrize(
    "body", [{"name": ""}, {"email": "not-an-email"}, {"email": ""}, {"phone": 5551234}]
)
def test_update_that_breaks_a_rule_is_400(client, body):
    contact = create(client, name="Valid")
    response = client.put(f"/api/contacts/{contact['id']}", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get(f"/api/contacts/{contact['id']}").get_json() == contact


def test_ids_are_never_reused(client):
    create(client, name="A")
    b = create(client, name="B")
    assert client.delete(f"/api/contacts/{b['id']}").status_code in (200, 204)
    c = create(client, name="C")
    assert c["id"] != b["id"]


def test_delete_then_404(client):
    contact = create(client, name="Temporary")
    response = client.delete(f"/api/contacts/{contact['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/contacts/{contact['id']}").status_code == 404
    assert client.get("/api/contacts").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/contacts/999").status_code == 404
    assert client.put("/api/contacts/999", json={"name": "x"}).status_code == 404
    assert client.delete("/api/contacts/999").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"email": EMAIL},
        {"name": "", "email": EMAIL},
        {"name": "   ", "email": EMAIL},
        {"name": 1, "email": EMAIL},
        {"name": "x"},
        {"name": "x", "email": ""},
        {"name": "x", "email": "grace.example.org"},
        {"name": "x", "email": "@example.org"},
        {"name": "x", "email": "grace@"},
        {"name": "x", "email": "a@b@example.org"},
        {"name": "x", "email": 42},
        {"name": "x", "email": EMAIL, "phone": 5551234},
    ],
)
def test_invalid_contact_is_400_with_an_error(client, body):
    response = client.post("/api/contacts", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get("/api/contacts").get_json() == []
