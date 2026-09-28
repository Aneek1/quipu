"""Hidden acceptance tests for the inventory app (never shown to the model).

They import create_app from the finished project's backend (on PYTHONPATH) and
define their own client, so nothing the model wrote in backend/tests can affect
them. Delete may answer 200 or 204; errors are checked for an `error` key only.
The spec says booleans are not integers, so `quantity: true` is expected to be 400.
"""
import pytest

from app import create_app


FIELDS = ("id", "name", "quantity")


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
    response = client.post("/api/items", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return view(response.get_json())


def decrement(client, item, amount):
    return client.post(f"/api/items/{item['id']}/decrement", json={"amount": amount})


def quantity_of(client, item):
    return client.get(f"/api/items/{item['id']}").get_json()["quantity"]


def test_create_returns_201_and_the_item_with_an_integer_id(client):
    item = create(client, name="Screws", quantity=12)
    assert isinstance(item["id"], int)
    assert item["name"] == "Screws"
    assert item["quantity"] == 12


def test_quantity_defaults_to_zero(client):
    assert create(client, name="Nuts")["quantity"] == 0


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/items")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, name="One")
    second = create(client, name="Two")
    items = client.get("/api/items").get_json()
    assert isinstance(items, list)
    assert sorted(i["id"] for i in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, name="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/items").get_json() == []


def test_get_one(client):
    item = create(client, name="Washers", quantity=3)
    response = client.get(f"/api/items/{item['id']}")
    assert response.status_code == 200
    assert view(response.get_json()) == item


def test_update(client):
    item = create(client, name="Draft")
    body = {"name": "Final", "quantity": 40}
    response = client.put(f"/api/items/{item['id']}", json=body)
    assert response.status_code == 200
    assert view(response.get_json()) == {"id": item["id"], **body}
    assert view(client.get(f"/api/items/{item['id']}").get_json()) == {"id": item["id"], **body}


def test_partial_update_keeps_the_fields_left_out(client):
    item = create(client, name="Keep me", quantity=2)
    response = client.put(f"/api/items/{item['id']}", json={"quantity": 9})
    assert response.status_code == 200
    expected = {"id": item["id"], "name": "Keep me", "quantity": 9}
    assert view(response.get_json()) == expected
    assert view(client.get(f"/api/items/{item['id']}").get_json()) == expected


@pytest.mark.parametrize(
    "body", [{"name": ""}, {"quantity": -1}, {"quantity": 2.5}, {"quantity": "3"}]
)
def test_update_that_breaks_a_rule_is_400(client, body):
    item = create(client, name="Valid", quantity=4)
    response = client.put(f"/api/items/{item['id']}", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert view(client.get(f"/api/items/{item['id']}").get_json()) == item


def test_ids_are_never_reused(client):
    create(client, name="A")
    b = create(client, name="B")
    assert client.delete(f"/api/items/{b['id']}").status_code in (200, 204)
    c = create(client, name="C")
    assert c["id"] != b["id"]


def test_delete_then_404(client):
    item = create(client, name="Temporary")
    response = client.delete(f"/api/items/{item['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/items/{item['id']}").status_code == 404
    assert client.get("/api/items").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/items/999").status_code == 404
    assert client.put("/api/items/999", json={"name": "x"}).status_code == 404
    assert client.delete("/api/items/999").status_code == 404
    assert client.post("/api/items/999/decrement", json={"amount": 1}).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"quantity": 1},
        {"name": ""},
        {"name": "   "},
        {"name": 3},
        {"name": "x", "quantity": -1},
        {"name": "x", "quantity": 1.5},
        {"name": "x", "quantity": "4"},
        {"name": "x", "quantity": True},
        {"name": "x", "quantity": None},
    ],
)
def test_invalid_item_is_400_with_an_error(client, body):
    response = client.post("/api/items", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get("/api/items").get_json() == []


def test_decrement_returns_the_item_with_the_lower_quantity(client):
    item = create(client, name="Bolts", quantity=10)
    response = decrement(client, item, 4)
    assert response.status_code == 200
    assert view(response.get_json()) == {"id": item["id"], "name": "Bolts", "quantity": 6}
    assert quantity_of(client, item) == 6


def test_decrement_to_exactly_zero_is_allowed(client):
    item = create(client, name="Bolts", quantity=3)
    response = decrement(client, item, 3)
    assert response.status_code == 200
    assert response.get_json()["quantity"] == 0


def test_decrement_below_zero_is_409_and_changes_nothing(client):
    item = create(client, name="Bolts", quantity=3)
    response = decrement(client, item, 4)
    assert response.status_code == 409
    assert "error" in response.get_json()
    assert quantity_of(client, item) == 3
    empty = create(client, name="Empty")
    assert decrement(client, empty, 1).status_code == 409
    assert quantity_of(client, empty) == 0


@pytest.mark.parametrize("body", [{}, {"amount": 0}, {"amount": -2}, {"amount": 1.5}, {"amount": "1"}, {"amount": True},
                                    {"amount": None}])
def test_invalid_decrement_amount_is_400(client, body):
    item = create(client, name="Bolts", quantity=5)
    response = client.post(f"/api/items/{item['id']}/decrement", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert quantity_of(client, item) == 5
