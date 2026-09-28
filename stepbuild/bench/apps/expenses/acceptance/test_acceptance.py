"""Hidden acceptance tests for the expenses app (never shown to the model).

They import create_app from the finished project's backend (on PYTHONPATH) and
define their own client, so nothing the model wrote in backend/tests can affect
them. Delete may answer 200 or 204; errors are checked for an `error` key only.
The spec says booleans are not numbers, so `amount: true` is expected to be 400.
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
    body.setdefault("amount", 10)
    response = client.post("/api/expenses", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return response.get_json()


def test_create_returns_201_and_the_expense_with_an_integer_id(client):
    expense = create(client, description="Train", amount=4.75, category="travel")
    assert isinstance(expense["id"], int)
    assert expense["description"] == "Train"
    assert expense["amount"] == 4.75
    assert expense["category"] == "travel"


def test_category_defaults_to_general(client):
    assert create(client, description="Coffee")["category"] == "general"


@pytest.mark.parametrize("amount", [1, 0.01, 250.5])
def test_positive_integer_and_decimal_amounts_are_accepted(client, amount):
    assert create(client, description="x", amount=amount)["amount"] == amount


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/expenses")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, description="One")
    second = create(client, description="Two")
    items = client.get("/api/expenses").get_json()
    assert isinstance(items, list)
    assert sorted(e["id"] for e in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, description="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/expenses").get_json() == []


def test_get_one(client):
    expense = create(client, description="Books", amount=30, category="study")
    response = client.get(f"/api/expenses/{expense['id']}")
    assert response.status_code == 200
    assert response.get_json() == expense


def test_update(client):
    expense = create(client, description="Draft")
    body = {"description": "Final", "amount": 99.5, "category": "misc"}
    response = client.put(f"/api/expenses/{expense['id']}", json=body)
    assert response.status_code == 200
    assert response.get_json() == {"id": expense["id"], **body}
    assert client.get(f"/api/expenses/{expense['id']}").get_json() == {"id": expense["id"], **body}


def test_partial_update_keeps_the_fields_left_out(client):
    expense = create(client, description="Keep me", amount=5, category="food")
    response = client.put(f"/api/expenses/{expense['id']}", json={"amount": 6})
    assert response.status_code == 200
    expected = {"id": expense["id"], "description": "Keep me", "amount": 6, "category": "food"}
    assert response.get_json() == expected
    assert client.get(f"/api/expenses/{expense['id']}").get_json() == expected


@pytest.mark.parametrize(
    "body", [{"description": ""}, {"amount": -3}, {"amount": 0}, {"amount": "7"}, {"category": 1}]
)
def test_update_that_breaks_a_rule_is_400(client, body):
    expense = create(client, description="Valid")
    response = client.put(f"/api/expenses/{expense['id']}", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get(f"/api/expenses/{expense['id']}").get_json() == expense


def test_ids_are_never_reused(client):
    create(client, description="A")
    b = create(client, description="B")
    assert client.delete(f"/api/expenses/{b['id']}").status_code in (200, 204)
    c = create(client, description="C")
    assert c["id"] != b["id"]


def test_delete_then_404(client):
    expense = create(client, description="Temporary")
    response = client.delete(f"/api/expenses/{expense['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/expenses/{expense['id']}").status_code == 404
    assert client.get("/api/expenses").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/expenses/999").status_code == 404
    assert client.put("/api/expenses/999", json={"amount": 1}).status_code == 404
    assert client.delete("/api/expenses/999").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"amount": 5},
        {"description": "", "amount": 5},
        {"description": "   ", "amount": 5},
        {"description": 3, "amount": 5},
        {"description": "x"},
        {"description": "x", "amount": -0.5},
        {"description": "x", "amount": -10},
        {"description": "x", "amount": 0},
        {"description": "x", "amount": "5"},
        {"description": "x", "amount": True},
        {"description": "x", "amount": None},
        {"description": "x", "amount": 5, "category": 12},
    ],
)
def test_invalid_expense_is_400_with_an_error(client, body):
    response = client.post("/api/expenses", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get("/api/expenses").get_json() == []
