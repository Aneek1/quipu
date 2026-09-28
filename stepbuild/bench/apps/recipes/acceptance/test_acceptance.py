"""Hidden acceptance tests for the recipes app (never shown to the model).

They import create_app from the finished project's backend (on PYTHONPATH) and
define their own client, so nothing the model wrote in backend/tests can affect
them. Delete may answer 200 or 204; errors are checked for an `error` key only.
"""
import pytest

from app import create_app


FIELDS = ("id", "title", "ingredients", "minutes")


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
    response = client.post("/api/recipes", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return view(response.get_json())


def test_create_returns_201_and_the_recipe_with_an_integer_id(client):
    recipe = create(client, title="Pancakes", ingredients=["flour", "milk"], minutes=20)
    assert isinstance(recipe["id"], int)
    assert recipe["title"] == "Pancakes"
    assert recipe["ingredients"] == ["flour", "milk"]
    assert recipe["minutes"] == 20


def test_defaults_fill_in_ingredients_and_minutes(client):
    recipe = create(client, title="Toast")
    assert recipe["ingredients"] == []
    assert recipe["minutes"] == 0


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/recipes")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, title="One")
    second = create(client, title="Two")
    items = client.get("/api/recipes").get_json()
    assert isinstance(items, list)
    assert sorted(r["id"] for r in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, title="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/recipes").get_json() == []


def test_get_one(client):
    recipe = create(client, title="Soup", ingredients=["water"], minutes=30)
    response = client.get(f"/api/recipes/{recipe['id']}")
    assert response.status_code == 200
    assert view(response.get_json()) == recipe


def test_update(client):
    recipe = create(client, title="Draft")
    body = {"title": "Final", "ingredients": ["salt"], "minutes": 5}
    response = client.put(f"/api/recipes/{recipe['id']}", json=body)
    assert response.status_code == 200
    assert view(response.get_json()) == {"id": recipe["id"], **body}
    assert view(client.get(f"/api/recipes/{recipe['id']}").get_json()) == {"id": recipe["id"], **body}


def test_partial_update_keeps_the_fields_left_out(client):
    recipe = create(client, title="Keep me", ingredients=["egg"], minutes=10)
    response = client.put(f"/api/recipes/{recipe['id']}", json={"minutes": 12})
    assert response.status_code == 200
    expected = {"id": recipe["id"], "title": "Keep me", "ingredients": ["egg"], "minutes": 12}
    assert view(response.get_json()) == expected
    assert view(client.get(f"/api/recipes/{recipe['id']}").get_json()) == expected


@pytest.mark.parametrize(
    "body", [{"title": ""}, {"minutes": -1}, {"minutes": "5"}, {"ingredients": [3]}]
)
def test_update_that_breaks_a_rule_is_400(client, body):
    recipe = create(client, title="Valid", minutes=3)
    response = client.put(f"/api/recipes/{recipe['id']}", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert view(client.get(f"/api/recipes/{recipe['id']}").get_json()) == recipe


def test_ids_are_never_reused(client):
    create(client, title="A")
    b = create(client, title="B")
    assert client.delete(f"/api/recipes/{b['id']}").status_code in (200, 204)
    c = create(client, title="C")
    assert c["id"] != b["id"]


def test_delete_then_404(client):
    recipe = create(client, title="Temporary")
    response = client.delete(f"/api/recipes/{recipe['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/recipes/{recipe['id']}").status_code == 404
    assert client.get("/api/recipes").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/recipes/999").status_code == 404
    assert client.put("/api/recipes/999", json={"title": "x"}).status_code == 404
    assert client.delete("/api/recipes/999").status_code == 404


def test_zero_minutes_is_allowed(client):
    assert create(client, title="Salad", minutes=0)["minutes"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"minutes": 5},
        {"title": ""},
        {"title": "   "},
        {"title": 7},
        {"title": "x", "ingredients": "flour, milk"},
        {"title": "x", "ingredients": ["flour", 2]},
        {"title": "x", "ingredients": ["flour", "  "]},
        {"title": "x", "minutes": -5},
        {"title": "x", "minutes": 2.5},
        {"title": "x", "minutes": "10"},
        {"title": "x", "minutes": True},
        {"title": "x", "minutes": None},
        {"title": "x", "ingredients": None},
    ],
)
def test_invalid_recipe_is_400_with_an_error(client, body):
    response = client.post("/api/recipes", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get("/api/recipes").get_json() == []
