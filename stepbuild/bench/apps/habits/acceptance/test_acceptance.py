"""Hidden acceptance tests for the habits app (never shown to the model).

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
    response = client.post("/api/habits", json=body)
    assert response.status_code == 201, response.get_data(as_text=True)
    return response.get_json()


def checkins_url(habit):
    return f"/api/habits/{habit['id']}/checkins"


def test_create_returns_201_and_the_habit_with_an_integer_id(client):
    habit = create(client, name="Read", description="20 pages")
    assert isinstance(habit["id"], int)
    assert habit["name"] == "Read"
    assert habit["description"] == "20 pages"


def test_description_defaults_to_an_empty_string(client):
    assert create(client, name="Walk")["description"] == ""


def test_list_starts_empty_and_returns_an_array(client):
    response = client.get("/api/habits")
    assert response.status_code == 200
    assert response.get_json() == []
    first = create(client, name="One")
    second = create(client, name="Two")
    items = client.get("/api/habits").get_json()
    assert isinstance(items, list)
    assert sorted(h["id"] for h in items) == sorted([first["id"], second["id"]])
    assert first["id"] != second["id"]


def test_each_app_starts_empty(client):
    create(client, name="Only in this app")
    other = create_app()
    other.config["TESTING"] = True
    assert other.test_client().get("/api/habits").get_json() == []


def test_get_one(client):
    habit = create(client, name="Meditate", description="morning")
    response = client.get(f"/api/habits/{habit['id']}")
    assert response.status_code == 200
    assert response.get_json() == habit


def test_update(client):
    habit = create(client, name="Draft")
    body = {"name": "Final", "description": "new"}
    response = client.put(f"/api/habits/{habit['id']}", json=body)
    assert response.status_code == 200
    assert response.get_json() == {"id": habit["id"], **body}
    assert client.get(f"/api/habits/{habit['id']}").get_json() == {"id": habit["id"], **body}


def test_partial_update_keeps_the_fields_left_out(client):
    habit = create(client, name="Keep me", description="old")
    response = client.put(f"/api/habits/{habit['id']}", json={"description": "x"})
    assert response.status_code == 200
    expected = {"id": habit["id"], "name": "Keep me", "description": "x"}
    assert response.get_json() == expected
    assert client.get(f"/api/habits/{habit['id']}").get_json() == expected


@pytest.mark.parametrize("body", [{"name": ""}, {"name": "   "}, {"description": 3}])
def test_update_that_breaks_a_rule_is_400(client, body):
    habit = create(client, name="Valid")
    response = client.put(f"/api/habits/{habit['id']}", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get(f"/api/habits/{habit['id']}").get_json() == habit


def test_ids_are_never_reused(client):
    create(client, name="A")
    b = create(client, name="B")
    assert client.delete(f"/api/habits/{b['id']}").status_code in (200, 204)
    c = create(client, name="C")
    assert c["id"] != b["id"]


def test_delete_then_404(client):
    habit = create(client, name="Temporary")
    response = client.delete(f"/api/habits/{habit['id']}")
    assert response.status_code in (200, 204)
    assert client.get(f"/api/habits/{habit['id']}").status_code == 404
    assert client.get("/api/habits").get_json() == []


def test_missing_ids_are_404(client):
    assert client.get("/api/habits/999").status_code == 404
    assert client.put("/api/habits/999", json={"name": "x"}).status_code == 404
    assert client.delete("/api/habits/999").status_code == 404
    assert client.get("/api/habits/999/checkins").status_code == 404
    assert client.post("/api/habits/999/checkins", json={"date": "2026-09-28"}).status_code == 404


@pytest.mark.parametrize(
    "body", [{"description": "no name"}, {"name": ""}, {"name": "  "}, {"name": 4}, {"name": "x", "description": 4}]
)
def test_invalid_habit_is_400_with_an_error(client, body):
    response = client.post("/api/habits", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get("/api/habits").get_json() == []


def test_first_checkin_is_201_with_the_record(client):
    habit = create(client, name="Run")
    response = client.post(checkins_url(habit), json={"date": "2026-09-28"})
    assert response.status_code == 201
    checkin = response.get_json()
    assert isinstance(checkin["id"], int)
    assert checkin["habit_id"] == habit["id"]
    assert checkin["date"] == "2026-09-28"


def test_checkin_is_idempotent_per_day(client):
    habit = create(client, name="Run")
    first = client.post(checkins_url(habit), json={"date": "2026-09-28"})
    again = client.post(checkins_url(habit), json={"date": "2026-09-28"})
    assert first.status_code == 201
    assert again.status_code == 200
    assert again.get_json() == first.get_json()
    listed = client.get(checkins_url(habit))
    assert listed.status_code == 200
    assert listed.get_json() == [first.get_json()]


def test_checkins_on_different_days_and_habits_are_separate(client):
    run = create(client, name="Run")
    read = create(client, name="Read")
    a = client.post(checkins_url(run), json={"date": "2026-09-28"}).get_json()
    b = client.post(checkins_url(run), json={"date": "2026-09-29"}).get_json()
    other = client.post(checkins_url(read), json={"date": "2026-09-28"})
    assert other.status_code == 201
    assert len({a["id"], b["id"], other.get_json()["id"]}) == 3
    dates = sorted(c["date"] for c in client.get(checkins_url(run)).get_json())
    assert dates == ["2026-09-28", "2026-09-29"]
    assert [c["habit_id"] for c in client.get(checkins_url(read)).get_json()] == [read["id"]]


def test_a_new_habit_has_no_checkins(client):
    assert client.get(checkins_url(create(client, name="Fresh"))).get_json() == []


@pytest.mark.parametrize(
    "body", [{}, {"date": ""}, {"date": 20260928}, {"date": "28/09/2026"}, {"date": "2026-9-28"}, {"date": "2026-02-30"}]
)
def test_invalid_checkin_date_is_400(client, body):
    habit = create(client, name="Run")
    response = client.post(checkins_url(habit), json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()
    assert client.get(checkins_url(habit)).get_json() == []
