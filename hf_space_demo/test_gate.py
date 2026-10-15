"""Gate tests without Modal, Gradio or Cloudflare: a fake siteverify and a fake clock."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gate as g


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


def make(ok=True):
    clock = Clock()

    async def verify(token, ip):
        return ok and token == "good"

    gt = g.Gate("site-key", "secret-key", now=clock, verify=verify)
    app = FastAPI()
    gt.install(app)

    @app.get("/")
    async def home():
        return {"chat": True}

    @app.post("/gradio_api/queue/join")
    async def join():
        return {"queued": True}

    return gt, clock, TestClient(app, base_url="https://testserver")


def passed(client):
    r = client.post("/verify", data={"cf-turnstile-response": "good"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    return r


def test_browser_without_pass_goes_to_check():
    _, _, c = make()
    r = c.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/verify"
    page = c.get("/verify").text
    assert 'data-sitekey="site-key"' in page and "secret-key" not in page


def test_api_without_pass_is_refused():
    _, _, c = make()
    assert c.post("/gradio_api/queue/join").status_code == 403
    assert c.get("/", headers={"accept": "application/json"}).status_code == 403


def test_failed_check_sets_no_cookie():
    _, _, c = make(ok=False)
    r = c.post("/verify", data={"cf-turnstile-response": "good"}, follow_redirects=False)
    assert r.status_code == 200 and "check didn&#x27;t go through" in r.text
    assert g.COOKIE not in r.cookies


def test_pass_opens_chat_and_api():
    _, _, c = make()
    passed(c)
    assert c.get("/").json() == {"chat": True}
    assert c.post("/gradio_api/queue/join").json() == {"queued": True}


def test_forged_and_expired_passes_fail():
    gt, clock, c = make()
    good = gt.issue()
    sid, exp, mac = good.split(".")
    assert gt.check(good) == sid
    assert gt.check(f"{sid}.{int(exp) + 999}.{mac}") is None          # stretched expiry
    assert gt.check("garbage") is None and gt.check(None) is None
    other = g.Gate("site-key", "another-secret", now=clock)
    assert other.check(good) is None                                  # different secret
    clock.t += g.PASS_SECONDS + 1
    assert gt.check(good) is None                                     # expired


def test_hourly_limit_per_pass():
    _, clock, c = make()
    passed(c)
    for _ in range(g.PER_PASS_PER_HOUR):
        assert c.post("/gradio_api/queue/join").status_code == 200
    assert c.post("/gradio_api/queue/join").status_code == 429
    clock.t += 3601
    assert c.post("/gradio_api/queue/join").status_code == 200


def test_daily_limit_across_everyone():
    gt, clock, _ = make()
    gt.day_count = g.PER_DAY
    gt.day = __import__("time").strftime("%Y-%m-%d", __import__("time").gmtime(clock.t))
    assert "daily limit" in gt.allow_reply("someone")
    clock.t += 86400
    assert gt.allow_reply("someone") is None
