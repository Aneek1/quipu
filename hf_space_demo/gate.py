"""Cloudflare Turnstile in front of the whole demo, plus a reply cap.

A visitor without a valid pass cookie gets the Turnstile page; solving it sets a signed
cookie (12 h) and sends them on to the chat. Every other path needs the cookie too, so
Gradio's API can't be scripted around the page. Replies (Gradio queue joins) are capped
per pass and per day, so a script that does get through can't run up the bill.

Counters live in memory: the demo runs in at most one container, and they reset when it
scales to zero, which only happens after 10 idle minutes."""
import hashlib
import hmac
import html
import secrets
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

COOKIE = "quipu_pass"
PASS_SECONDS = 12 * 3600
PER_PASS_PER_HOUR = 30       # replies one visitor can ask for in an hour
PER_DAY = 1000               # replies across everyone in a UTC day
SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quipu chat</title>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
<style>body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0b0c;
color:#e8e6e3;font:15px/1.5 system-ui,sans-serif}}main{{text-align:center;padding:16px}}
p{{color:#9a9894}}.err{{color:#f0a646}}</style></head><body><main>
<h1 style="font-weight:600;font-size:20px">Quipu chat</h1>
<p>A quick check that you're a person, then the chat opens.</p>{error}
<form id="f" method="post" action="/verify">
<div class="cf-turnstile" data-sitekey="{site_key}" data-callback="done"></div></form>
<script>function done(){{document.getElementById("f").submit()}}</script>
</main></body></html>"""


def _sign(key: bytes, sid: str, exp: int) -> str:
    return hmac.new(key, f"{sid}.{exp}".encode(), hashlib.sha256).hexdigest()


class Gate:
    def __init__(self, site_key: str, secret_key: str, now=time.time, verify=None):
        self.site_key, self.secret = site_key, secret_key
        self.key = hashlib.sha256(b"quipu-pass:" + secret_key.encode()).digest()
        self.now = now
        self.verify = verify or self._siteverify
        self.hourly: dict[str, list[float]] = {}
        self.day, self.day_count = "", 0

    async def _siteverify(self, token: str, ip: str | None) -> bool:
        data = {"secret": self.secret, "response": token}
        if ip:
            data["remoteip"] = ip
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(SITEVERIFY, data=data)
        out = r.json()
        if not out.get("success"):    # e.g. invalid-input-secret after a key rotation
            print("turnstile check failed:", out.get("error-codes"), out.get("hostname"), flush=True)
        return bool(out.get("success"))

    def issue(self) -> str:
        sid, exp = secrets.token_hex(8), int(self.now()) + PASS_SECONDS
        return f"{sid}.{exp}.{_sign(self.key, sid, exp)}"

    def check(self, value: str | None) -> str | None:
        """The pass's session id if the cookie is genuine and unexpired, else None."""
        try:
            sid, exp_s, mac = (value or "").split(".")
            exp = int(exp_s)
        except ValueError:
            return None
        if not hmac.compare_digest(mac, _sign(self.key, sid, exp)) or exp < self.now():
            return None
        return sid

    def allow_reply(self, sid: str) -> str | None:
        """None if this reply may run, else the reason it can't."""
        now = self.now()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self.day:
            self.day, self.day_count = day, 0
        recent = [t for t in self.hourly.get(sid, []) if t > now - 3600]
        if self.day_count >= PER_DAY:
            return "The demo has reached its daily limit. Please try again tomorrow."
        if len(recent) >= PER_PASS_PER_HOUR:
            return "You've reached the hourly limit for this demo. Please try again later."
        recent.append(now)
        self.hourly[sid] = recent
        self.day_count += 1
        return None

    def page(self, error: str = "") -> HTMLResponse:
        err = f'<p class="err">{html.escape(error)}</p>' if error else ""
        return HTMLResponse(PAGE.format(site_key=html.escape(self.site_key), error=err))

    def install(self, app: FastAPI) -> None:
        @app.get("/verify")
        async def verify_page():
            return self.page()

        @app.post("/verify")
        async def verify(request: Request):
            form = await request.form()
            token = str(form.get("cf-turnstile-response") or "")
            ip = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else None)
            if not token or not await self.verify(token, ip):
                return self.page("The check didn't go through. Please try again.")
            resp = RedirectResponse("/", status_code=303)
            resp.set_cookie(COOKIE, self.issue(), max_age=PASS_SECONDS, httponly=True,
                            secure=True, samesite="lax")
            return resp

        @app.middleware("http")
        async def guard(request: Request, call_next):
            path = request.url.path
            if path == "/verify":
                return await call_next(request)
            sid = self.check(request.cookies.get(COOKIE))
            if sid is None:
                if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
                    return RedirectResponse("/verify", status_code=303)
                return PlainTextResponse("Open the demo in a browser to pass the check first.", 403)
            if request.method == "POST" and path.endswith("/queue/join"):
                reason = self.allow_reply(sid)
                if reason:
                    return PlainTextResponse(reason, 429)
            return await call_next(request)
