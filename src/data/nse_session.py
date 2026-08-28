"""
Cookie-bootstrapped session for the nseindia.com JSON APIs (as opposed to the
nsearchives.nseindia.com static-file archive, which needs no session at all).

NSE's WAF returns 403 on a cold GET of the homepage but still sets a valid
session cookie on that 403 response -- hitting the homepage once and reusing
the resulting cookies is standard practice for these APIs (the same trick
used by nsepython/jugaad-data) and is what makes /api/corporates-* work.
"""
from __future__ import annotations

import time

import requests

from src.data.ingest.bhavcopy import USER_AGENT

BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def get_nse_session(timeout: float = 15.0) -> requests.Session:
    s = requests.Session()
    s.headers.update(BASE_HEADERS)
    s.get("https://www.nseindia.com", timeout=timeout)  # bootstraps cookies even on 403
    return s


def nse_api_get(session: requests.Session, url: str, referer: str, timeout: float = 15.0,
                 max_retries: int = 3):
    """GET an nseindia.com /api/ endpoint, re-bootstrapping the session once if it's gone stale."""
    headers = {**BASE_HEADERS, "Accept": "application/json", "Referer": referer}
    for attempt in range(max_retries):
        r = session.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                pass  # fall through to retry
        if r.status_code in (401, 403):
            session.get("https://www.nseindia.com", timeout=timeout)  # refresh cookies
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"nse_api_get failed after {max_retries} attempts: {url}")
