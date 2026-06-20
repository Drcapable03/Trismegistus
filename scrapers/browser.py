"""Shared Scrapling fetch helpers with stealth defaults."""

import os

from scrapling.fetchers import Fetcher, StealthyFetcher

DEFAULT_TIMEOUT_MS = 60_000


def use_stealth() -> bool:
    return os.getenv("TRIS_SCRAPE_STEALTH", "true").lower() in {"1", "true", "yes"}


def stealth_fetch(url: str, **kwargs):
    """StealthyFetcher with anti-bot defaults for protected sites."""
    opts = {
        "headless": os.getenv("TRIS_SCRAPE_HEADLESS", "true").lower() != "false",
        "network_idle": True,
        "timeout": DEFAULT_TIMEOUT_MS,
        "block_ads": True,
        "google_search": True,
    }
    opts.update(kwargs)
    return StealthyFetcher.fetch(url, **opts)


def light_fetch(url: str, **kwargs):
    """Lightweight HTTP fetch for simple pages."""
    return Fetcher.get(url, timeout=30, **kwargs)


def fetch_page(url: str, force_stealth: bool = False):
    if force_stealth or use_stealth():
        return stealth_fetch(url)
    try:
        return light_fetch(url)
    except Exception:
        return stealth_fetch(url)