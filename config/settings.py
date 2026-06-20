import os
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

load_dotenv(ROOT / ".env")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def today() -> datetime:
    return datetime.now()


def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def load_leagues_config() -> dict:
    config_path = ROOT / "config" / "leagues.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def enabled_leagues() -> dict[str, dict]:
    cfg = load_leagues_config()
    leagues = {
        name: info
        for name, info in cfg.get("leagues", {}).items()
        if info.get("enabled", True)
    }
    max_active = cfg.get("max_active_leagues", 99)
    if len(leagues) > max_active:
        raise ValueError(
            f"Too many enabled leagues ({len(leagues)}). "
            f"max_active_leagues is {max_active}. Disable some in config/leagues.yaml."
        )
    return leagues


def league_urls() -> dict[str, str]:
    """Return download URLs for all enabled leagues across current + historical seasons."""
    cfg = load_leagues_config()
    seasons = [cfg["season"], *cfg.get("historical_seasons", [])]
    base = "https://www.football-data.co.uk/mmz4281/{season}"
    urls = {}
    for name, info in enabled_leagues().items():
        code = info["code"]
        for season in seasons:
            label = f"{name} ({season})" if len(seasons) > 1 else name
            urls[label] = f"{base.format(season=season)}/{code}.csv"
    return urls


def fixtures_url() -> str:
    return load_leagues_config()["fixtures_url"]


def league_div_codes() -> list[str]:
    return [info["code"] for info in enabled_leagues().values()]


def bookie_blend_weight() -> float:
    cfg = load_leagues_config()
    model_cfg = cfg.get("model") or {}
    if model_cfg.get("bookie_blend_weight") is not None:
        return float(model_cfg["bookie_blend_weight"])
    return float(get_env("BOOKIE_BLEND_WEIGHT", "0.55"))


def set_bookie_blend_weight(weight: float) -> None:
    """Persist tuned blend weight to config/leagues.yaml without rewriting the file."""
    import re

    config_path = ROOT / "config" / "leagues.yaml"
    text = config_path.read_text(encoding="utf-8")
    value = round(float(weight), 3)
    if re.search(r"^\s*bookie_blend_weight:\s*[\d.]+", text, re.MULTILINE):
        text = re.sub(
            r"^(\s*bookie_blend_weight:\s*)[\d.]+",
            rf"\g<1>{value}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        anchor = "model:\n"
        if anchor not in text:
            text = text.rstrip() + f"\n\nmodel:\n  bookie_blend_weight: {value}\n"
        else:
            text = text.replace(anchor, f"{anchor}  bookie_blend_weight: {value}\n", 1)
    config_path.write_text(text, encoding="utf-8")


def league_summary() -> str:
    cfg = load_leagues_config()
    active = list(enabled_leagues().keys())
    inactive = [
        n for n, i in cfg.get("leagues", {}).items() if not i.get("enabled", True)
    ]
    seasons = [cfg["season"], *cfg.get("historical_seasons", [])]
    return (
        f"Active leagues ({len(active)}): {', '.join(active)}\n"
        f"Seasons: {', '.join(seasons)}\n"
        f"Inactive (ready to enable): {len(inactive)}"
    )