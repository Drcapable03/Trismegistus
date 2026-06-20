import os
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

load_dotenv(ROOT / ".env")


def today() -> datetime:
    return datetime.now()


def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def load_leagues_config() -> dict:
    config_path = ROOT / "config" / "leagues.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def league_urls() -> dict[str, str]:
    cfg = load_leagues_config()
    season = cfg["season"]
    base = f"https://www.football-data.co.uk/mmz4281/{season}"
    return {
        name: f"{base}/{info['code']}.csv"
        for name, info in cfg["leagues"].items()
    }


def fixtures_url() -> str:
    return load_leagues_config()["fixtures_url"]