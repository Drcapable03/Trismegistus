from pathlib import Path

import yaml

from config.settings import ROOT

_ALIASES_PATH = ROOT / "config" / "team_aliases.yaml"


def _load_aliases() -> dict:
    if not _ALIASES_PATH.exists():
        return {"understat": {}, "club_elo": {}}
    with open(_ALIASES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_understat(team: str) -> str:
    return _load_aliases().get("understat", {}).get(team, team)


def to_club_elo(team: str) -> str:
    return _load_aliases().get("club_elo", {}).get(team, team)


def from_understat(title: str) -> str:
    mapping = _load_aliases().get("understat", {})
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(title, title)