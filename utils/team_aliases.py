import re
from pathlib import Path

import yaml

from config.settings import ROOT

_ALIASES_PATH = ROOT / "config" / "team_aliases.yaml"


def _load_aliases() -> dict:
    if not _ALIASES_PATH.exists():
        return {"understat": {}, "club_elo": {}, "oddsportal": {}, "fbref": {}, "statsbomb": {}}
    with open(_ALIASES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_understat(team: str) -> str:
    return _load_aliases().get("understat", {}).get(team, team)


def to_club_elo(team: str) -> str:
    return _load_aliases().get("club_elo", {}).get(team, team)


def to_oddsportal(team: str) -> str:
    return _load_aliases().get("oddsportal", {}).get(team, team)


def from_understat(title: str) -> str:
    mapping = _load_aliases().get("understat", {})
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(title, title)


def from_oddsportal(title: str) -> str:
    mapping = _load_aliases().get("oddsportal", {})
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(title, title)


def from_fbref(title: str) -> str:
    mapping = _load_aliases().get("fbref", {})
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(title, title)


def from_statsbomb(title: str) -> str:
    mapping = _load_aliases().get("statsbomb", {})
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(title, title)


def _normalize_team(name: str) -> str:
    n = str(name).lower().strip()
    n = n.replace("'", "").replace(".", "").replace("&", "and")
    n = re.sub(r"\s+", " ", n)
    for old, new in (
        ("manchester utd", "manchester united"),
        ("man utd", "manchester united"),
        ("man united", "manchester united"),
        ("man city", "manchester city"),
        ("newcastle utd", "newcastle"),
        ("nottm forest", "nottingham"),
        ("nottingham forest", "nottingham"),
        ("spurs", "tottenham"),
        ("ath madrid", "atletico madrid"),
        ("ath bilbao", "athletic club"),
        ("paris sg", "paris saint germain"),
        ("psg", "paris saint germain"),
    ):
        if n == old or n.startswith(old + " "):
            n = n.replace(old, new, 1)
    return n.strip()


def team_variants(team: str) -> set[str]:
    names = {team, to_oddsportal(team), from_oddsportal(team)}
    return {_normalize_team(n) for n in names if n}


def teams_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left.strip().lower() == right.strip().lower():
        return True
    return bool(team_variants(left) & team_variants(right))