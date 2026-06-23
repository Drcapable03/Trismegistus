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


def current_season() -> str:
    return str(load_leagues_config()["season"])


def historical_seasons() -> list[str]:
    return [str(s) for s in load_leagues_config().get("historical_seasons", [])]


def all_seasons() -> list[str]:
    return [current_season(), *historical_seasons()]


def football_data_league_url(season: str, div_code: str) -> str:
    base = "https://www.football-data.co.uk/mmz4281/{season}"
    return f"{base.format(season=season)}/{div_code}.csv"


def league_urls() -> dict[str, str]:
    """Return download URLs for all enabled leagues across current + historical seasons."""
    seasons = all_seasons()
    urls = {}
    for name, info in enabled_leagues().items():
        code = info["code"]
        for season in seasons:
            label = f"{name} ({season})" if len(seasons) > 1 else name
            urls[label] = football_data_league_url(season, code)
    return urls


def fixtures_url() -> str:
    return load_leagues_config()["fixtures_url"]


def league_div_codes() -> list[str]:
    return [info["code"] for info in enabled_leagues().values()]


def oddsportal_league_urls(div_codes: list[str] | None = None) -> dict[str, dict[str, str]]:
    """OddsPortal fixture/results URLs keyed by football-data Div code."""
    cfg = load_leagues_config()
    portal = cfg.get("oddsportal") or {}
    codes = div_codes or league_div_codes()
    return {code: portal[code] for code in codes if code in portal}


def chaos_config() -> dict:
    cfg = load_leagues_config()
    return cfg.get("chaos") or {}


def use_intel_in_train() -> bool:
    chaos = chaos_config()
    if "use_intel_in_train" in chaos:
        return bool(chaos["use_intel_in_train"])
    if "use_sentiment_in_train" in chaos:
        return bool(chaos["use_sentiment_in_train"])
    return get_env("USE_INTEL_IN_TRAIN", get_env("USE_SENTIMENT_IN_TRAIN", "false")).lower() in {
        "1", "true", "yes",
    }


def use_sentiment_in_train() -> bool:
    """Legacy alias for use_intel_in_train."""
    return use_intel_in_train()


def pit_cache_intel() -> bool:
    chaos = chaos_config()
    if "pit_cache_intel" in chaos:
        return bool(chaos["pit_cache_intel"])
    if "pit_cache_sentiment" in chaos:
        return bool(chaos["pit_cache_sentiment"])
    return get_env("PIT_CACHE_INTEL", get_env("PIT_CACHE_SENTIMENT", "true")).lower() in {
        "1", "true", "yes",
    }


def pit_cache_sentiment() -> bool:
    """Legacy alias for pit_cache_intel."""
    return pit_cache_intel()


def intel_config() -> dict:
    path = ROOT / "config" / "intel.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def intel_calibration_probes() -> list[dict]:
    cfg = intel_config().get("calibration") or {}
    probes = cfg.get("sample_probes") or []
    return [p for p in probes if p.get("team")]


def youtube_channels_for_div(div_code: str | None) -> list[str]:
    if not div_code:
        return []
    youtube = intel_config().get("youtube") or {}
    channels = youtube.get("channels_by_div") or {}
    return list(channels.get(div_code, []))


def devig_method() -> str:
    cfg = load_leagues_config()
    model_cfg = cfg.get("model") or {}
    method = str(model_cfg.get("devig_method", get_env("DEVIG_METHOD", "shin"))).lower()
    return method if method in {"shin", "proportional"} else "shin"


def understat_league_map() -> dict[str, str]:
    cfg = load_leagues_config()
    return {
        "E0": "EPL",
        "SP1": "La_Liga",
        "D1": "Bundesliga",
        "I1": "Serie_A",
        "F1": "Ligue_1",
        **(cfg.get("understat_leagues") or {}),
    }


def enrichment_config() -> dict:
    cfg = load_leagues_config()
    return cfg.get("enrichment") or {}


def fbref_competition_map() -> dict[str, str]:
    cfg = enrichment_config()
    defaults = {
        "E0": "ENG Premier League",
        "SP1": "ESP La Liga",
        "D1": "DEU Bundesliga 1",
        "I1": "ITA Serie A",
        "F1": "FRA Ligue 1",
    }
    raw = cfg.get("fbref_competitions") or {}
    return {**defaults, **{str(k): str(v) for k, v in raw.items()}}


def fbref_season_label(season_code: str) -> str:
    s = str(season_code)
    if len(s) == 4 and s.isdigit():
        return f"20{s[:2]}-20{s[2:]}"
    return s


def statsbomb_competition_to_div() -> dict[str, str]:
    return {
        "Premier League": "E0",
        "La Liga": "SP1",
        "1. Bundesliga": "D1",
        "Serie A": "I1",
        "Ligue 1": "F1",
    }


def xg_source_priority() -> list[str]:
    cfg = enrichment_config()
    raw = cfg.get("xg_source_priority")
    if raw:
        return [str(s) for s in raw]
    return ["understat", "statsbomb", "fbref"]


def historical_understat_seasons() -> list[str]:
    cfg = load_leagues_config()
    if cfg.get("understat_seasons"):
        return [str(s) for s in cfg["understat_seasons"]]
    seasons = cfg.get("historical_seasons", [])
    mapped = []
    for s in seasons:
        if len(str(s)) == 4 and str(s).isdigit():
            mapped.append(str(2000 + int(str(s)[:2])))  # 2425 -> 2024
    current = cfg.get("season", "2526")
    if len(str(current)) == 4:
        mapped.append(str(2000 + int(str(current)[:2])))  # 2526 -> 2025
    return sorted(set(mapped)) or ["2024", "2025"]


def edge_margin_min() -> float:
    cfg = load_leagues_config()
    model_cfg = cfg.get("model") or {}
    if model_cfg.get("edge_margin_min") is not None:
        return float(model_cfg["edge_margin_min"])
    return float(get_env("EDGE_MARGIN_MIN", "0.05"))


def set_edge_margin_min(margin: float) -> None:
    import re

    config_path = ROOT / "config" / "leagues.yaml"
    text = config_path.read_text(encoding="utf-8")
    value = round(float(margin), 3)
    if re.search(r"^\s*edge_margin_min:\s*[\d.]+", text, re.MULTILINE):
        text = re.sub(
            r"^(\s*edge_margin_min:\s*)[\d.]+",
            rf"\g<1>{value}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = text.replace(
            "bookie_blend_weight:",
            f"edge_margin_min: {value}\n  bookie_blend_weight:",
            1,
        )
    config_path.write_text(text, encoding="utf-8")


def dixon_coles_blend_weight() -> float:
    cfg = load_leagues_config()
    model_cfg = cfg.get("model") or {}
    if model_cfg.get("dixon_coles_blend_weight") is not None:
        return float(model_cfg["dixon_coles_blend_weight"])
    return float(get_env("DIXON_COLES_BLEND_WEIGHT", "0.05"))


def set_dixon_coles_blend_weight(weight: float) -> None:
    """Persist tuned Dixon-Coles blend weight to config/leagues.yaml."""
    import re

    config_path = ROOT / "config" / "leagues.yaml"
    text = config_path.read_text(encoding="utf-8")
    value = round(float(weight), 3)
    if re.search(r"^\s*dixon_coles_blend_weight:\s*[\d.]+", text, re.MULTILINE):
        text = re.sub(
            r"^(\s*dixon_coles_blend_weight:\s*)[\d.]+",
            rf"\g<1>{value}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        anchor = "model:\n"
        if anchor not in text:
            text = text.rstrip() + f"\n\nmodel:\n  dixon_coles_blend_weight: {value}\n"
        else:
            text = text.replace(
                anchor,
                f"{anchor}  dixon_coles_blend_weight: {value}\n",
                1,
            )
    config_path.write_text(text, encoding="utf-8")


def kelly_fraction() -> float:
    cfg = load_leagues_config()
    model_cfg = cfg.get("model") or {}
    if model_cfg.get("kelly_fraction") is not None:
        return float(model_cfg["kelly_fraction"])
    return float(get_env("KELLY_FRACTION", "0.25"))


def per_league_edge_margins() -> dict[str, float]:
    cfg = load_leagues_config()
    model_cfg = cfg.get("model") or {}
    raw = model_cfg.get("per_league_edge_margin") or {}
    return {str(k): float(v) for k, v in raw.items()}


def edge_margin_for_div(div: str | None) -> float:
    margins = per_league_edge_margins()
    if div and div in margins:
        return margins[div]
    return edge_margin_min()


def set_per_league_edge_margins(margins: dict[str, float]) -> None:
    import re

    config_path = ROOT / "config" / "leagues.yaml"
    text = config_path.read_text(encoding="utf-8")
    lines = ["  per_league_edge_margin:"]
    for code, m in sorted(margins.items()):
        lines.append(f"    {code}: {round(float(m), 3)}")
    block = "\n".join(lines) + "\n"

    if re.search(r"^\s*per_league_edge_margin:\s*$", text, re.MULTILINE):
        text = re.sub(
            r"^\s*per_league_edge_margin:\s*\n(?:\s+\w+:\s*[\d.]+\n)*",
            block,
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = text.replace(
            "  edge_margin_min:",
            f"{block}  edge_margin_min:",
            1,
        )
    config_path.write_text(text, encoding="utf-8")


def per_league_blend_weights() -> dict[str, float]:
    cfg = load_leagues_config()
    model_cfg = cfg.get("model") or {}
    raw = model_cfg.get("per_league_blend") or {}
    return {str(k): float(v) for k, v in raw.items()}


def set_per_league_blend_weights(weights: dict[str, float]) -> None:
    """Write per_league_blend block under model: in leagues.yaml."""
    import re

    config_path = ROOT / "config" / "leagues.yaml"
    text = config_path.read_text(encoding="utf-8")
    lines = ["  per_league_blend:"]
    for code, w in sorted(weights.items()):
        lines.append(f"    {code}: {round(float(w), 3)}")
    block = "\n".join(lines) + "\n"

    if re.search(r"^\s*per_league_blend:\s*$", text, re.MULTILINE):
        text = re.sub(
            r"^\s*per_league_blend:\s*\n(?:\s+\w+:\s*[\d.]+\n)*",
            block,
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = text.replace(
            "  edge_margin_min:",
            f"{block}  edge_margin_min:",
            1,
        )
    config_path.write_text(text, encoding="utf-8")


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
    seasons = all_seasons()
    return (
        f"Active leagues ({len(active)}): {', '.join(active)}\n"
        f"Seasons: {', '.join(seasons)}\n"
        f"Inactive (ready to enable): {len(inactive)}"
    )