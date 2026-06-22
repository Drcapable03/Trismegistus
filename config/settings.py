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
    seasons = [cfg["season"], *cfg.get("historical_seasons", [])]
    return (
        f"Active leagues ({len(active)}): {', '.join(active)}\n"
        f"Seasons: {', '.join(seasons)}\n"
        f"Inactive (ready to enable): {len(inactive)}"
    )