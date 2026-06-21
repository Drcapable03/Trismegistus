from datetime import datetime
from pathlib import Path

import joblib

from config.settings import MODELS_DIR


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_game_forger(forger, tag: str = "game_forger") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{tag}_{_timestamp()}.joblib"
    payload = {
        "outcome_model": forger.outcome_model,
        "goals_model": forger.goals_model,
        "outcome_features": forger.outcome_features,
        "goals_features": forger.goals_features,
        "sim_runs": forger.sim_runs,
        "bookie_blend_weight": forger.bookie_blend_weight,
        "per_league_blend": getattr(forger, "per_league_blend", {}),
        "per_league_edge_margin": getattr(forger, "per_league_edge_margin", {}),
        "dixon_coles_blend_weight": getattr(forger, "dc_blend_weight", 0.0),
        "edge_margin": forger.edge_margin,
        "calibrator": getattr(forger, "calibrator", None),
        "training_metadata": getattr(forger, "training_metadata", {}),
    }
    joblib.dump(payload, path)
    print(f"Saved model to {path}")
    return path


def latest_model_path(tag: str = "game_forger") -> Path | None:
    candidates = sorted(MODELS_DIR.glob(f"{tag}_*.joblib"), reverse=True)
    return candidates[0] if candidates else None


def load_game_forger(forger, path: str | Path) -> None:
    payload = joblib.load(path)
    forger.outcome_model = payload["outcome_model"]
    forger.goals_model = payload["goals_model"]
    forger.outcome_features = payload["outcome_features"]
    forger.goals_features = payload["goals_features"]
    forger.sim_runs = payload.get("sim_runs", forger.sim_runs)
    if "bookie_blend_weight" in payload:
        forger.bookie_blend_weight = float(payload["bookie_blend_weight"])
    if "per_league_blend" in payload:
        forger.per_league_blend = dict(payload["per_league_blend"])
    if "per_league_edge_margin" in payload:
        forger.per_league_edge_margin = dict(payload["per_league_edge_margin"])
    if "dixon_coles_blend_weight" in payload:
        forger.dc_blend_weight = float(payload["dixon_coles_blend_weight"])
    if "edge_margin" in payload:
        forger.edge_margin = float(payload["edge_margin"])
    if "calibrator" in payload and payload["calibrator"] is not None:
        forger.calibrator = payload["calibrator"]
    if "training_metadata" in payload:
        forger.training_metadata = payload["training_metadata"]
    print(f"Loaded model from {path}")