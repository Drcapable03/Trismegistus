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
    }
    joblib.dump(payload, path)
    print(f"Saved model to {path}")
    return path


def load_game_forger(forger, path: str | Path) -> None:
    payload = joblib.load(path)
    forger.outcome_model = payload["outcome_model"]
    forger.goals_model = payload["goals_model"]
    forger.outcome_features = payload["outcome_features"]
    forger.goals_features = payload["goals_features"]
    forger.sim_runs = payload.get("sim_runs", forger.sim_runs)
    print(f"Loaded model from {path}")


def latest_model_path(tag: str = "game_forger") -> Path | None:
    candidates = sorted(MODELS_DIR.glob(f"{tag}_*.joblib"), reverse=True)
    return candidates[0] if candidates else None