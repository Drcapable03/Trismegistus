"""Unified loaders for secondary xG enrichment caches."""

import pandas as pd

from config.settings import xg_source_priority
from utils import fbref_cache, statsbomb_cache, xg_cache


def load_xg_caches_by_priority() -> list[tuple[str, pd.DataFrame]]:
    loaders = {
        "understat": xg_cache.load_xg_matches,
        "statsbomb": statsbomb_cache.load_statsbomb_xg,
        "fbref": fbref_cache.load_fbref_xg,
    }
    caches: list[tuple[str, pd.DataFrame]] = []
    for source in xg_source_priority():
        loader = loaders.get(source)
        if loader is None:
            continue
        df = loader()
        if not df.empty:
            caches.append((source, df))
    return caches