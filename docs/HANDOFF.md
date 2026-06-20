# Trismegistus — Context Handoff Note

> Write a new entry at ~180k context tokens or before any major pivot.
> Keep entries short; link to files, not full code dumps.

---

## Entry 001 — 2026-06-20 (Revival kickoff)

### Session goal
Phase 1 revival: fix pipeline foundation, document OSS research, establish git commit discipline.

### Project state
- **Repo:** `C:\Users\USER\Trismegistus` → `github.com/Drcapable03/Trismegistus`
- **Dormant since:** 2025-03-16 (29 commits total)
- **Stack:** Python 3.11+, Poetry, SQLite, scikit-learn, pandas
- **Primary predictor:** `predictors/game_forger.py` (GradientBoosting + Poisson + Monte Carlo)

### User decisions
1. Phase order: **1 Predictions → 2 News/Intel → 3 Platform**
2. API budget: **free/open first**, freemium OK as fallback
3. OSS interest: **Scrapling** (D4Vinci) for stealth scraping in Phase 2
4. **Git commit after every logical change batch**

### Phase 1 work in progress
| Task | Status | Files |
|------|--------|-------|
| OSS research doc | Done | `docs/OSS_RESEARCH.md` |
| Dependency fix | Done | `pyproject.toml`, `.env.example` |
| Config layer | Done | `config/leagues.yaml`, `config/settings.py`, `config/team_cities.yaml` |
| Train/predict split | Done | `predictors/game_forger.py`, `main.py` |
| Non-destructive DB | Done | `utils/db.py` (dynamic CSV-driven schema) |
| Backtest wiring | Done | `scripts/backtest.py`, `main.py`, `BlunderSniffer` |
| Tests | Done | `tests/` — 7 passing |
| penaltyblog integration | Deferred Phase 1b | evaluation baseline |
| Scrapling integration | Deferred Phase 2 | `features/scrapers/` |
| Chaos caching | Pending | `scripts/fetch_chaos.py` |
| Model persistence (joblib) | Pending | `data/models/` |

### Known bugs fixed this session
- Hardcoded dates `2025-03-02` / `2025-03-15` → dynamic `datetime.now()`
- Training on unlabeled future fixtures → train on completed only
- `reset_table` every run → `ensure_schema` + incremental load
- Rigid DB schema missing 2025/26 CSV columns → dynamic schema from CSV headers
- Python 3.14 build failures → Poetry env on Python 3.12 (`poetry env use py -3.12`)

### Verified run (2026-06-20)
```
poetry run python main.py --reset --ingest --backtest --limit 30
# 1763 matches ingested (5 leagues + fixtures)
# Holdout accuracy: 83.3% vs bookie 50.0% (small sample, 30 matches)
```

### Key files to read on resume
1. `main.py` — CLI entry point
2. `predictors/game_forger.py` — ML core
3. `config/leagues.yaml` — league URLs
4. `docs/OSS_RESEARCH.md` — what to adopt next
5. `pyproject.toml` — dependencies

### Phase 1b completed (2026-06-20)
- Chaos SQLite cache (`utils/chaos_cache.py`) — `--refresh-cache` / `--no-cache`
- Model persistence (`predictors/registry.py`) — `--save-model` / `--load-model`
- Overround-stripped bookie baseline (`evaluation/implied_odds.py`)
- Expandable leagues config (9 inactive slots, multi-season `2425`+`2526`)
- `docs/API_COSTS.md` — subscription guidance ($0 viable)

### Phase 2 completed (2026-06-20)
- Scrapling integrated (`scrapers/`) — OddsPortal WC odds, Google News, injury search
- World Cup pipeline: `--worldcup-scrape`, `--worldcup-ingest`, `--worldcup-predict`
- Verified: 87 WC matches scraped, odds/news/injuries live, 4 predictions emitted
- `soccerdata` deferred (pytest/cssselect conflict with scrapling) — revisit as optional extra

### Next steps
1. Cache Scrapling odds DF to SQLite (avoid re-scrape per `fetch_odds` call)
2. `soccerdata` in separate optional poetry group OR pytest upgrade
3. Referee data scrape (replace mock)
4. Phase 3: FastAPI platform

### Open questions
- None blocking Phase 1

---

## Template for next entry

```
## Entry NNN — YYYY-MM-DD

### Completed since last handoff
-

### Current blockers
-

### Files changed
-

### Next 3 actions
1.
2.
3.
```