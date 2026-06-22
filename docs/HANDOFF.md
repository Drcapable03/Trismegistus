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

## Entry 002 — 2026-06-22 (Post-Phase-5 roadmap: Phase 6A)

### Completed since last handoff
- Phases 1–5 + live `--predict` wiring (commits through `063ede4`)
- **Phase 6A — Live predict validation** (`phase6a:` commit pending)
  - `scripts/validate_live_predict.py` — pre-flight checks + E2E smoke test
  - CLI: `--validate-live`, `--predict --dry-run` (ingested odds only, no live scrape)
  - `--explore` now reports fixture readiness guidance
  - Bug fix: `_feature_row_df()` aligns Series→DataFrame for GBC/Poisson predict (NaN crash)
  - `_is_completed()` handles missing `FTR` column (fixtures rows)

### Current blockers
- **No upcoming Big 5 fixtures** — football-data.co.uk `fixtures.csv` has no E0/SP1/D1/I1/F1 rows yet (season window). Re-ingest when 2626 fixtures appear.
- Club Elo API still flaky (503 → 1500 fallback)
- `soccerdata` still skipped (cssselect vs scrapling)

### Remaining roadmap (6 phases, no UI)
1. ~~Live predict validation~~ **Done (6A)**
2. Big 5 live odds scrape (OddsHarvester / Scrapling OddsPortal)
3. Intel calibration (Reddit creds, YouTube channels, optional ROI harness)
4. Retune Dixon-Coles blend weight
5. Expand historical seasons
6. Secondary enrichment (FBref / StatsBomb — only if ROI still flat)

### Verified run (2026-06-22)
```
poetry run python main.py --validate-live --limit 80
# 7/7 checks passed (E2E smoke: Liverpool vs Brentford synthetic fixture)
poetry run pytest tests/ -q
# 74 passed, 1 skipped
```

### Next 3 actions
1. Phase 6B: Big 5 live odds scrape
2. Re-run `--predict --limit 0` when new-season fixtures land
3. Update `docs/OSS_RESEARCH.md` integration table

---

## Entry 003 — 2026-06-22 (Phase 6B — Big 5 live odds)

### Completed
- OddsPortal Scrapling scrape for Big 5 (`config/leagues.yaml` oddsportal URLs)
- `scrape_big5_odds()` / `scrape_league_odds()` in `scrapers/oddsportal.py`
- Div-scoped SQLite cache (`utils/odds_cache.save_scrape(div_filter=...)`)
- Team alias matching for OddsPortal names (`config/team_aliases.yaml` oddsportal section)
- `agents/odds_agent.fetch_odds()` chain: DB → cached scrape → Odds API
- CLI: `--fetch-odds`, `--fetch-odds-league E0`
- `TRIS_AUTO_SCRAPE_ODDS` env (default false — run `--fetch-odds` before predict)

### Verified run
```
poetry run python main.py --fetch-odds --fetch-odds-league E0
# Scraped 10 EPL matches, cached (TTL 6h)
poetry run pytest tests/ -q
# 80 passed, 1 skipped
```

### Next 3 actions
1. Phase 6C: Intel calibration (Reddit creds, YouTube channels)
2. `--fetch-odds` for all Big 5 before match weeks
3. Re-run `--predict` when football-data fixtures land

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