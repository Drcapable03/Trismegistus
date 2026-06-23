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

## Entry 004 — 2026-06-22 (Phase 6C — Intel calibration)

### Completed
- Curated YouTube `channels_by_div` per Big 5 in `config/intel.yaml`
- League-aware YouTube discovery (`discover_youtube_video_ids`, `div_code` in intel agent)
- Reddit credential preflight (`verify_reddit_connection`, `--calibrate-intel`)
- `scripts/calibrate_intel.py` — static source checks + optional live probes
- `scripts/intel_roi.py` — chaos-cache coverage, sentiment summary, holdout ablation
- CLI: `--calibrate-intel`, `--calibrate-intel-live`, `--intel-roi`
- `GameForger.backtest_on_holdout(intel_override=...)` for ablation tests

### Verified run
```
poetry run python main.py --calibrate-intel
# 4/5 passed (Reddit creds optional)
poetry run pytest tests/ -q
# 88 passed, 1 skipped
```

### Next 3 actions
1. Phase 6D: Retune Dixon-Coles blend weight
2. Add Reddit creds to `.env` and re-run `--calibrate-intel`
3. Populate chaos intel cache via `--predict --refresh-cache` when fixtures land

---

## Entry 005 — 2026-06-22 (Phase 6D — Dixon-Coles blend tuning)

### Completed
- `scripts/tune_dc_blend.py` — grid-search `dixon_coles_blend_weight` (0.0–0.30) on walk-forward holdout; optimizes selective ROI
- `set_dixon_coles_blend_weight()` in `config/settings.py`
- CLI: `--tune-dc-blend`
- `tests/test_tune_dc_blend.py` — persist helper, DC compose blend, mocked ROI selection

### Tuned result
- Holdout sweep (700 test matches, selective edge picks): best ROI at **w=0.10** (−17.6% on 41 bets vs −20.7% at w=0.05)
- `config/leagues.yaml` updated: `dixon_coles_blend_weight: 0.1`

### Verified run
```
poetry run python main.py --ingest --limit 0
poetry run python main.py --tune-dc-blend --limit 0
# Saved dixon_coles_blend_weight=0.100
poetry run pytest tests/ -q
# 91 passed, 1 skipped
```

### Next 3 actions
1. Phase 6E: Expand historical seasons
2. Add Reddit creds to `.env` and re-run `--calibrate-intel`
3. Populate chaos intel cache via `--predict --refresh-cache` when fixtures land

---

## Entry 006 — 2026-06-22 (Phase 6E — Expand historical seasons)

### Completed
- Expanded `historical_seasons` in `config/leagues.yaml`: 2425, 2324, 2223, 2122 (+ current 2526)
- Season helpers in `config/settings.py`: `current_season()`, `historical_seasons()`, `all_seasons()`, `football_data_league_url()`
- `Season` column tagged on ingest (`utils/db.py`, `main.py`)
- `scripts/expand_history.py` — URL preflight (25/25 CSVs) + per-season DB coverage report
- CLI: `--expand-history`; `--explore` shows season breakdown
- `tests/test_expand_history.py` (5 tests)

### Verified run
```
poetry run python main.py --expand-history
# 25/25 URLs ok
poetry run python main.py --reset --ingest --limit 0
# 8,908 Big 5 completed matches ingested (5 seasons)
poetry run python main.py --backtest --limit 0
# walk-forward 7127 train / 1781 test; holdout 52.6%
poetry run pytest tests/ -q
# 96 passed, 1 skipped
```

### Next 3 actions
1. Phase 6F: Secondary enrichment (FBref/StatsBomb) if ROI still flat
2. Add Reddit creds to `.env` and re-run `--calibrate-intel`
3. Populate chaos intel cache via `--predict --refresh-cache` when fixtures land

---

## Entry 007 — 2026-06-22 (Phase 6F — Secondary xG enrichment)

### Completed
- StatsBomb open-data cache (`utils/statsbomb_cache.py`) — xG aggregated from shot events via `statsbombpy`
- FBref cache (`utils/fbref_cache.py`) + `scripts/fetch_fbref.py` via penaltyblog (graceful 403 fallback)
- Unified xG priority loader (`utils/xg_sources.py`): understat → statsbomb → fbref → shots proxy
- PIT features + `GameForger.train(enrichment_xg=...)` ablation flag
- `scripts/enrichment_roi.py` — per-source holdout coverage + enriched vs shots-only ROI
- CLI: `--fetch-fbref`, `--fetch-statsbomb`, `--fetch-statsbomb-limit`, `--enrichment-roi`
- `config/leagues.yaml` `enrichment.xg_source_priority`

### Notes
- FBref returns 403 in this environment (bot block) — cache path ready when reachable
- StatsBomb open data covers limited Big 5 seasons (La Liga depth best; EPL sparse)
- Full `--fetch-statsbomb` is slow (1 events API call per match); use `--fetch-statsbomb-limit` for smoke tests

### Verified run
```
poetry run python main.py --fetch-statsbomb --fetch-statsbomb-limit 3
# 77 matches cached across open Big 5 seasons
poetry run python main.py --enrichment-roi --limit 100
# coverage report + ablation harness
poetry run pytest tests/ -q
# 104 passed, 1 skipped
```

### Next 3 actions
1. Run full `--fetch-statsbomb` + `--fetch-xg` to maximize holdout xG overlap
2. Phase 3: FastAPI platform (post-Phase-6 roadmap complete)
3. Populate chaos intel cache via `--predict --refresh-cache` when fixtures land

---

## Entry 008 — 2026-06-23 (Operational cache prep)

### Completed
- Full Understat xG pull: **8,908 matches** cached (Big 5, seasons 2021–2025)
- `scripts/prep_caches.py` + CLI `--prep-caches` (orchestrates xG + StatsBomb + weather archive)
- Enrichment ROI after Understat fill: **83.2%** holdout xG overlap; selective ROI delta **+45.9%** vs shots-only proxy (limit=500 ablation)
- Started full `--fetch-statsbomb` and `--archive-chaos --limit 0` (long-running background jobs)

### Cache status (after Understat)
| Source | Rows |
|--------|------|
| Understat xG | 8,908 |
| StatsBomb xG | 77 → full fetch in progress |
| FBref xG | 0 (403-blocked) |
| Chaos weather | 111 → archive in progress |

### Intel note
- **0 upcoming Big 5 fixtures** in DB — live intel fills on `--predict --refresh-cache` when football-data publishes fixtures

### Commands
```bash
poetry run python main.py --prep-caches --limit 0          # all three steps
poetry run python main.py --fetch-xg                       # Understat only
poetry run python main.py --fetch-statsbomb                # StatsBomb full (slow)
poetry run python main.py --archive-chaos --limit 0        # weather for all completed
poetry run python main.py --enrichment-roi --limit 500
```

### Next 3 actions
1. Let `--fetch-statsbomb` and `--archive-chaos` finish locally
2. Re-run `--enrichment-roi --limit 500` after StatsBomb completes
3. Phase 3: FastAPI platform

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