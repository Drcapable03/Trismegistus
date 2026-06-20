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

### Next steps after Phase 1 core
1. Run `poetry install && poetry run pytest`
2. Run `poetry run python main.py --backtest`
3. Integrate `penaltyblog` implied-odds for backtest baseline
4. Phase 2: Scrapling wrapper for injuries/odds fallback

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