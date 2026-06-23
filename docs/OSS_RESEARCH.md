# Open-Source Research for Trismegistus

Last updated: 2026-06-20 (expanded)

Priority: **free/open first**, freemium APIs only where scraping is unreliable.

## Tier 1 — Adopt now (High fit, actively maintained)

### [soccerdata](https://github.com/probberechts/soccerdata) (Apache-2.0, PyPI) — **1.8k stars, v1.9.0 Apr 2026**
Unified scrapers with **built-in local cache** for:
FBref, Understat, WhoScored, Club Elo, ESPN, Football-Data.co.uk, Sofascore, SoFIFA.

**Trismegistus fit:**
- Phase 1b/2: Pull xG from Understat, team ratings from Club Elo
- Multi-source stats without maintaining individual scrapers
- Caching pattern we mirrored in `utils/chaos_cache.py`

```python
import soccerdata as sd
fbref = sd.FBref("ENG-Premier League", "2526")
schedule = fbref.read_schedule()
```

### [penaltyblog](https://github.com/martineastwood/penaltyblog) (MIT, PyPI) — **v1.11.0 Jun 2026**
Dixon-Coles/Poisson, implied odds, Elo/Massey/Colley/Pi ratings, Understat scraper, StatsBomb connector.

**Trismegistus fit:** Phase 1b — we implemented basic overround stripping in `evaluation/implied_odds.py`;
penaltyblog can replace this with Shin method later.

### [Scrapling](https://github.com/D4Vinci/Scrapling) (PyPI: `scrapling`)
Stealth scraping for injuries, OddsPortal backup, transfer news (Phase 2).

### [football-data.co.uk](https://www.football-data.co.uk) (free CSV)
Primary ingest — now supports multi-season + 10 league slots in `config/leagues.yaml`.

## Tier 2 — Historical stats & enrichment

### [worldfootballr](https://github.com/jaseziv/worldfootballr) (R, API wrappers)
FBref, Transfermarkr, Understat access. Reference for feature ideas; Python equivalent is `soccerdata`.

### [StatsBomb Open Data](https://github.com/statsbomb/open-data) (free)
Event-level xG. penaltyblog/soccerdata both connect.

### [openfootball/football.json](https://github.com/openfootball/football.json)
Free JSON leagues/cups worldwide. Good for fixture metadata, not odds.

### [OddsHarvester](https://github.com/jordantete/OddsHarvester)
Multi-bookmaker odds scraper — free alternative to The Odds API.

### [soccerapi](https://github.com/S1M0N38/soccerapi)
Wrappers for Soccerway, BetExplorer, WhoScored.

### [gingeleski/odds-portal-scraper](https://github.com/gingeleski/odds-portal-scraper)
Historical OddsPortal odds.

## Tier 3 — Reference only

| Project | Use |
|---------|-----|
| [eddwebster/football_analytics](https://github.com/eddwebster/football_analytics) | EDA notebooks |
| [ProphitBet-Soccer-Bets-Predictor](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor) | GUI patterns |

## Integration roadmap (updated)

| Phase | Tool | Status |
|-------|------|--------|
| 1 | football-data.co.uk | Done |
| 1b | chaos cache, implied odds, model persistence | Done |
| 1b | expandable leagues config | Done |
| 2 | Scrapling (OddsPortal, news, injuries) | **Done** |
| 2 | soccerdata (Understat xG, Club Elo) | Deferred — dep conflict with scrapling |
| 2 | penaltyblog (Shin implied odds) | **Done** (Shin de-vig) |
| 6B | OddsPortal Big 5 live odds (`--fetch-odds`) | **Done** |
| 6B | OddsHarvester | Not adopted — Scrapling OddsPortal used instead |
| 6F | StatsBomb open data (`statsbombpy` via penaltyblog dep) | **Done** |
| 6F | FBref xG (penaltyblog scraper) | **Done** (403-blocked live; cache ready) |
| 2 | soccerdata (unified scrapers) | Still deferred — cssselect vs scrapling |
| 3 | FastAPI platform | **Done** |

## Cost summary

See [API_COSTS.md](API_COSTS.md) — **$0/month is viable** for backtesting and learning.