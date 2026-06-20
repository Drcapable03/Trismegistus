# Open-Source Research for Trismegistus

Last updated: 2026-06-20

This document catalogs GitHub/PyPI tools we can adopt to reduce build-from-scratch work.
Priority: **free/open first**, freemium APIs only where scraping is unreliable.

## Tier 1 — Adopt in Phase 1–2 (High fit)

### [penaltyblog](https://github.com/martineastwood/penaltyblog) (MIT, PyPI)
**What:** Production football analytics library — Dixon-Coles/Poisson models, implied odds,
Elo/Massey/Colley/Pi ratings, Understat/Club Elo scrapers, StatsBomb connector.

**Trismegistus fit:**
- Replace hand-rolled Poisson + bookie margin math with battle-tested implementations
- Add implied-probability extraction from our B365 columns (better backtest baseline)
- Phase 2: Understat xG scraper for richer features

**Integration:** `pip install penaltyblog` — use alongside existing `GameForger`, not as full replacement yet.

### [Scrapling](https://github.com/D4Vinci/Scrapling) (PyPI: `scrapling`)
**What:** Adaptive Python scraping framework with `StealthyFetcher` (anti-bot bypass),
adaptive CSS selectors, spider framework, pause/resume crawls.

**Trismegistus fit (Phase 2):**
- Scrape injury lists from club sites / Premier League when APIs fail
- Scrape OddsPortal or similar for odds backup when The Odds API quota is exhausted
- Scrape transfer news pages for Phase 2 gossip/intelligence

**Integration pattern:**
```python
from scrapling.fetchers import StealthyFetcher
page = StealthyFetcher.fetch(url, headless=True, network_idle=True)
rows = page.css('.injury-row', adaptive=True)
```

**Note:** Weather stays on Open-Meteo (free API, no scraping needed). Scrapling is for
HTML-heavy targets that block plain `requests`.

### [football-data.co.uk](https://www.football-data.co.uk) (free CSV)
Already integrated. Expand to multi-league via season codes in `config/leagues.yaml`.

## Tier 2 — Phase 2 Enrichment

### [StatsBomb Open Data](https://github.com/statsbomb/open-data) (free)
Event-level xG for select competitions. penaltyblog has a connector. Use for feature depth,
not as primary league coverage.

### [OddsHarvester](https://github.com/jordantete/OddsHarvester)
Scrapes odds from multiple bookmakers. Good Scrapling alternative reference; evaluate
license and maintenance before adopting.

### [soccerapi](https://github.com/S1M0N38/soccerapi)
API wrappers for soccerway, betexplorer, whoscored. Useful if we need fixture metadata
beyond football-data.co.uk.

### [gingeleski/odds-portal-scraper](https://github.com/gingeleski/odds-portal-scraper)
OddsPortal historical odds. Pair with Scrapling if OddsPortal changes layout.

## Tier 3 — Reference / Patterns Only

| Project | Use |
|---------|-----|
| [eddwebster/football_analytics](https://github.com/eddwebster/football_analytics) | EDA and modelling notebooks |
| [ProphitBet-Soccer-Bets-Predictor](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor) | GUI betting predictor patterns |
| [kochlisGit/...](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor) | Feature engineering ideas |

## Recommended Integration Roadmap

| Phase | OSS Tool | Task |
|-------|----------|------|
| 1 | penaltyblog (implied odds) | Better bookie baseline in `evaluation/backtest` |
| 1 | — | Fix pipeline with existing stack first |
| 2 | Scrapling | Injury + news + odds fallback scrapers |
| 2 | penaltyblog (scrapers) | Understat xG, Club Elo ratings |
| 2 | StatsBomb open | xG features for supported leagues |
| 3 | FastAPI (standard) | Web platform |

## What NOT to adopt (yet)

- LLM agent frameworks for predictions — noisy and costly for H/D/A
- Full replacement of GameForger with penaltyblog models — validate via backtest first
- PostgreSQL — SQLite sufficient until Phase 3 multi-user