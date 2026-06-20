# API & Service Costs — Trismegistus Guide

Last updated: 2026-06-20

**Philosophy:** Free/open data first. Pay only when free paths are exhausted.

## Free forever (no subscription needed)

| Service | What you get | Trismegistus usage |
|---------|--------------|-------------------|
| [football-data.co.uk](https://www.football-data.co.uk) | Historical results, odds, fixtures CSV | Primary ingest (`config/leagues.yaml`) |
| [Open-Meteo](https://open-meteo.com) | Historical weather | `scripts/fetch_chaos.py` |
| [Nominatim/OSM](https://nominatim.org) | Geocoding | Team → city lookup |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Event-level xG (select comps) | Phase 2 enrichment |
| B365 odds in CSV | Historical bookmaker lines | Already in match data — **no API needed for backtest** |

**Cheapest path for odds:** Use `B365H/D/A` from football-data.co.uk for historical analysis.
For **live** future odds, see freemium options below or Phase 2 Scrapling scrape.

## Freemium APIs (optional)

### The Odds API — [the-odds-api.com](https://the-odds-api.com)
| Tier | Cost | Credits/month | Fit |
|------|------|---------------|-----|
| Free | $0 | 500 | ~500 live odds requests; fine for occasional `--predict` runs |
| 20K | ~$30/mo | 20,000 | Regular weekly predictions across multiple leagues |

**Free alternative:** Historical odds from CSV; for live odds use `OddsHarvester` / Scrapling scrape (Phase 2, $0).

### NewsAPI — [newsapi.org](https://newsapi.org/pricing)
| Tier | Cost | Limit | Fit |
|------|------|-------|-----|
| Developer | $0 | 100 req/day | Very limited; OK for testing |
| Business | $449/mo | Unlimited | **Not recommended** for this project |

**Free alternative:** Skip news in Phase 1 (defaults to 0.1 sentiment). Phase 2: RSS/Google scrape via Scrapling, club Twitter feeds, or `googlesearch-python`.

### API-Football / RapidAPI
| Tier | Cost | Notes |
|------|------|-------|
| Free | $0 | ~100 req/day on RapidAPI — fixtures/standings |
| Pro | ~$10–30/mo | More requests |

**Free alternative:** `soccerdata` / `soccerapi` scrapers for FBref, Understat, ESPN (no key).

## What you likely need to spend

| Use case | Recommended | Monthly cost |
|----------|-------------|--------------|
| Learning + backtesting | **Nothing** — CSV + Open-Meteo | **$0** |
| Weekly predictions (few leagues) | The Odds API free tier | **$0** |
| Daily predictions + news sentiment | The Odds API 20K + skip NewsAPI | **~$30** |
| Full news intelligence | Scrapling self-host (time, not money) | **$0** (+ dev effort) |

## Recommended subscription order (if any)

1. **Start at $0** — current pipeline works without keys
2. **The Odds API free** — only if you need live odds for `--predict`
3. **Never NewsAPI paid** — use scraping in Phase 2 instead
4. **penaltyblog / soccerdata** — PyPI packages, free

## Environment variables

```env
# All optional
ODDS_API_KEY=      # The Odds API — live odds only
NEWS_API_KEY=      # NewsAPI — skip unless you accept free tier limits
WEATHER_API_KEY=   # Unused; Open-Meteo needs no key
```