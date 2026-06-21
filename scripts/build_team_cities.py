"""Merge Big 5 team -> city mappings into config/team_cities.yaml."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CITIES_PATH = ROOT / "config" / "team_cities.yaml"

# football-data.co.uk team names -> home city for weather geocoding
BIG5_CITIES: dict[str, str] = {
    # EPL
    "Arsenal": "London", "Aston Villa": "Birmingham", "Bournemouth": "Bournemouth",
    "Brentford": "London", "Brighton": "Brighton", "Burnley": "Burnley",
    "Chelsea": "London", "Crystal Palace": "London", "Everton": "Liverpool",
    "Fulham": "London", "Ipswich": "Ipswich", "Leeds": "Leeds",
    "Leicester": "Leicester", "Liverpool": "Liverpool", "Luton": "Luton",
    "Man City": "Manchester", "Man United": "Manchester", "Newcastle": "Newcastle",
    "Nott'm Forest": "Nottingham", "Southampton": "Southampton",
    "Sunderland": "Sunderland", "Tottenham": "London", "West Ham": "London",
    "Wolves": "Wolverhampton",
    # La Liga
    "Alaves": "Vitoria-Gasteiz", "Ath Bilbao": "Bilbao", "Ath Madrid": "Madrid",
    "Barcelona": "Barcelona", "Betis": "Seville", "Celta": "Vigo",
    "Elche": "Elche", "Espanol": "Barcelona", "Getafe": "Madrid",
    "Girona": "Girona", "Las Palmas": "Las Palmas", "Leganes": "Leganes",
    "Levante": "Valencia", "Mallorca": "Palma", "Osasuna": "Pamplona",
    "Oviedo": "Oviedo", "Real Madrid": "Madrid", "Sevilla": "Seville",
    "Sociedad": "San Sebastian", "Valencia": "Valencia", "Valladolid": "Valladolid",
    "Vallecano": "Madrid", "Villarreal": "Villarreal",
    # Bundesliga
    "Augsburg": "Augsburg", "Bayern Munich": "Munich", "Bochum": "Bochum",
    "Dortmund": "Dortmund", "Ein Frankfurt": "Frankfurt", "Freiburg": "Freiburg",
    "Hamburg": "Hamburg", "Heidenheim": "Heidenheim", "Hoffenheim": "Sinsheim",
    "Holstein Kiel": "Kiel", "Leverkusen": "Leverkusen", "M'gladbach": "Monchengladbach",
    "Mainz": "Mainz", "RB Leipzig": "Leipzig", "St Pauli": "Hamburg",
    "Stuttgart": "Stuttgart", "Union Berlin": "Berlin", "Werder Bremen": "Bremen",
    "Wolfsburg": "Wolfsburg", "FC Koln": "Cologne",
    # Serie A
    "Atalanta": "Bergamo", "Bologna": "Bologna", "Cagliari": "Cagliari",
    "Como": "Como", "Cremonese": "Cremona", "Empoli": "Empoli",
    "Fiorentina": "Florence", "Genoa": "Genoa", "Inter": "Milan",
    "Juventus": "Turin", "Lazio": "Rome", "Lecce": "Lecce",
    "Milan": "Milan", "Monza": "Monza", "Napoli": "Naples",
    "Parma": "Parma", "Pisa": "Pisa", "Roma": "Rome", "Sassuolo": "Sassuolo",
    "Torino": "Turin", "Udinese": "Udine", "Venezia": "Venice", "Verona": "Verona",
    # Ligue 1
    "Angers": "Angers", "Auxerre": "Auxerre", "Brest": "Brest",
    "Le Havre": "Le Havre", "Lens": "Lens", "Lille": "Lille",
    "Lorient": "Lorient", "Lyon": "Lyon", "Marseille": "Marseille",
    "Metz": "Metz", "Monaco": "Monaco", "Montpellier": "Montpellier",
    "Nantes": "Nantes", "Nice": "Nice", "Paris FC": "Paris",
    "Paris SG": "Paris", "Reims": "Reims", "Rennes": "Rennes",
    "St Etienne": "Saint-Etienne", "Strasbourg": "Strasbourg", "Toulouse": "Toulouse",
}


def build_team_cities() -> dict[str, str]:
    existing: dict[str, str] = {}
    if CITIES_PATH.exists():
        with open(CITIES_PATH, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
    merged = {**existing, **BIG5_CITIES}
    return dict(sorted(merged.items()))


def write_team_cities() -> Path:
    merged = build_team_cities()
    header = (
        "# Home team -> city for weather geocoding (football-data.co.uk names)\n"
        "# Regenerate Big 5 entries: poetry run python scripts/build_team_cities.py\n"
    )
    CITIES_PATH.write_text(header + yaml.dump(merged, allow_unicode=True, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(merged)} team cities to {CITIES_PATH}")
    return CITIES_PATH


if __name__ == "__main__":
    write_team_cities()