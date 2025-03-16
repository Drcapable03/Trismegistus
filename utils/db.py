import pandas as pd
from sqlalchemy import create_engine, text

# SQLite DB
engine = create_engine("sqlite:///data/trismegistus.db")

def reset_table(table_name):
    """Drop and recreate table with full schema for football-data.co.uk CSVs."""
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Div TEXT,
                Date TEXT,
                Time TEXT,
                HomeTeam TEXT,
                AwayTeam TEXT,
                FTHG REAL,  -- Full-Time Home Goals
                FTAG REAL,  -- Full-Time Away Goals
                FTR TEXT,   -- Full-Time Result
                HTHG REAL,  -- Half-Time Home Goals
                HTAG REAL,  -- Half-Time Away Goals
                HTR TEXT,   -- Half-Time Result
                Referee TEXT,
                HS REAL,    -- Home Shots
                AS REAL,    -- Away Shots
                HST REAL,   -- Home Shots on Target
                AST REAL,   -- Away Shots on Target
                HF REAL,    -- Home Fouls
                AF REAL,    -- Away Fouls
                HC REAL,    -- Home Corners
                AC REAL,    -- Away Corners
                HY REAL,    -- Home Yellows
                AY REAL,    -- Away Yellows
                HR REAL,    -- Home Reds
                AR REAL,    -- Away Reds
                B365H REAL, -- Bet365 Home Odds
                B365D REAL, -- Bet365 Draw Odds
                B365A REAL, -- Bet365 Away Odds
                BWH REAL,   -- Betway Home Odds
                BWD REAL,   -- Betway Draw Odds
                BWA REAL,   -- Betway Away Odds
                BFH REAL,   -- Betfair Home Odds
                BFD REAL,   -- Betfair Draw Odds
                BFA REAL,   -- Betfair Away Odds
                PSH REAL,   -- Pinnacle Home Odds
                PSD REAL,   -- Pinnacle Draw Odds
                PSA REAL,   -- Pinnacle Away Odds
                WHH REAL,   -- William Hill Home Odds
                WHD REAL,   -- William Hill Draw Odds
                WHA REAL,   -- William Hill Away Odds
                "1XBH" REAL, -- 1XBET Home Odds
                "1XBD" REAL, -- 1XBET Draw Odds
                "1XBA" REAL, -- 1XBET Away Odds
                MaxH REAL,  -- Max Home Odds
                MaxD REAL,  -- Max Draw Odds
                MaxA REAL,  -- Max Away Odds
                AvgH REAL,  -- Avg Home Odds
                AvgD REAL,  -- Avg Draw Odds
                AvgA REAL,  -- Avg Away Odds
                BFEH REAL,  -- Betfair Exchange Home
                BFED REAL,  -- Betfair Exchange Draw
                BFEA REAL,  -- Betfair Exchange Away
                "B365>2.5" REAL, -- Bet365 Over 2.5
                "B365<2.5" REAL, -- Bet365 Under 2.5
                "P>2.5" REAL,    -- Pinnacle Over 2.5
                "P<2.5" REAL,    -- Pinnacle Under 2.5
                "Max>2.5" REAL,  -- Max Over 2.5
                "Max<2.5" REAL,  -- Max Under 2.5
                "Avg>2.5" REAL,  -- Avg Over 2.5
                "Avg<2.5" REAL,  -- Avg Under 2.5
                "BFE>2.5" REAL,  -- Betfair Over 2.5
                "BFE<2.5" REAL,  -- Betfair Under 2.5
                AHh REAL,    -- Asian Handicap Home
                B365AHH REAL, -- Bet365 AH Home
                B365AHA REAL, -- Bet365 AH Away
                PAHH REAL,   -- Pinnacle AH Home
                PAHA REAL,   -- Pinnacle AH Away
                MaxAHH REAL, -- Max AH Home
                MaxAHA REAL, -- Max AH Away
                AvgAHH REAL, -- Avg AH Home
                AvgAHA REAL, -- Avg AH Away
                BFEAHH REAL, -- Betfair AH Home
                BFEAHA REAL  -- Betfair AH Away
            )
        """))
        conn.commit()
    print(f"Reset table {table_name} with full schema.")

def load_csv_to_db(csv_path, table_name, if_exists="replace"):
    """Load CSV into SQL table."""
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Loaded {csv_path} into {table_name} with if_exists='{if_exists}'")