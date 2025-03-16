import pandas as pd
from sqlalchemy import create_engine, text

# SQLite DB
engine = create_engine("sqlite:///data/trismegistus.db")

def reset_table(table_name):
    """Drop and recreate table with minimal schema."""
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Date TEXT,
                HomeTeam TEXT,
                AwayTeam TEXT
            )
        """))
        conn.commit()
    print(f"Reset table {table_name} with basic schema.")

def load_csv_to_db(csv_path, table_name, if_exists="replace"):
    """Load CSV into SQL table."""
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Loaded {csv_path} into {table_name} with if_exists='{if_exists}'")