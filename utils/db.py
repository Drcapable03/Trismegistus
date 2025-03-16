import pandas as pd
from sqlalchemy import create_engine

# SQLite DB (adjust path if different)
engine = create_engine("sqlite:///data/trismegistus.db")

def load_csv_to_db(csv_path, table_name, if_exists="replace"):
    """
    Load CSV into SQL table.
    if_exists: 'replace' (default) overwrites, 'append' adds to existing table.
    """
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Loaded {csv_path} into {table_name} with if_exists='{if_exists}'")