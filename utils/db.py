import pandas as pd
from sqlalchemy import create_engine, inspect, text

from config.settings import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DATA_DIR / 'trismegistus.db'}")


def table_exists(table_name: str) -> bool:
    return inspect(engine).has_table(table_name)


def _quote_ident(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34) + chr(34))}"'


def _add_missing_columns(table_name: str, df: pd.DataFrame) -> None:
    inspector = inspect(engine)
    existing = {col["name"] for col in inspector.get_columns(table_name)}
    with engine.connect() as conn:
        for col in df.columns:
            if col not in existing:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {_quote_ident(col)} TEXT"))
                existing.add(col)
        conn.commit()


def ensure_schema(table_name: str = "matches") -> None:
    """Ensure matches table exists; columns are added dynamically from CSV data."""
    if not table_exists(table_name):
        pd.DataFrame(columns=["Div", "Date", "HomeTeam", "AwayTeam"]).to_sql(
            table_name, engine, if_exists="replace", index=False,
        )
        print(f"Created empty table '{table_name}' (columns added on ingest).")
    else:
        print(f"Table '{table_name}' already exists.")


def reset_table(table_name: str) -> None:
    """Drop table — use only with --reset flag."""
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.commit()
    print(f"Reset table {table_name}.")


def load_csv_to_db(csv_path: str, table_name: str, if_exists: str = "append") -> int:
    """Load CSV into SQL table, extending schema as needed and deduplicating rows."""
    ensure_schema(table_name)
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"No rows in {csv_path}")
        return 0

    if table_exists(table_name):
        row_count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table_name}", engine)["n"][0]
        if row_count > 0:
            _add_missing_columns(table_name, df)
            existing = pd.read_sql(
                f"SELECT HomeTeam, AwayTeam, Date FROM {table_name}", engine,
            )
            if not existing.empty:
                df = df.merge(existing, on=["HomeTeam", "AwayTeam", "Date"], how="left", indicator=True)
                df = df[df["_merge"] == "left_only"].drop(columns="_merge")
        else:
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            print(f"Loaded {len(df)} rows from {csv_path} into {table_name}")
            return len(df)

    if df.empty:
        print(f"All rows from {csv_path} already in {table_name}")
        return 0

    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} new rows from {csv_path} into {table_name}")
    return len(df)


def read_matches() -> pd.DataFrame:
    ensure_schema("matches")
    if not table_exists("matches"):
        return pd.DataFrame()
    count = pd.read_sql("SELECT COUNT(*) AS n FROM matches", engine)["n"][0]
    if count == 0:
        return pd.DataFrame()
    return pd.read_sql("SELECT * FROM matches", engine)