from sqlalchemy import create_engine
import pandas as pd

# Connection string
engine = create_engine("postgresql+psycopg2://postgres:newpassword@localhost/Fooball_predictor")

def load_csv_to_db(csv_path, table_name):
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Data loaded to {table_name}!")

if __name__ == "__main__":
    load_csv_to_db("C:/Users/ASUS/Trismegistus/data/raw_matches.csv", "matches")