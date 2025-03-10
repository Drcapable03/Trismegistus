from sqlalchemy import create_engine

# Connection string
engine = create_engine("postgresql+psycopg2://postgres:<newpassword>@localhost/football_predictor")
print("Database connected!")