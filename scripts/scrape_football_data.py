import requests
import pandas as pd

def scrape_matches():
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"  # Premier League 2023-24
    response = requests.get(url)
    with open("C:/Users/ASUS/Trismegistus/data/raw_matches.csv", "wb") as f:
        f.write(response.content)
    print("Matches downloaded!")  

if __name__ == "__main__":
    scrape_matches()