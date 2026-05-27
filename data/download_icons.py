import sys
import requests
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ICON_FOLDER

def download_champion_icons():
    """
    Queries Riot Games' Data Dragon API to discover the latest patch runtime,
    fetches the full global champion roster metadata, and downloads the 
    official base square icons as PNG assets into the local icon directory.
    """
    # Query Data Dragon for latest live patch release version
    versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
    latest = versions[0]
    
    # Extract structural list of all active champions
    url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/champion.json"
    champions = requests.get(url).json()["data"]
    
    os.makedirs(ICON_FOLDER, exist_ok=True)
    
    # Download and serialize each square image web asset to disk
    for name in champions:
        icon_url = f"https://ddragon.leagueoflegends.com/cdn/{latest}/img/champion/{name}.png"
        img_data = requests.get(icon_url).content

        with open(os.path.join(ICON_FOLDER, f"{name}.png"), "wb") as f:
            f.write(img_data)
        print(f"Downloaded: {name}")

download_champion_icons()
