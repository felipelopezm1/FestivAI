import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import requests 

# --- Configurable Paths ---
CSV_PATH = Path("source_files/user_files/user_data.csv")
ARTIST_TIME_PATH = Path("source_files/artist_files/artist_files/artist_dataset.csv")
DATE_FOLDER = datetime.now().strftime("%Y-%m-%d")
INPUT_BASE = Path("source_files/artist_files/images/slogan_outputs") / DATE_FOLDER
USER_OUTPUT_BASE = Path("source_files/user_files/output_by_user_final") / DATE_FOLDER
USER_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(CSV_PATH)
artist_df = pd.read_csv(ARTIST_TIME_PATH)


def ollama_score(genres, fav_artists, candidate_artist):
    prompt = f"User likes genres: {genres}\nFavorite artists: {fav_artists}\nHow well does '{candidate_artist}' match their taste?\nReply only with a number from 1 to 10."
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        if response.ok:
            digits = ''.join(filter(str.isdigit, response.json()['response']))
            return int(digits) if digits else 0
    except Exception as e:
        print(f"Ollama error scoring {candidate_artist}: {e}")
    return 0

all_generated = {}
for label_folder in INPUT_BASE.glob("*"):
    if not label_folder.is_dir():
        continue
    for path in label_folder.glob("*.png"):
        artist_name = path.stem.split("_", 1)[-1].replace("_", " ").title()
        all_generated[artist_name] = all_generated.get(artist_name, []) + [path]

# --- Match and copy top images per user ---
print("\n🔄 Distributing top artist images to user folders...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    uname = row['name'].strip().replace(" ", "_")
    genres = row['genres']
    fav_artists = row['fav_artists']
    user_folder = USER_OUTPUT_BASE / uname
    user_folder.mkdir(parents=True, exist_ok=True)

    # ✅ Collect already assigned artist names in this folder
    existing_artists = set()
    for img_path in user_folder.glob("*.png"):
        parts = img_path.stem.split("_", 2)
        if len(parts) == 3:
            existing_artist = parts[2].replace("_", " ").title()
            existing_artists.add(existing_artist)

    # Score and select top matches
    artist_scores = []
    for artist, paths in all_generated.items():
        score = ollama_score(genres, fav_artists, artist)
        for path in paths:
            artist_scores.append((score, artist, path))

    top_matches = sorted(artist_scores, reverse=True)[:6]

    # Copy only if artist not already in user folder
    for rank, (score, artist, path) in enumerate(top_matches, start=1):
        if artist in existing_artists:
            print(f"⏩ {uname} already has: {artist}")
            continue

        filename = f"{rank}_{path.name}"
        destination = user_folder / filename
        shutil.copy2(path, destination)
        print(f"📁 {uname} → #{rank}: {artist} (score {score})")
        existing_artists.add(artist)  # Update to prevent future repeats in same run

print("✅ Distribution complete.")