import requests
import pandas as pd
import re

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

user_data = pd.read_csv("source_files/user_files/user_data.csv")
artist_data = pd.read_csv("source_files/artist_files/artist_files/artist_dataset.csv")
artist_data.columns = artist_data.columns.str.strip().str.lower()

def clean_response(text):
    # Remove non-ASCII characters and non-English text
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.replace('"', '').replace("'", "").strip()
    return " ".join(text.split()[:5])

def generate_personalized_prompt(user_row):
    fav_artists = str(user_row.get('fav_artists', '')).split(', ')
    genres = str(user_row.get('genres', '')).split(', ')
    
    primary_artist = fav_artists[0].strip() if fav_artists else "a headliner"
    primary_genre = genres[0].strip().lower() if genres else "music"

    prompt = (
        f"Write a 3 to 5 word bold slogan for a music festival featuring {primary_artist}. "
        f"Include the artist's name exactly. Make it catchy, powerful, in English only. "
        f"Do NOT use punctuation, emojis, or quotes. Return ONLY the slogan, nothing else."
    )
    return prompt, primary_artist

def generate_slogan(prompt):
    response = requests.post(OLLAMA_API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 30}
    })

    if response.ok:
        raw = response.json()["response"].strip()
        return clean_response(raw)
    else:
        raise RuntimeError(f"Ollama error: {response.status_code} {response.text}")

def main():
    results = []

    for _, user_row in user_data.iterrows():
        user_name = user_row.get('name', 'Unnamed')
        try:
            prompt, artist = generate_personalized_prompt(user_row)
            print(f"\n🧠 Prompt for {user_name} ({artist}):\n{prompt}\n")
            slogan = generate_slogan(prompt)
            print(f"🔥 Slogan: {slogan}\n")
            results.append({"name": user_name, "artist": artist, "slogan": slogan})
        except Exception as e:
            print(f"❌ Error for {user_name}: {e}")
            results.append({"name": user_name, "slogan": "ERROR"})

if __name__ == "__main__":
    main()
