import requests
import pandas as pd

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

# Load user and artist data
user_data = pd.read_csv("source_files/user_files/user_data.csv")
artist_data = pd.read_csv("source_files/artist_files/artist_files/artist_dataset.csv") 

# prompt tailored to each user
def generate_personalized_prompt(user_row):
    fav_artists = str(user_row['fav_artists']).split(', ')
    genres = str(user_row['genres']).split(', ')
    
    primary_artist = fav_artists[0].strip() if fav_artists else "a headliner"
    primary_genre = genres[0].strip().lower() if genres else "music"

    prompt = (
        f"Write a bold, punchy slogan for a music festival poster featuring the artist {primary_artist}. "
        f"The slogan must be no more than 5 words and must include the artist's name exactly as written, including any symbols. "
        f"It should reflect the vibe of {primary_genre} and use exciting, visual, and urgent language. "
        f"Do NOT use punctuation, emojis, asterisks, quotes, or any symbols outside of !"
        f"Do NOT write in full sentences or all caps. "
        f"The slogan will be placed in large text at the bottom left of a poster with a logo in the top right and an artist photo in the center. "
    )
    return prompt

# slogan from Ollama
def generate_slogan(prompt):
    response = requests.post(OLLAMA_API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    if response.ok:
        return response.json()["response"].strip()
    else:
        raise RuntimeError(f"Ollama error: {response.status_code} {response.text}")

# Func to process all users
def main():
    results = []

    for _, user_row in user_data.iterrows():
        user_name = user_row['name']
        prompt = generate_personalized_prompt(user_row)

        print(f"\nPrompt for {user_name}:\n{prompt}\n")

        try:
            slogan = generate_slogan(prompt)
            print(f"Slogan for {user_name}:\n{slogan}\n")
            results.append({"name": user_name, "slogan": slogan})
        except Exception as e:
            print(f"Error generating slogan for {user_name}: {e}")
            results.append({"name": user_name, "slogan": "ERROR"})

    
 

if __name__ == "__main__":
    main()