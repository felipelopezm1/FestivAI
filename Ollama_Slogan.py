import requests
import random

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  

ARTISTS = [
    "Two Friends", "Mau P", "Martin Garix", "Levity", "Bladee", "flip_turn", "Orion Sun", "Glass Beams", "Twice", "Charlotte Lawrence", "MK Gee", "Kaicrewsade", "Bilimuri", "Remi Wolf", "Gigi Perez", "Knock2", "Murda Beatz", "Ravyn Lenae", "Vincent Lima", "ISOxo", "Ca7riel & Paco Amoroso", "Fujii Kaze", "Wyatt Flores", "Wunderhorse", "Tapeb", "Doechii", "Sunami", "Dua Saleh", "Wallows", "Clairo", "Salute", "The Dare", "Sabrina Carpenter", "A$AP Rocky", "Young Miko", "Sierra Ferrell", "RÜFÜS DU SOL", "BUNT.", "Olivia Rodrigo", "Ocean Alley", "T-Pain", "JPEGMAFIA", "Kenny Mason", "Marina", "Tyler, The Creator"

]

def generate_slogan(artist):
    prompt = (
        f"You are a slogan generator for a music festival poster.\n"
        f"Only return a slogan — nothing else. The slogan must:\n"
        f"- Be a bold, punchy phrase\n"
        f"- Be 3 to 5 words max\n"
        f"- Contain the artist name: '{artist}'\n"
        f"- Use exciting, visual language\n"
        f"- Avoid punctuation, emojis, and full sentences\n"
        f"- Use lowercase or title case (no ALL CAPS)\n"
        f"Examples:\n"
        f"Catch Bladee Live\n"
        f"Feel Marina’s Energy\n"
        f"Tyler Takes the Stage\n"
        f"Korn Brings the Heat\n"
        f"Now, return the slogan for: {artist}\n"
        f"(Only reply with the slogan. No intro, no quotes.)"
    )

    response = requests.post(OLLAMA_API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    if response.ok:
        raw = response.json()["response"].strip()
        # Cleanup step: remove unwanted characters or text
        slogan = raw.strip('"').strip("'").split("\n")[0].strip()
        return slogan
    else:
        raise RuntimeError(f"Ollama error: {response.status_code} {response.text}")


def main():
    artist = random.choice(ARTISTS)
    print(f"\nArtist: {artist}")
    try:
        slogan = generate_slogan(artist)
        print(f"Slogan: {slogan}")
    except Exception as e:
        print(f" Error generating slogan: {e}")

if __name__ == "__main__":
    main()
