import os
import re
import cv2
import requests
import shutil
import pandas as pd
from pathlib import Path
from Ollama_Slogan import generate_slogan

# File paths
CSV_PATH = Path("source_files/user_files/user_data.csv")
INPUT_DIR = Path("source_files/artist_files/images/medium_artists")
OUTPUT_DIR = Path("source_files/artist_files/images/slogan_outputs")
BACKGROUND_PATH = Path("source_files/artist_files/images/background/bg_genre_1.png")
LOGO_PATH = Path("source_files/artist_files/images/logo/logo.png")
USER_OUTPUT_BASE = Path("source_files/user_files/output_by_user_final")


# Create output dirs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USER_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Font config
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 4
COLOR = (0, 0, 0)
THICKNESS = 7
LINE_TYPE = cv2.LINE_AA
BOTTOM_PADDING = 50

# Utility functions
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF" "\U00002500-\U00002BEF" "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251" "\U0001f926-\U0001f937" "\U00010000-\U0010ffff"
        "\u200d" "\u2640-\u2642" "\u2600-\u2B55" "\u23cf" "\u23e9" "\u231a"
        "\ufe0f" "\u3030" "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def overlay_png(base_img, overlay_img, x, y):
    if overlay_img.shape[2] != 4:
        raise ValueError("Overlay image does not have alpha channel")
    h, w = overlay_img.shape[:2]
    alpha = overlay_img[:, :, 3] / 255.0
    for c in range(3):
        base_img[y:y+h, x:x+w, c] = (
            alpha * overlay_img[:, :, c] + (1 - alpha) * base_img[y:y+h, x:x+w, c]
        )
    return base_img

def draw_slogan_multiline_left_bottom(img, text, max_width_px=1000, bottom_padding=50, left_padding=60):
    words = text.split()
    lines, current_line = [], []
    for word in words:
        trial_line = " ".join(current_line + [word])
        size = cv2.getTextSize(trial_line, FONT, FONT_SCALE, THICKNESS)[0]
        if size[0] > max_width_px:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    lines.append(" ".join(current_line))
    line_height = int(cv2.getTextSize("Test", FONT, FONT_SCALE, THICKNESS)[0][1] * 1.5)
    total_text_height = line_height * len(lines)
    start_y = img.shape[0] - bottom_padding - total_text_height + line_height
    for i, line in enumerate(lines):
        text_y = start_y + i * line_height
        text_x = left_padding
        cv2.putText(img, line, (text_x + 2, text_y + 2), FONT, FONT_SCALE, (255, 255, 255), THICKNESS + 2, LINE_TYPE)
        cv2.putText(img, line, (text_x, text_y), FONT, FONT_SCALE, COLOR, THICKNESS, LINE_TYPE)
    return img

def extract_artist_name(filename):
    return filename.stem.replace("ma_", "").replace("_", " ").title()

def ollama_score(user_genres, user_artists, candidate_artist):
    prompt = (
        f"User likes genres: {user_genres}\n"
        f"Favorite artists: {user_artists}\n"
        f"How well does '{candidate_artist}' match their taste?\n"
        f"Reply only with a number from 1 to 10."
    )
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        if response.ok:
            response_text = response.json()['response']
            digits = ''.join(filter(str.isdigit, response_text))
            return int(digits) if digits else 0
    except Exception as e:
        print("Ollama error:", e)
    return 0

# Load background and logo
background = cv2.imread(str(BACKGROUND_PATH))
logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)
BG_HEIGHT, BG_WIDTH = background.shape[:2]

# Generate slogan images
image_map = {}
for file_path in INPUT_DIR.glob("ma_*.png"):
    artist_name = extract_artist_name(file_path)
    try:
        slogan = generate_slogan(artist_name).strip().strip('"').strip("'")
        slogan = remove_emojis(slogan)
    except Exception as e:
        print(f"Slogan fail for {artist_name}: {e}")
        continue
    artist_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if artist_img is None: continue
    scale = min((BG_WIDTH * 0.9) / artist_img.shape[1], (BG_HEIGHT * 0.6) / artist_img.shape[0])
    new_size = (int(artist_img.shape[1]*scale), int(artist_img.shape[0]*scale))
    artist_resized = cv2.resize(artist_img, new_size, interpolation=cv2.INTER_AREA)
    composed = background.copy()
    x_offset = (BG_WIDTH - new_size[0]) // 2
    y_offset = int(BG_HEIGHT * 0.1)
    composed = overlay_png(composed, artist_resized, x_offset, y_offset)
    composed = draw_slogan_multiline_left_bottom(composed, slogan)
    composed = overlay_png(composed, logo, BG_WIDTH - logo.shape[1] - 30, 30)
    out_path = OUTPUT_DIR / file_path.name
    cv2.imwrite(str(out_path), composed)
    image_map[artist_name] = out_path

# Match images to users
df = pd.read_csv(CSV_PATH)
for _, row in df.iterrows():
    uname = row["name"].strip().replace(" ", "_")
    user_folder = USER_OUTPUT_BASE / uname
    user_folder.mkdir(parents=True, exist_ok=True)
    # Collect scores for all artists
    artist_scores = []
    for artist, image_path in image_map.items():
        score = ollama_score(row['genres'], row['fav_artists'], artist)
        artist_scores.append((score, artist, image_path))

    # Sort by score descending and pick top 3
    top_matches = sorted(artist_scores, reverse=True)[:3]

    # Copy top 3 images to user folder
    for rank, (score, artist, image_path) in enumerate(top_matches, start=1):
        filename = f"{rank}_{image_path.name}"
        shutil.copy2(image_path, user_folder / filename)
        print(f"{uname} → #{rank}: {artist} (score {score})")

