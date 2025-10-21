import os
import re
import cv2
import requests
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from Ollama_Slogan import generate_slogan

# Paths and constants
CSV_PATH = Path("source_files/user_files/user_data.csv")
DATE_FOLDER = datetime.now().strftime("%Y-%m-%d")
OUTPUT_BASE = Path("source_files/artist_files/images/slogan_outputs") / DATE_FOLDER
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
USER_OUTPUT_BASE = Path("source_files/user_files/output_by_user_final") / DATE_FOLDER
USER_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

INPUT_DIRS = {
    'medium_artists': (Path("source_files/artist_files/images/medium_artists"), 'ma_', 3),
    'more_artists': (Path("source_files/artist_files/images/more_artists"), 'moa_', 2),
    'headliners_major_acts': (Path("source_files/artist_files/images/headliners_major_acts"), 'hma_', 1),
    'final_section': (Path("source_files/artist_files/images/final_section"), 'fs_', 1),
}

BACKGROUND_PATH = Path("source_files/artist_files/images/background/bg_genre_1.png")
LOGO_PATH = Path("source_files/artist_files/images/logo/logo.png")
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 4
COLOR = (0, 0, 0)
THICKNESS = 7
LINE_TYPE = cv2.LINE_AA

# Utilities
def remove_emojis(text):
    return re.compile("[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF" "\U0001F1E0-\U0001F1FF" \
        "\U00002500-\U00002BEF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "\U0001f926-\U0001f937" \
        "\U00010000-\U0010ffff" "\u200d" "\u2640-\u2642" "\u2600-\u2B55" "\u23cf" "\u23e9" "\u231a" "\ufe0f" "\u3030" "]+", re.UNICODE).sub(r'', text)

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
    words, lines, current_line = text.split(), [], []
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
    start_y = img.shape[0] - bottom_padding - line_height * len(lines) + line_height
    for i, line in enumerate(lines):
        text_y = start_y + i * line_height
        cv2.putText(img, line, (left_padding + 2, text_y + 2), FONT, FONT_SCALE, (255, 255, 255), THICKNESS + 2, LINE_TYPE)
        cv2.putText(img, line, (left_padding, text_y), FONT, FONT_SCALE, COLOR, THICKNESS, LINE_TYPE)
    return img

def ollama_score(genres, fav_artists, candidate_artist):
    prompt = f"User likes genres: {genres}\nFavorite artists: {fav_artists}\nHow well does '{candidate_artist}' match their taste?\nReply only with a number from 1 to 10."
    try:
        response = requests.post("http://localhost:11434/api/generate", json={"model": "mistral", "prompt": prompt, "stream": False})
        if response.ok:
            digits = ''.join(filter(str.isdigit, response.json()['response']))
            return int(digits) if digits else 0
    except Exception as e:
        print("Ollama error:", e)
    return 0

# Load base assets
background = cv2.imread(str(BACKGROUND_PATH))
logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)
BG_HEIGHT, BG_WIDTH = background.shape[:2]

# Load user data
df = pd.read_csv(CSV_PATH)

# Process all inputs
all_generated = {}
for label, (input_dir, prefix, max_count) in INPUT_DIRS.items():
    for file_path in input_dir.glob(f"{prefix}*.png"):
        artist_name = file_path.stem.replace(prefix, '').replace('_', ' ').title()
        try:
            slogan = remove_emojis(generate_slogan(artist_name).strip().strip('"\''))
        except Exception as e:
            print(f"Slogan failed for {artist_name}: {e}")
            continue
        artist_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if artist_img is None:
            continue
        scale = min((BG_WIDTH * 0.9) / artist_img.shape[1], (BG_HEIGHT * 0.6) / artist_img.shape[0])
        new_size = (int(artist_img.shape[1]*scale), int(artist_img.shape[0]*scale))
        artist_resized = cv2.resize(artist_img, new_size, interpolation=cv2.INTER_AREA)
        composed = background.copy()
        composed = overlay_png(composed, artist_resized, (BG_WIDTH - new_size[0]) // 2, int(BG_HEIGHT * 0.1))
        composed = draw_slogan_multiline_left_bottom(composed, slogan)
        composed = overlay_png(composed, logo, BG_WIDTH - logo.shape[1] - 30, 30)
        output_dir = OUTPUT_BASE / label
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / file_path.name
        cv2.imwrite(str(out_path), composed)
        all_generated[artist_name] = all_generated.get(artist_name, []) + [out_path]
        print(f"✅ Created: {out_path}")

# Match top N images for each user from all sets
for _, row in df.iterrows():
    uname = row['name'].strip().replace(" ", "_")
    user_folder = USER_OUTPUT_BASE / uname
    user_folder.mkdir(parents=True, exist_ok=True)
    artist_scores = []
    for artist, paths in all_generated.items():
        score = ollama_score(row['genres'], row['fav_artists'], artist)
        for path in paths:
            artist_scores.append((score, artist, path))
    top_matches = sorted(artist_scores, reverse=True)[:6]  # 3 + 2 + 1 = 6
    for rank, (score, artist, path) in enumerate(top_matches, start=1):
        filename = f"{rank}_{path.name}"
        shutil.copy2(path, user_folder / filename)
        print(f"📁 {uname} → #{rank}: {artist} (score {score})")
