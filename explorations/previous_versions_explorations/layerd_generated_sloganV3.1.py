import os
import re
import cv2
import requests
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import numpy as np
from Ollama_Slogan import generate_slogan
import random
# Paths and constants
CSV_PATH = Path("source_files/user_files/user_data.csv")
FESTIVAL_PATH = Path("source_files/festival_files/festival_data.csv")
ARTIST_TIME_PATH = Path("source_files/artist_files/artist_files/artist_dataset.csv")
ARTIFACTS_PATH = Path("source_files/artist_files/images/background/artifacts/lollapalooza")
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

BACKGROUND_PATH = Path("source_files/artist_files/images/background/bg_genre_2.png")
LOGO_PATH = Path("source_files/artist_files/images/logo/logo.png")
FONT_SLOGAN_PATH = Path("source_files/artist_files/images/fonts/Impact.ttf")
FONT_META_PATH = Path("source_files/artist_files/images/fonts/Montserrat-VariableFont_wght.ttf")

# Utilities
def remove_emojis(text):
    return re.compile(
        "["                                  # Opening character class
        "\U0001F600-\U0001F64F"              # Emoticons
        "\U0001F300-\U0001F5FF"              # Symbols & pictographs
        "\U0001F680-\U0001F6FF"              # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"              # Flags
        "\U00002500-\U00002BEF"              # Chinese characters
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u200d"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]",                                 
        re.UNICODE
    ).sub(r'', text)



def overlay_png(base_img, overlay_img, x, y):
    """
    Overlays a transparent PNG (with alpha channel) onto a base image.
    """
    if overlay_img.shape[2] != 4:
        raise ValueError("Overlay image does not have alpha channel")

    h, w = overlay_img.shape[:2]
    overlay_rgb = overlay_img[:, :, :3]
    overlay_alpha = overlay_img[:, :, 3] / 255.0

    # Make sure base image has 4 channels
    if base_img.shape[2] == 3:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2BGRA)

    result = base_img.copy()

    for c in range(3):
        result[y:y+h, x:x+w, c] = (
            overlay_alpha * overlay_rgb[:, :, c] +
            (1.0 - overlay_alpha) * result[y:y+h, x:x+w, c]
        )

    # Preserve alpha in result (set to 255 in target region if not already handled)
    result[y:y+h, x:x+w, 3] = np.clip(
        overlay_alpha * 255 + (1.0 - overlay_alpha) * result[y:y+h, x:x+w, 3],
        0, 255
    ).astype(np.uint8)

    return result


def get_genre_color(genre):
    prompt = f"Suggest a vibrant RGB color for the music genre: {genre}. Reply only in the format R,G,B."
    try:
        response = requests.post("http://localhost:11434/api/generate", json={"model": "mistral", "prompt": prompt, "stream": False})
        if response.ok:
            rgb_str = response.json()['response'].strip()
            match = re.match(r'(\d+),(\d+),(\d+)', rgb_str)
            if match:
                return tuple(map(int, match.groups()))
    except Exception as e:
        print("Genre color fallback due to error:", e)
    return (0, 0, 0)

def draw_text_with_outline(draw, position, text, font, fill, outline_fill, stroke_width):
    x, y = position
    draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=outline_fill)

def draw_slogan_multiline(img_pil, text, font, color, left_padding=60, bottom_padding=50, max_width_px=1000):
    img_pil = img_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_pil)
    
    # Load artifact tiles
    artifact_paths = list(ARTIFACTS_PATH.glob("*.png"))
    if not artifact_paths:
        raise FileNotFoundError("No artifact images found in the specified directory.")

    words = text.split()
    lines, current_line = [], []
    for word in words:
        test_line = " ".join(current_line + [word])
        if draw.textlength(test_line, font=font) > max_width_px:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    lines.append(" ".join(current_line))

    line_height = font.getbbox("A")[3] + 20
    total_height = line_height * len(lines)
    start_y = img_pil.height - bottom_padding - total_height

    rect_width = max(draw.textlength(line, font=font) for line in lines) + 40
    rect_height = total_height + 20
    rect_x = left_padding - 20
    rect_y = start_y - 10

    # Create artifact collage background
    artifact_bg = Image.new('RGBA', (int(rect_width), int(rect_height)), (0, 0, 0, 0))
    
    tile_size = 100  # You can tweak this to control density
    for y in range(0, artifact_bg.height, tile_size):
        for x in range(0, artifact_bg.width, tile_size):
            artifact_path = random.choice(artifact_paths)
            tile = Image.open(artifact_path).convert("RGBA")
            tile = ImageOps.fit(tile, (tile_size, tile_size), method=Image.Resampling.LANCZOS)
            artifact_bg.paste(tile, (x, y), tile)

    # Paste artifact collage into the final image
    img_pil.paste(artifact_bg, (int(rect_x), int(rect_y)), artifact_bg)

    # Draw the slogan text on top
    for i, line in enumerate(lines):
        draw_text_with_outline(draw, (left_padding, start_y + i * line_height), line, font, fill=color, outline_fill=(255, 255, 255), stroke_width=4)

    return img_pil



def draw_schedule_text(img_pil, artist, schedule_df, font):
    draw = ImageDraw.Draw(img_pil)
    artist_clean = artist.strip().lower()
    schedule_df['artist_normalized'] = schedule_df['artist'].astype(str).str.strip().str.lower()
    row = schedule_df[schedule_df['artist_normalized'] == artist_clean]

    date, time, stage = 'TBD', 'TBA', 'TBA'
    if not row.empty:
        date = row.iloc[0].get('date', 'TBD')
        time = row.iloc[0].get('time', 'TBA')
        stage = row.iloc[0].get('stage', 'TBA')

    lines = [f"{date} @ {time}", f"Stage: {stage}"]
    x = 40
    y_start = 60
    line_spacing = 45

    for i, line in enumerate(lines):
        y = y_start + i * line_spacing
        draw_text_with_outline(draw, (x, y), line, font, fill=(0, 0, 0), outline_fill=(255, 255, 255), stroke_width=2)
    return img_pil

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

def add_glow_effect(img, glow_color=(255, 255, 255), blur_radius=25, alpha=0.6):
    """
    Adds a glow behind an RGBA image without changing its opacity.
    """
    if img.shape[2] != 4:
        raise ValueError("Image must have an alpha channel")

    h, w = img.shape[:2]
    margin = blur_radius * 2
    new_h, new_w = h + margin, w + margin

    # Extract alpha and make glow mask
    alpha_channel = img[:, :, 3]
    glow_mask = cv2.GaussianBlur(alpha_channel, (0, 0), blur_radius)

    # Create the glow layer
    glow_layer = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    for i in range(3):  # RGB
        glow_layer[margin//2:margin//2 + h, margin//2:margin//2 + w, i] = (
            glow_mask * glow_color[i] / 255
        ).astype(np.uint8)
    glow_layer[margin//2:margin//2 + h, margin//2:margin//2 + w, 3] = (
        glow_mask * alpha
    ).astype(np.uint8)

    # Create the output canvas and combine layers
    output = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    output = overlay_png(output, glow_layer, 0, 0)
    output = overlay_png(output, img, margin//2, margin//2)

    return output



# Load fonts and data
font_slogan = ImageFont.truetype(str(FONT_SLOGAN_PATH), 80)
font_meta = ImageFont.truetype(str(FONT_META_PATH), 36)
background = cv2.imread(str(BACKGROUND_PATH))
logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)
BG_HEIGHT, BG_WIDTH = background.shape[:2]
df = pd.read_csv(CSV_PATH)
schedule_df = pd.read_csv(ARTIST_TIME_PATH)
artist_df = pd.read_csv(ARTIST_TIME_PATH)

# Process images
all_generated = {}
for label, (input_dir, prefix, max_count) in INPUT_DIRS.items():
    for file_path in input_dir.glob(f"{prefix}*.png"):
        artist_name = file_path.stem.replace(prefix, '').replace('_', ' ').title()
        genre_row = artist_df[artist_df['artist'].str.lower() == artist_name.lower()]
        genre = genre_row.iloc[0]['genres'] if not genre_row.empty else 'music'
        color = get_genre_color(genre)

        try:
            raw_slogan = generate_slogan(artist_name)
            slogan = remove_emojis(" ".join(raw_slogan.split()[:5])).strip()
            if not slogan or any(k in slogan.lower() for k in ["write", "slogan", "instructions"]):
                continue
        except Exception as e:
            print(f"Slogan failed for {artist_name}: {e}")
            continue

        artist_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if artist_img is None:
            continue

        scale = min((BG_WIDTH * 0.9) / artist_img.shape[1], (BG_HEIGHT * 0.6) / artist_img.shape[0])
        new_size = (int(artist_img.shape[1]*scale), int(artist_img.shape[0]*scale))
        artist_resized = cv2.resize(artist_img, new_size, interpolation=cv2.INTER_AREA)
        artist_resized = add_glow_effect(artist_resized, glow_color=(255, 255, 255), blur_radius=25, alpha=0.6)


        composed = background.copy()
        composed = overlay_png(composed, artist_resized, (BG_WIDTH - new_size[0]) // 2, int(BG_HEIGHT * 0.1))
        composed = overlay_png(composed, logo, BG_WIDTH - logo.shape[1] - 30, 30)

        composed_pil = Image.fromarray(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB))
        composed_pil = draw_slogan_multiline(composed_pil, slogan, font_slogan, color)
        composed_pil = draw_schedule_text(composed_pil, artist_name, schedule_df, font_meta)
        final_img = cv2.cvtColor(np.array(composed_pil), cv2.COLOR_RGB2BGR)

        output_dir = OUTPUT_BASE / label
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / file_path.name
        cv2.imwrite(str(out_path), final_img)
        all_generated[artist_name] = all_generated.get(artist_name, []) + [out_path]
        print(f"✅ Created: {out_path}")

print("\n🔍 Matching users to their top images...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    uname = row['name'].strip().replace(" ", "_")
    user_folder = USER_OUTPUT_BASE / uname
    user_folder.mkdir(parents=True, exist_ok=True)
    artist_scores = []
    for artist, paths in all_generated.items():
        score = ollama_score(row['genres'], row['fav_artists'], artist)
        for path in paths:
            artist_scores.append((score, artist, path))
    top_matches = sorted(artist_scores, reverse=True)[:6]
    for rank, (score, artist, path) in enumerate(top_matches, start=1):
        filename = f"{rank}_{path.name}"
        shutil.copy2(path, user_folder / filename)
        print(f"📁 {uname} → #{rank}: {artist} (score {score})")
