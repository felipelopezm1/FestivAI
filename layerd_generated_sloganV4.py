
"""
Festiv-AI Personlized ads for Music Festivals 

This application generates personalized festival posters with slogans, design elements,
and artist information. Built on Lollapalooza 2025 data as a proof-of-concept MVP.
"""

# ================== IMPORTS ==================

# Import core libraries and modules 
import os
import re
import cv2
import requests
import json
import shutil
import random
import pandas as pd

# Import file system & utilities
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from io import BytesIO
from uuid import uuid4

# Image processing
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops
import numpy as np

# Genertive AI Custom modules
from Ollama_Slogan import generate_slogan
import qrcode




# ================== CONFIGURATION ==================

# Define paths to data sources
CSV_PATH = Path("source_files/user_files/user_data.csv")
FESTIVAL_PATH = Path("source_files/festival_files/festival_data.csv")
ARTIST_TIME_PATH = Path("source_files/artist_files/artist_files/artist_dataset.csv")
ARTIFACTS_PATH = Path("source_files/artist_files/images/background/artifacts/lollapalooza")

# Create output directories based on the date the application was used 
DATE_FOLDER = datetime.now().strftime("%Y-%m-%d")
OUTPUT_BASE = Path("source_files/artist_files/images/slogan_outputs") / DATE_FOLDER
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
USER_OUTPUT_BASE = Path("source_files/user_files/output_by_user_final") / DATE_FOLDER
USER_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Define input directories for artist images by category
INPUT_DIRS = {
    'medium_artists': (Path("source_files/artist_files/images/medium_artists"), 'ma_', 3),
    'more_artists': (Path("source_files/artist_files/images/more_artists"), 'moa_', 2),
    'headliners_major_acts': (Path("source_files/artist_files/images/headliners_major_acts"), 'hma_', 1),
    'final_section': (Path("source_files/artist_files/images/final_section"), 'fs_', 1),
}

# Define paths to design assits and artifacts
BACKGROUND_PATH = Path("source_files/artist_files/images/background/bg_genre_2.png")
ARTIFACTS_BG_PATH = Path("source_files/artist_files/images/background/artifacts_background/lollapalooza")
LOGO_PATH = Path("source_files/artist_files/images/logo/logo.png")
FONT_SLOGAN_PATH = Path("source_files/artist_files/images/fonts/Impact.ttf")
FONT_META_PATH = Path("source_files/artist_files/images/fonts/Montserrat-VariableFont_wght.ttf")


# JSON schedules for festival calendar
calendar_json_paths = [
    "explorations/lollapalooza_thursday_2025.json",
    "explorations/lollapalooza_friday_2025.json",
    "explorations/lollapalooza_saturday_2025.json",
    "explorations/lollapalooza_sunday_2025.json"
]




# ================== FESTIVAL EVENT DATA LOADING & NORMALIZATION ==================

# Master list of all events
festival_events = []

# Load and normalize all events from the JSON files
for path in calendar_json_paths:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for stage_name, acts in data.items():
            for act in acts:
                festival_events.append({
                    "artist": act["artist"].strip().lower(),
                    "time": act["time"].strip(),
                    "stage": stage_name.strip()
                })

 


# ================== UTILITIES ==================

# Utilities
def remove_emojis(text):
    """
    Removes emojis and miscellaneous Unicode symbols.
    """
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




# ==================  IMAGE, BACKGROUND AND TEXT GENERATION  ==================


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



def create_artifact_background_canvas(size, tile_paths, max_tiles=80, min_scale=0.2, max_scale=0.6):
    """
    Creates a collage of randomly placed and rotated artifact tiles.
    """
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    width, height = size

    for _ in range(max_tiles):
        tile_path = random.choice(tile_paths)
        try:
            tile = Image.open(tile_path).convert("RGBA")
        except Exception as e:
            print(f"Skipping tile {tile_path}: {e}")
            continue

        # Random scale
        scale = random.uniform(min_scale, max_scale)
        new_w = int(tile.width * scale)
        new_h = int(tile.height * scale)
        tile = tile.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Random rotation
        angle = random.randint(0, 360)
        tile = tile.rotate(angle, expand=True)

        # Random position
        x = random.randint(-new_w // 2, width - new_w // 2)
        y = random.randint(-new_h // 2, height - new_h // 2)

        # Paste using alpha
        canvas.alpha_composite(tile, (x, y))

    return canvas




def create_qr_code_image(data_url, box_size=6):
    """
    Creates a QR code image for a given URL and returns a PIL image.
    """
    qr = qrcode.QRCode(version=1, box_size=box_size, border=1)
    qr.add_data(data_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img.convert("RGBA")




def add_paper_cutout_effect(png_img, edge_thickness=15, blur_radius=3):
    """
    Adds a white torn paper effect around a transparent PNG image.
    """
    if png_img.shape[2] != 4:
        raise ValueError("Image must have alpha channel")

    pil_img = Image.fromarray(cv2.cvtColor(png_img, cv2.COLOR_BGRA2RGBA))
    alpha = pil_img.getchannel("A")

    # Create expanded white border by growing the alpha
    border = alpha.filter(ImageFilter.MaxFilter(edge_thickness))
    border = border.filter(ImageFilter.GaussianBlur(blur_radius))

    # Create white border layer
    border_layer = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    border_layer.putalpha(border)

    # Subtract original alpha to get only the border
    mask = ImageChops.subtract(border, alpha)
    border_layer.putalpha(mask)

    # Composite: white border first, then original image
    combined = Image.alpha_composite(border_layer, pil_img)

    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGBA2BGRA)



def get_genre_color(genre):
    """
    Dynamically request genre-based color suggestion from Ollama.
    """
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
    """
    Helper function for outlined text rendering.
    """
    x, y = position
    draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=outline_fill)



def draw_slogan_multiline(img_pil, text, font, color, left_padding=60, bottom_padding=50, max_width_px=1000):
    """
    Draw writes a multiline slogan text on the background image.
    """

    # Ensure image is in RGBA mode to support transparency
    img_pil = img_pil.convert("RGBA")
    draw = ImageDraw.Draw(img_pil)
    
    # Load artifacts for the slogans
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
    """
    writes the artist's scheduled performance details (date, time, stage) on the given poster image
    """

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
        draw_text_with_outline(draw, (x, y), line, font, fill=(0, 0, 0), outline_fill=(255, 255, 255), stroke_width=4)
    return img_pil




def generate_artist_calendar_ics(artist_name):
    """
    Generates an iCalendar (.ics) file content for the given artist's performances.
    """

    artist_normalized = artist_name.strip().lower()
    ics_content = """BEGIN:VCALENDAR
VERSION:2.0
CALSCALE:GREGORIAN
PRODID:-//LollaPoster//PML-QRGen//EN
"""

    for event in festival_events:
        if event["artist"] == artist_normalized:
            try:
                time_range = event["time"].split("-")
                start_time_raw = time_range[0].strip()
                end_time_raw = time_range[1].strip()

                def to_ical_time(t_str):
                    t_str = t_str.strip()
                    parts = t_str.split(":")
                    hour = int(parts[0])
                    minute = int(parts[1]) if len(parts) > 1 else 0
                    return f"{hour:02}{minute:02}00"

                start = to_ical_time(start_time_raw)
                end = to_ical_time(end_time_raw)
                date = "20250701"  # Replace with actual date if dynamic

                uid = str(uuid4())

                ics_content += f"""BEGIN:VEVENT
UID:{uid}
SUMMARY:{artist_name.title()} at {event["stage"]}
DTSTART:{date}T{start}Z
DTEND:{date}T{end}Z
LOCATION:{event["stage"]}
DESCRIPTION:Lollapalooza 2025 Performance
END:VEVENT
"""
            except Exception as e:
                print(f"⚠️ Time format issue for {artist_name}: {e}")
                continue

    ics_content += "END:VCALENDAR"
    return ics_content




def ollama_score(genres, fav_artists, candidate_artist):
    """
    Uses Ollama to calculate a compatibility score (1-10) between user preferences and artist.
    Uses genres and favorite artists to prompt the model.
    """

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





# ================== FINAL POSTER GENERATION PIPELINE ==================
# This loop processes all available artist images, applies background collage,
# slogan generation, paper cutout & glow effects, metadata stamping, QR code insertion,
# calendar export (ICS), and finally saves the poster.



# Load fonts and data
font_slogan = ImageFont.truetype(str(FONT_SLOGAN_PATH), 80)
font_meta = ImageFont.truetype(str(FONT_META_PATH), 48)
background = cv2.imread(str(BACKGROUND_PATH))
logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)
BG_HEIGHT, BG_WIDTH = background.shape[:2]
df = pd.read_csv(CSV_PATH)
schedule_df = pd.read_csv(ARTIST_TIME_PATH)
artist_df = pd.read_csv(ARTIST_TIME_PATH)


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

        artist_resized = add_paper_cutout_effect(artist_resized)

        # Create artifact collage (RGBA PIL)
        artifact_paths = list(ARTIFACTS_BG_PATH.glob("*.png"))
        background_pil = Image.open(BACKGROUND_PATH).convert("RGBA")

        if artifact_paths:
            artifact_collage = create_artifact_background_canvas((BG_WIDTH, BG_HEIGHT), artifact_paths)
            background_with_artifacts = Image.alpha_composite(background_pil, artifact_collage)
        else:
            background_with_artifacts = background_pil

        # Convert to OpenCV format (BGRA)
        composed = cv2.cvtColor(np.array(background_with_artifacts), cv2.COLOR_RGBA2BGRA)

        # Ensure background has 4 channels (BGRA) and reduce its opacity slightly
        if background.shape[2] == 3:
            background_bgra = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        else:
            background_bgra = background.copy()

        background_bgra[:, :, 3] = (background_bgra[:, :, 3].astype(np.float32) * 0.35).astype(np.uint8)

        # Overlay semi-transparent background over the collage
        composed = overlay_png(composed, background_bgra, 0, 0)

        composed = overlay_png(composed, artist_resized, (BG_WIDTH - new_size[0]) // 2, int(BG_HEIGHT * 0.1))
        
        # Resize logo
        logo_scale = 0.7  # 50% size, adjust this value as needed
        logo_resized = cv2.resize(
            logo,
            (int(logo.shape[1] * logo_scale), int(logo.shape[0] * logo_scale)),
            interpolation=cv2.INTER_AREA
        )
        composed = overlay_png(composed, logo_resized, BG_WIDTH - logo_resized.shape[1] - 30, 30)
        
        composed_pil = Image.fromarray(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB))
        composed_pil = draw_slogan_multiline(composed_pil, slogan, font_slogan, color)
        composed_pil = draw_schedule_text(composed_pil, artist_name, schedule_df, font_meta)
                # Step 3a: Save artist-specific .ics file
        artist_ics_content = generate_artist_calendar_ics(artist_name)
        artist_ics_filename = f"{artist_name.replace(' ', '_').lower()}.ics"
        ics_output_path = USER_OUTPUT_BASE / artist_ics_filename
        with open(ics_output_path, "w", encoding="utf-8") as f:
            f.write(artist_ics_content)

        # Create QR code linking to the .ics file
        # Assuming you'll later host these .ics files at a known URL structure:
        base_calendar_url = "https://git.arts.ac.uk/pages/23041393/PML_Project_Group_1/calendar/"
        qr_url = base_calendar_url + artist_ics_filename
        qr_pil = create_qr_code_image(qr_url, box_size=6)

        # Step 3c: Paste QR into image (top-left corner under schedule)
        qr_w, qr_h = qr_pil.size
        qr_x = 40  # same as schedule text x
        qr_y = 60 + 2 * 45 + 20  # schedule height + some spacing
        composed_pil.paste(qr_pil, (qr_x, qr_y), qr_pil)

        
        final_img = cv2.cvtColor(np.array(composed_pil), cv2.COLOR_RGB2BGR)

        output_dir = OUTPUT_BASE / label
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / file_path.name
        cv2.imwrite(str(out_path), final_img)
        all_generated[artist_name] = all_generated.get(artist_name, []) + [out_path]
        print(f"✅ Created: {out_path}")





# ================== USER PERSONALIZATION AND RECOMMENDATION ==================
# For each user profile, score available posters based on compatibility with user tastes.
# Copy top-N (default 6) matching posters to user-specific output folders.

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
        
        