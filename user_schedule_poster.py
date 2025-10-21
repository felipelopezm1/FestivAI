import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops
import qrcode
import random

# Paths and constants
CSV_PATH = Path("source_files/user_files/user_data.csv")
ARTIST_TIME_PATH = Path("source_files/artist_files/artist_files/artist_dataset.csv")
BACKGROUND_PATH = Path("source_files/artist_files/images/background/bg_genre_2.png")
ARTIFACTS_PATH = Path("source_files/artist_files/images/background/artifacts/lollapalooza")
LOGO_PATH = Path("source_files/artist_files/images/logo/logo.png")
FONT_PATH = Path("source_files/artist_files/images/fonts/Montserrat-VariableFont_wght.ttf")
QR_BASE_URL = "https://git.arts.ac.uk/pages/23041393/PML_Project_Group_1/calendar/"

DATE_FOLDER = datetime.now().strftime("%Y-%m-%d")
USER_OUTPUT_BASE = Path("source_files/user_files/output_by_user_final") / DATE_FOLDER
POSTER_WIDTH, POSTER_HEIGHT = 1080, 1920

# Load fonts and data
font_title = ImageFont.truetype(str(FONT_PATH), 90)
font_info = ImageFont.truetype(str(FONT_PATH), 54)
df = pd.read_csv(CSV_PATH)
schedule_df = pd.read_csv(ARTIST_TIME_PATH)
schedule_df['artist_normalized'] = schedule_df['artist'].astype(str).str.strip().str.lower()

# Helper: Day from date
def get_day_from_date(date_str):
    try:
        return datetime.strptime(date_str, "%B %d, %Y").strftime("%A")
    except:
        return "TBA"

# Helper: QR image from URL
def create_qr_code_image(url, box_size=6):
    qr = qrcode.QRCode(version=1, box_size=box_size, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGBA")

# Helper: draw text with stroke
def draw_text_with_outline(draw, position, text, font, fill, outline_fill, stroke_width):
    draw.text(position, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=outline_fill)

# Helper: draw centered multi-line title
def draw_text_centered(draw, text, font, center_x, y, fill, stroke_fill=(0, 0, 0), stroke_width=4):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        width = draw.textlength(line, font=font)
        draw.text((center_x - width / 2, y + i * 100), line, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

# Create background with artifact tiles
def create_artifact_background_canvas(size, tile_paths, max_tiles=80, min_scale=0.2, max_scale=0.6):
    base = Image.open(BACKGROUND_PATH).convert("RGBA").resize(size)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    width, height = size
    for _ in range(max_tiles):
        tile_path = random.choice(tile_paths)
        try:
            tile = Image.open(tile_path).convert("RGBA")
        except:
            continue
        scale = random.uniform(min_scale, max_scale)
        tile = tile.resize((int(tile.width * scale), int(tile.height * scale)))
        tile = tile.rotate(random.randint(0, 360), expand=True)
        x = random.randint(-tile.width // 2, width - tile.width // 2)
        y = random.randint(-tile.height // 2, height - tile.height // 2)
        canvas.alpha_composite(tile, (x, y))
    return Image.alpha_composite(base, canvas)

# Main poster generator
def create_user_schedule_poster(username, artist_files):
    artifact_paths = list(ARTIFACTS_PATH.glob("*.png"))
    background = create_artifact_background_canvas((POSTER_WIDTH, POSTER_HEIGHT), artifact_paths)
    draw = ImageDraw.Draw(background)

    draw_text_centered(draw, f"{username.title()}’s\nLine-up", font_title, POSTER_WIDTH // 2, 60,
                       fill=(255, 255, 255), stroke_fill=(0, 0, 0), stroke_width=6)

    section_height = 570
    for i, (ranked_file, artist_name) in enumerate(artist_files[:3]):
        y_offset = 250 + i * section_height

        artist_img = Image.open(ranked_file).convert("RGBA")
        artist_img.thumbnail((580, 400), Image.Resampling.LANCZOS)
        background.paste(artist_img, (80, y_offset), artist_img)

        meta = schedule_df[schedule_df['artist_normalized'] == artist_name.lower().strip()]
        if not meta.empty:
            row = meta.iloc[0]
            day = get_day_from_date(row.get("date", ""))
            time = row.get("time", "TBA")
            stage = row.get("stage", "TBA")
        else:
            day, time, stage = "TBA", "TBA", "TBA"

        draw_text_with_outline(draw, (540, y_offset + 125), artist_name, font_info,
                               fill=(255, 255, 255), outline_fill=(0, 0, 0), stroke_width=5)
        draw_text_with_outline(draw, (540, y_offset + 195), f"{day} · {time}", font_info,
                               fill=(255, 255, 0), outline_fill=(0, 0, 0), stroke_width=5)
        draw_text_with_outline(draw, (540, y_offset + 265), stage, font_info,
                               fill=(200, 200, 200), outline_fill=(0, 0, 0), stroke_width=5)

        # Artist-specific QR
        qr_url = QR_BASE_URL + f"{artist_name.replace(' ', '_').lower()}.ics"
        qr_img = create_qr_code_image(qr_url, box_size=4).resize((120, 120))
        qr_margin = 40
        qr_x = POSTER_WIDTH - qr_img.width - qr_margin
        qr_y = y_offset + 450 - qr_img.height - qr_margin
        background.paste(qr_img, (qr_x, qr_y), qr_img)


    # Add logo
    logo = Image.open(LOGO_PATH).convert("RGBA")
    logo.thumbnail((250, 150), Image.Resampling.LANCZOS)
    background.paste(logo, (POSTER_WIDTH - logo.width - 40, 40), logo)

    # Save
    user_folder = USER_OUTPUT_BASE / username
    schedule_folder = user_folder / "schedule"
    schedule_folder.mkdir(exist_ok=True)
    background.convert("RGB").save(schedule_folder / "schedule_poster.png")
    print(f"✅ Poster created for {username}")

# Main loop
for _, row in df.iterrows():
    uname = row['name'].strip().replace(" ", "_")
    user_folder = USER_OUTPUT_BASE / uname
    if not user_folder.exists():
        print(f"❌ No folder for {uname}")
        continue

    artist_files = []
    for file in sorted(user_folder.glob("*.png")):
        if file.name.lower().startswith("schedule"): continue
        parts = file.stem.split("_", 2)
        if len(parts) >= 3:
            artist_name = parts[2].replace("_", " ").title()
            artist_files.append((file, artist_name))

    if artist_files:
        create_user_schedule_poster(uname, artist_files)
    else:
        print(f"❌ No valid artist images for {uname}")
