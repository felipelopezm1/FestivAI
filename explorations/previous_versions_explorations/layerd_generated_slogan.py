import os
import cv2
from pathlib import Path
from Ollama_Slogan import generate_slogan
import re

# Define paths
INPUT_DIR = Path("source_files/artist_files/images/medium_artists")
OUTPUT_DIR = Path("source_files/artist_files/images/slogan_outputs")
BACKGROUND_PATH = Path("source_files/artist_files/images/background/bg_genre_1.png")
LOGO_PATH = Path("source_files/artist_files/images/logo/logo.png")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002500-\U00002BEF"  # chinese characters
        "\U00002702-\U000027B0"
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
        "\ufe0f"  # dingbats
        "\u3030"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def draw_slogan_multiline_left_bottom(img, text, max_width_px=1000, bottom_padding=50, left_padding=60):
    words = text.split()
    lines = []
    current_line = []

    # Break the slogan into multiple lines if needed
    for word in words:
        trial_line = " ".join(current_line + [word])
        size = cv2.getTextSize(trial_line, FONT, FONT_SCALE, THICKNESS)[0]
        if size[0] > max_width_px:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    lines.append(" ".join(current_line))

    # Calculate total text height
    line_height = int(cv2.getTextSize("Test", FONT, FONT_SCALE, THICKNESS)[0][1] * 1.5)
    total_text_height = line_height * len(lines)

    # Start drawing from bottom up
    start_y = BG_HEIGHT - bottom_padding - total_text_height + line_height

    for i, line in enumerate(lines):
        text_y = start_y + i * line_height
        text_x = left_padding

        # Shadow
        cv2.putText(img, line, (text_x + 2, text_y + 2), FONT, FONT_SCALE, (0, 0, 0), THICKNESS + 1, LINE_TYPE)
        # Main text
        cv2.putText(img, line, (text_x, text_y), FONT, FONT_SCALE, COLOR, THICKNESS, LINE_TYPE)

    return img



# Load background and logo
background = cv2.imread(str(BACKGROUND_PATH))
if background is None:
    raise FileNotFoundError("Background image not found!")

logo = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)  # Preserve transparency
if logo is None:
    raise FileNotFoundError("Logo image not found!")

BG_HEIGHT, BG_WIDTH = background.shape[:2]

# Font config for OpenCV
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 4
COLOR = (0, 0, 0)
THICKNESS = 7
LINE_TYPE = cv2.LINE_AA
BOTTOM_PADDING = 50

def overlay_png(base_img, overlay_img, x, y):
    """Overlay a transparent PNG (with alpha channel) onto a base BGR image."""
    if overlay_img.shape[2] != 4:
        raise ValueError("Overlay image does not have an alpha channel")

    h, w = overlay_img.shape[:2]
    alpha = overlay_img[:, :, 3] / 255.0
    for c in range(3):  # Apply only to B, G, R channels
        base_img[y:y+h, x:x+w, c] = (
            alpha * overlay_img[:, :, c] +
            (1 - alpha) * base_img[y:y+h, x:x+w, c]
        )
    return base_img

# Process images
for file_path in INPUT_DIR.glob("ma_*.png"):
    artist_raw = file_path.stem.replace("ma_", "").replace("_", " ")
    artist_name = artist_raw.title()

    try:
        slogan = generate_slogan(artist_name).strip().strip('"').strip("'").strip()
        slogan = remove_emojis(slogan)
        print(f"Generated slogan for {artist_name}: {slogan}")
        print(f"Emojies Removed")
    except Exception as e:
        print(f"Failed to get slogan for {artist_name}: {e}")
        continue

    artist_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)  # preserves transparency (alpha channel)
    if artist_img is None:
        print(f"Could not load image {file_path}")
        continue

    # Resize artist image
    max_width = int(BG_WIDTH * 0.9)
    max_height = int(BG_HEIGHT * 0.6)
    h, w = artist_img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_size = (int(w * scale), int(h * scale))
    artist_resized = cv2.resize(artist_img, new_size, interpolation=cv2.INTER_AREA)

    # Start composing
    composed = background.copy()

    # Overlay artist image (centered)
    x_offset = (BG_WIDTH - new_size[0]) // 2
    y_offset = int(BG_HEIGHT * 0.1)
    artist_resized = cv2.resize(artist_img, new_size, interpolation=cv2.INTER_AREA)
    composed = overlay_png(composed, artist_resized, x_offset, y_offset)


    # Add multiline slogan text near bottom
    composed = draw_slogan_multiline_left_bottom(composed, slogan, max_width_px=1000)


    # Overlay logo (top-right)
    logo_x = BG_WIDTH - logo.shape[1] - 30  # 30px margin
    logo_y = 30
    composed = overlay_png(composed, logo, logo_x, logo_y)

    # Save result
    output_path = OUTPUT_DIR / file_path.name
    cv2.imwrite(str(output_path), composed)
    print(f"Saved: {output_path}")  #comment