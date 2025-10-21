import os
import cv2
from pathlib import Path
from Ollama_Slogan import generate_slogan

# Define paths
INPUT_DIR = Path("source_files/artist_files/images/medium_artists")
OUTPUT_DIR = Path("source_files/artist_files/images/slogan_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Font config for OpenCV
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
COLOR = (255, 255, 255)  # white
THICKNESS = 2
LINE_TYPE = cv2.LINE_AA
BOTTOM_PADDING = 20

# Iterate through each image
for file_path in INPUT_DIR.glob("ma_*.png"):
    artist_raw = file_path.stem.replace("ma_", "").replace("_", " ")
    artist_name = artist_raw.title()

    try:
        # Generate slogan using Ollama
        slogan = generate_slogan(artist_name)
        print(f"Generated slogan for {artist_name}: {slogan}")
    except Exception as e:
        print(f"Failed to get slogan for {artist_name}: {e}")
        continue

    # Read image
    image = cv2.imread(str(file_path))
    if image is None:
        print(f"Could not load image {file_path}")
        continue

    # Get text size and position
    text_size = cv2.getTextSize(slogan, FONT, FONT_SCALE, THICKNESS)[0]
    text_x = int((image.shape[1] - text_size[0]) / 2)
    text_y = image.shape[0] - BOTTOM_PADDING

    # Add text with background for contrast
    overlay = image.copy()
    cv2.putText(overlay, slogan, (text_x, text_y), FONT, FONT_SCALE, COLOR, THICKNESS, LINE_TYPE)
    image = overlay

    # Save output image
    output_path = OUTPUT_DIR / file_path.name
    cv2.imwrite(str(output_path), image)
    print(f"Saved: {output_path}")
