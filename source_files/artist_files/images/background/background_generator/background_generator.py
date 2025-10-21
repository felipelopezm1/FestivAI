import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# === CONFIGURATION ===
SEED_FOLDER = "source_files/artist_files/seeds/input_seeds_festival_story_ads"
OUTPUT_FOLDER = "source_files/artist_files/images/background"
MODEL_NAME = "Lykon/dreamshaper-8"
FIXED_RESOLUTION = (1080, 1350)
NUM_VARIATIONS = 3  # how many images per seed

# === LOAD MODEL ===
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
    pipe.enable_xformers_memory_efficient_attention()

# === GENERATE IMAGES ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
seed_files = [f for f in os.listdir(SEED_FOLDER) if f.endswith(".txt")]

for seed_file in seed_files:
    with open(os.path.join(SEED_FOLDER, seed_file), "r", encoding="utf-8") as f:
        concept = f.read().strip()

    genre = os.path.splitext(seed_file)[0]

    for i in range(1, NUM_VARIATIONS + 1):
        prompt = (
            f"Abstract geometric background, inspired by {concept}, "
            f"vibrant shapes, minimalism, flowing curves, clean lines, high detail, colorful aesthetic"
        )

        print(f"🎨 Generating: bg_{genre}_{i}.png")
        image = pipe(prompt, height=FIXED_RESOLUTION[1], width=FIXED_RESOLUTION[0]).images[0]

        filename = f"bg_{genre}_{i}.png"
        image.save(os.path.join(OUTPUT_FOLDER, filename))

print("✅ All images saved in 'source_files/artist_files/images/background/'")
