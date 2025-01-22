import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import random


def generate_cipher_dataset(
    output_dir="final_dataset_test_1000", image_size=(200, 200), samples_per_digit=1000
):
    Path(output_dir).mkdir(exist_ok=True)
    font_size = 80

    font_paths = [
        "/content/sample_data/Arial.ttf",
        "/content/fonts/Courier_New.ttf",
        "/content/fonts/TimesNewRoman.ttf",
        "/content/fonts/Verdana.ttf",
    ]

    sizes = [50, 60, 70, 80, 90]
    available_fonts = [path for path in font_paths if os.path.exists(path)]

    if not available_fonts:
        raise Exception("No fonts found. Please check font paths.")

    for digit in range(1, 10):
        digit_dir = os.path.join(output_dir, str(digit))
        Path(digit_dir).mkdir(exist_ok=True)

        for sample in range(samples_per_digit):
            font_path = random.choice(available_fonts)
            font_size = random.choice(sizes)
            font = ImageFont.truetype(font_path, font_size)

            img = Image.new("L", image_size, color="white")
            draw = ImageDraw.Draw(img)

            left, top, right, bottom = font.getbbox(str(digit))
            w = right - left
            h = bottom - top
            x = (image_size[0] - w) // 2
            y = (image_size[1] - h) // 2

            draw.text((x, y), str(digit), fill="black", font=font)
            img_array = np.array(img)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array)
            img.save(os.path.join(digit_dir, f"digit_{sample}.png"))

    return f"Generated {samples_per_digit} images for each digit 1-9 in {output_dir}"
