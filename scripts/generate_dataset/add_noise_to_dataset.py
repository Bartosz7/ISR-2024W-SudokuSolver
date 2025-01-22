import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import random


def augment_dataset(input_dir, output_dir="1000augmented_dataset"):
    Path(output_dir).mkdir(exist_ok=True)

    def rotate_image(img):
        angle = random.uniform(-15, 15)
        bg_color = img.getpixel((0, 0))
        rotated = img.rotate(angle, expand=True, fillcolor=bg_color)
        rotated = rotated.resize(img.size)

        # Add noise after rotation
        rotated_array = np.array(rotated)
        noise = np.random.normal(0, 15, rotated_array.shape)
        rotated_array = np.clip(rotated_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(rotated_array)

    def add_lines(draw, width, height):
        line_types = ["left", "right", "upper", "lower", "corner"]
        line_type = random.choice(line_types)

        if line_type == "left":
            x = random.randint(0, width // 3)
            draw.line([(x, 0), (x, height)], fill="black", width=2)
        elif line_type == "right":
            x = random.randint(2 * width // 3, width)
            draw.line([(x, 0), (x, height)], fill="black", width=2)
        elif line_type == "upper":
            y = random.randint(0, height // 3)
            draw.line([(0, y), (width, y)], fill="black", width=2)
        elif line_type == "lower":
            y = random.randint(2 * height // 3, height)
            draw.line([(0, y), (width, y)], fill="black", width=2)
        elif line_type == "corner":
            corner = random.choice(["tl", "tr", "bl", "br"])
            line_length = min(width, height) // 3

            if corner == "tl":
                draw.line(
                    [(0, line_length), (0, 0), (line_length, 0)], fill="black", width=2
                )
            elif corner == "tr":
                draw.line(
                    [(width - line_length, 0), (width, 0), (width, line_length)],
                    fill="black",
                    width=2,
                )
            elif corner == "bl":
                draw.line(
                    [(0, height - line_length), (0, height), (line_length, height)],
                    fill="black",
                    width=2,
                )
            elif corner == "br":
                draw.line(
                    [
                        (width - line_length, height),
                        (width, height),
                        (width, height - line_length),
                    ],
                    fill="black",
                    width=2,
                )

    def add_outline(draw, width, height):
        if random.random() < 0.5:  # 50% chance to add outline
            margin = random.randint(5, 20)
            draw.rectangle(
                [(margin, margin), (width - margin, height - margin)],
                outline="black",
                width=random.randint(1, 3),
            )

    def change_background_color(img_array):
        color_shifts = {
            "grey": np.array([0.9, 0.9, 0.9]),
            "blue": np.array([0.9, 0.9, 1.0]),
            "yellow": np.array([1.0, 1.0, 0.9]),
        }
        color = random.choice(list(color_shifts.keys()))
        mask = img_array > 200
        img_array = np.stack([img_array] * 3, axis=-1)
        img_array[mask] = img_array[mask] * color_shifts[color]
        return img_array

    for digit_folder in os.listdir(input_dir):
        digit_path = os.path.join(input_dir, digit_folder)
        if os.path.isdir(digit_path):
            out_digit_path = os.path.join(output_dir, digit_folder)
            Path(out_digit_path).mkdir(exist_ok=True)

            for img_file in os.listdir(digit_path):
                img = Image.open(os.path.join(digit_path, img_file))
                img_array = np.array(img)

                # Change background
                img_array = change_background_color(img_array)
                img = Image.fromarray(img_array.astype(np.uint8))

                # Add lines and outline
                draw = ImageDraw.Draw(img)
                add_lines(draw, img.width, img.height)
                add_outline(draw, img.width, img.height)

                # Final rotation
                img = rotate_image(img)

                img.save(os.path.join(out_digit_path, f"augmented_{img_file}"))

    return f"Dataset augmented and saved to {output_dir}"
