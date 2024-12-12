import random

import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_random_jpg(data_path):
    jpg_files = list(data_path.rglob('*.jpg'))
    if not jpg_files:
        return None
    file = random.choice(jpg_files)
    image = cv2.imread(str(file))
    return image
    # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def plot_original_and_cells(original_image, cells, spacing=5, border_thickness=2):
    """Plot the original image and extracted cells side by side,
    with separated cells and green borders around each cell."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # a blank canvas for the cells with spacing between them
    cell_height, cell_width = cells[0][0].shape
    canvas_height = 9 * (cell_height + 2 * border_thickness) + 8 * spacing
    canvas_width = 9 * (cell_width + 2 * border_thickness) + 8 * spacing
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    for i in range(9):
        for j in range(9):
            y_start = i * (cell_height + 2 * border_thickness + spacing)
            y_end = y_start + cell_height + 2 * border_thickness
            x_start = j * (cell_width + 2 * border_thickness + spacing)
            x_end = x_start + cell_width + 2 * border_thickness
            # green border
            canvas[y_start:y_end, x_start:x_end] = [0, 255, 0]
            # place the cell inside the border
            canvas[y_start + border_thickness:y_end - border_thickness,
                   x_start + border_thickness:x_end - border_thickness] = cv2.cvtColor(cells[i][j], cv2.COLOR_GRAY2BGR)

    # extracted cells
    axes[1].imshow(canvas)
    axes[1].set_title("Extracted Cells")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
