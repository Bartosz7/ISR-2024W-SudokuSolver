import cv2
import numpy as np
import os
import fire
from src.predict_digit import predict_digit
from src.cnn_arch import load_model
from src.sudoku_solve_algoithm import solve, print_board

# label smoothing as a method for smoothing probs, during trianing
# def load_model(path="digit_model.pth", device=None):
#     # If device is not specified, use "cuda" if available
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SimpleDigitCNN(dropout_p=0.5)
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.to(device)
#     model.eval()
#     return model


def save_cells_as_images(cells, output_dir):
    """Save individual cells as images in a specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            cell_path = os.path.join(output_dir, f"cell_{i}_{j}.png")
            cv2.imwrite(cell_path, cell)


def fragment_image(board_image):
    """Fragments the board image to the individual cells images"""
    cell_height, cell_width = board_image.shape[0] // 9, board_image.shape[1] // 9
    cells = []
    for i in range(9):
        row = []
        for j in range(9):
            x, y = j * cell_width, i * cell_height
            cell = board_image[y:y + cell_height, x:x + cell_width]
            row.append(cell)
        cells.append(row)
    return cells


def is_cell_empty(cell_image):
    """
    Combined approach using both pixel density and contour analysis
    """
    # convert to grayscale if not already
    if len(cell_image.shape) > 2:
        cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

    # check mean intensity
    mean_intensity = np.mean(cell_image)
    if mean_intensity > 215:  # possibly white cell
        return True

    # check contours
    _, binary = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return True

    largest_contour_area = max(cv2.contourArea(cnt) for cnt in contours)
    cell_area = cell_image.shape[0] * cell_image.shape[1]

    return largest_contour_area < (cell_area * 0.03)


def solve_sudoku_from_image(image_path: str):
    """Main function for solving sudoku board from image."""
    # load and convert to grayscale
    image = cv2.imread(str(image_path))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
    )
    # alternative:
    # _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find the largest contour (assumed to be the board grid)
    largest_contour = max(contours, key=cv2.contourArea)
    # and get its bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # warp the image to better fit the grid
    pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    # standard size
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(binary, matrix, (450, 450))
    # removing small objects in the foreground
    opening = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel=(3, 3), iterations=1)
    opening = cv2.bitwise_not(opening)
    # deconstructing board into cells
    cells = fragment_image(opening)
    save_cells_as_images(cells, "tempdir")
    model = load_model("../models/digit1.pth")

    board = [[0 for j in range(9)] for i in range(9)]
    for image_path in os.listdir("tempdir"):
        row, col = int(image_path.split("_")[1]), int(image_path.split("_")[2][0])
        full_img_path = os.path.join("tempdir", image_path)
        digit = predict_digit(model=model, image_path=full_img_path)
        board[row][col] = digit
    board = np.array(board)
    return board


if __name__ == "__main__":
    fire.Fire(solve_sudoku_from_image)
