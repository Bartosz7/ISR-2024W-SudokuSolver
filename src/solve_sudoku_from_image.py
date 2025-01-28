import cv2
import numpy as np
import os
import torch
import fire
from src.predict_digit import load_model, load_blank_classifier, predict_digit
from src.sudoku_solve_algoithm import solve, print_board
from src.model import CNNClassifier  # Ensure this is the correct import path
import torch.nn.functional as F


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


def is_cell_empty(cell_image, blank_classifier, device, threshold=0.5):
    """
    Use a pre-trained blank classifier to detect if a cell is blank.

    Args:
        cell_image (np.ndarray): Grayscale image of the cell (28x28 or similar size).
        blank_classifier (torch.nn.Module): Pre-trained blank cell classifier.
        device (torch.device): The device (CPU or GPU) to run inference.
        threshold (float): Confidence threshold for blank classification.

    Returns:
        bool: True if the cell is classified as blank, otherwise False.
    """
    blank_classifier.eval()  # Set model to evaluation mode

    # Resize the cell image to the expected input size
    cell_image = cv2.resize(cell_image, (28, 28))  # Ensure this matches the training size

    # Preprocess the cell image
    cell_tensor = torch.tensor(cell_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    cell_tensor = cell_tensor.to(device) / 255.0  # Normalize to [0, 1]

    # Run inference
    with torch.no_grad():
        output = blank_classifier(cell_tensor)
        probabilities = F.softmax(output, dim=1)  # Get probabilities for each class
        blank_prob = probabilities[0][0].item()  # Probability of the "blank" class

    # had to invert, as i mistook the classes
    return not (blank_prob > threshold)


def solve_sudoku_from_image(image_path: str, print_results: bool = False, solve_sudoku: bool = True):
    """Main function for solving sudoku board from image."""
    # load and convert to grayscale
    image = cv2.imread(str(image_path))
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
    )
    # find the largest contour (assumed to be the board grid)
    # and get its bounding box
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # warp the image to better fit the grid, standard size 450 x 450
    pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(binary, matrix, (450, 450))

    # removing small objects in the foreground
    opening = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel=(3, 3), iterations=1)
    opening = cv2.bitwise_not(opening)

    # deconstructing board into cells
    cells = fragment_image(opening)
    save_cells_as_images(cells, "tempdir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_model("../models/final_model_0_3_with_augmented.pth")

    # Load the blank cell classifier
    blank_classifier = CNNClassifier()
    blank_classifier.load_state_dict(torch.load("../models/blank_cell_classifier_1.pth"))
    blank_classifier.to(device)
    blank_classifier.eval()

    board = [[0 for j in range(9)] for i in range(9)]
    for image_path in os.listdir("tempdir"):
        row, col = int(image_path.split("_")[1]), int(image_path.split("_")[2][0])
        full_img_path = os.path.join("tempdir", image_path)
        # Check if the cell is blank
        cell_img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
        if is_cell_empty(cell_img, blank_classifier, device):
            board[row][col] = 0
        else:
            confidence, digit = predict_digit(model=model, device=device,image_path=full_img_path)
            # print(row, col, digit, confidence)
            board[row][col] = digit
    board = np.array(board)

    if print_results:
        print("Recognized Board:")
        print(board)

    if solve_sudoku:
        solve(board)
        if print_results:
            print("Solved Board:")
            print(board)
    return board


if __name__ == "__main__":
    fire.Fire(solve_sudoku_from_image)
