from tqdm import tqdm
import os
import numpy as np
from pathlib import Path
from typing import Dict
from collections import defaultdict


def read_sudoku_from_dat(file_path: Path) -> np.ndarray:
    """Reads the sudoku board from a .dat file."""
    with open(file_path, "r") as file:
        lines = file.readlines()[2:]  # skip camera and img details
        board = []
        for line in lines:
            board.append([int(x) for x in line.split()])
        return np.array(board)


def evaluate_recognition(
    dataset_dir: str,
    image_extensions: list = [".jpg"],
    read_recognized_func=None,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluates the Sudoku digit recognition based on the dataset folder.

    Args:
        dataset_dir (str): Path to the dataset directory.
        image_extensions (list): List of valid image file extensions.
        read_recognized_func (callable): Function to recognize Sudoku board from an image file.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    if not read_recognized_func:
        raise ValueError("Please provide a function to recognize Sudoku boards from images.")

    total_boards = 0
    total_correct_boards = 0
    total_digits = 0
    total_correct_digits = 0
    total_errors = 0
    digit_errors = defaultdict(int)

    # Collect all image files
    image_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                image_files.append(image_path)

    # Iterate over image files with tqdm progress bar
    for image_path in tqdm(image_files, desc="Evaluating Sudoku Boards"):
        dat_path = image_path.replace(".jpg", ".dat")

        if not os.path.exists(dat_path):
            print(f"Warning: Missing .dat file for {image_path}")
            continue

        # Read the expected and recognized Sudoku boards
        expected_board = read_sudoku_from_dat(dat_path)
        recognized_board = read_recognized_func(image_path, **kwargs)

        total_boards += 1
        if np.array_equal(expected_board, recognized_board):
            total_correct_boards += 1

        # Compute digit-level metrics
        for i in range(9):
            for j in range(9):
                total_digits += 1
                if expected_board[i, j] == recognized_board[i, j]:
                    total_correct_digits += 1
                else:
                    total_errors += 1
                    digit_errors[(i, j)] += 1

    # Compute metrics
    avg_recognition_per_board = total_correct_digits / total_digits if total_digits > 0 else 0
    fully_correct_boards = total_correct_boards
    overall_digit_accuracy = total_correct_digits / total_digits if total_digits > 0 else 0
    error_rate = total_errors / total_digits if total_digits > 0 else 0

    return {
        "total_boards": total_boards,
        "fully_correct_boards": fully_correct_boards,
        "avg_recognition_per_board": avg_recognition_per_board,
        "overall_digit_accuracy": overall_digit_accuracy,
        "total_errors": total_errors,
        "error_rate": error_rate,
    }


# if __name__ == "__main__":
#     dataset_dir = "data/sudoku_dataset"
#     metrics = evaluate_recognition(
#         dataset_dir=dataset_dir,
#         read_recognized_func=read_sudoku_from_recognized
#     )

#     print("Evaluation Metrics:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value}")
