import os
import unittest
import numpy as np
from pathlib import Path


def read_sudoku_from_dat(file_path: Path) -> np.ndarray:
    """Reads the sudoku board from a .dat file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()[2:]  # skip camera and img details
        board = []
        for line in lines:
            board.append([int(x) for x in line.split()])
        return np.array(board)


def read_sudoku_from_recognized(image_file):
    pass


class TestSudokuRecognition(unittest.TestCase):

    def setUp(self):
        """Setup test directories and files."""
        self.dataset_dir = "data/sudoku_dataset"
        self.image_extensions = [".jpg"]

    def test_sudoku_recognition(self):
        """Tests whether recognized Sudoku boards match .dat files."""
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if any(file.endswith(ext) for ext in self.image_extensions):
                    image_path = os.path.join(root, file)
                    dat_path = image_path.replace('.jpg', '.dat')
                    self.assertTrue(os.path.exists(dat_path),
                                    f"Missing .dat file for {file}")
                    recognized_board = read_sudoku_from_recognized(image_path)
                    expected_board = read_sudoku_from_dat(dat_path)

                    # compare numpy arrays
                    np.testing.assert_array_equal(
                        recognized_board, expected_board,
                        err_msg=f"Mismatch between boards for {file}"
                    )


if __name__ == "__main__":
    unittest.main()