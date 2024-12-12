def print_board(board):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - -")
        
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
                
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")

def find_empty(board):

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return (row, col)
    return None

def is_valid(board, number, position):

    for col in range(9):
        if board[position[0]][col] == number and position[1] != col:
            return False
    
    for row in range(9):
        if board[row][position[1]] == number and position[0] != row:
            return False
    
    box_x = position[1] // 3
    box_y = position[0] // 3
    
    for row in range(box_y * 3, box_y * 3 + 3):
        for col in range(box_x * 3, box_x * 3 + 3):
            if board[row][col] == number and (row, col) != position:
                return False
    
    return True

def solve(board):
    empty = find_empty(board)
    
    if not empty:
        return True
    
    row, col = empty
    
    for number in range(1, 10):
        # Check if the number is valid in this position
        if is_valid(board, number, (row, col)):
            # Place the number if it's valid
            board[row][col] = number
            
            # Recursively try to solve the rest of the board
            if solve(board):
                return True
            
            # If placing the number didn't lead to a solution,
            # backtrack by setting it back to 0
            board[row][col] = 0
    
    return False

# if __name__ == "__main__":
#     example_board = [
#         [5, 3, 0, 0, 7, 0, 0, 0, 0],
#         [6, 0, 0, 1, 9, 5, 0, 0, 0],
#         [0, 9, 8, 0, 0, 0, 0, 6, 0],
#         [8, 0, 0, 0, 6, 0, 0, 0, 3],
#         [4, 0, 0, 8, 0, 3, 0, 0, 1],
#         [7, 0, 0, 0, 2, 0, 0, 0, 6],
#         [0, 6, 0, 0, 0, 0, 2, 8, 0],
#         [0, 0, 0, 4, 1, 9, 0, 0, 5],
#         [0, 0, 0, 0, 8, 0, 0, 7, 9]
#     ]
    
#     print("Original Sudoku Board:")
#     print_board(example_board)
#     print("\nSolving...\n")
    
#     if solve(example_board):
#         print("Solved Sudoku Board:")
#         print_board(example_board)
#     else:
#         print("No solution exists!")