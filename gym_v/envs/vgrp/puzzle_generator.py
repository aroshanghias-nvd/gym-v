"""Universal puzzle generator using backtracking for VGRP-Bench puzzles."""

from __future__ import annotations

import random
from typing import Any


def solve_puzzle_backtrack(
    factory, game_state: dict[str, Any], max_attempts: int = 1000
) -> bool:
    """Simple backtracking solver that works with any PuzzleFactory.

    Args:
        factory: A PuzzleFactory instance with constraints and get_possible_values method
        game_state: Dictionary containing 'board' and optionally other state like 'clues'
        max_attempts: Maximum recursion attempts before giving up

    Returns:
        True if a solution was found, False otherwise
    """
    board = game_state["board"]
    size = len(board)

    if not hasattr(factory, "_solve_attempts"):
        factory._solve_attempts = 0

    factory._solve_attempts += 1
    if factory._solve_attempts > max_attempts:
        return False

    # Find first empty cell (0)
    for row in range(size):
        for col in range(size):
            if board[row][col] == 0:
                # Try each possible value
                possible_values = factory.get_possible_values(game_state, row, col)
                random.shuffle(possible_values)  # Randomize for variation

                for value in possible_values:
                    board[row][col] = value
                    if solve_puzzle_backtrack(factory, game_state, max_attempts):
                        return True
                    board[row][col] = 0
                return False

    return True  # All cells filled


def generate_puzzle(
    factory, size: int, num_hints: int, max_attempts: int = 100, **extra_state
) -> tuple[list, list] | None:
    """Generate a puzzle with solution using the factory.

    Args:
        factory: A PuzzleFactory instance
        size: Size of the puzzle grid
        num_hints: Number of cells to leave as hints
        max_attempts: How many times to try generating a valid puzzle
        **extra_state: Additional state dict items (e.g., clues, inequalities)

    Returns:
        Tuple of (puzzle_board, solution_board) or None if generation failed
    """
    for _attempt in range(max_attempts):
        # Create an empty board
        board = [[0 for _ in range(size)] for _ in range(size)]
        game_state = {"board": board, **extra_state}

        # Reset solve attempts counter
        factory._solve_attempts = 0

        # Try to fill the board
        if solve_puzzle_backtrack(factory, game_state):
            # Save solution
            solution = [row[:] for row in board]

            # Create puzzle by sampling hints
            if num_hints > 0:
                puzzle = factory.sample_hints(solution, num_hints)
            else:
                puzzle = [[0 for _ in range(size)] for _ in range(size)]

            return puzzle, solution

    return None
