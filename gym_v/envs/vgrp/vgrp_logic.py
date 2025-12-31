"""Core logic for VGRP-Bench puzzles (factories, constraints, solvers)."""

from __future__ import annotations

import copy
import random
from typing import Any

# =============================================================================
# Base Classes
# =============================================================================


class Constraint:
    def __init__(self) -> None:
        self.name = ""

    def check(self, game_state: dict[str, Any]) -> bool:
        pass


class PuzzleFactory:
    def __init__(self) -> None:
        self.constraints = []
        self.game_name = "unknown"
        self.size = 0

    def sample_hints(
        self, board: list[list[Any]], num_sample_hints: int
    ) -> list[list[Any]]:
        # Create a new board filled with zeros
        new_board = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]
        # Sample num_sample_hints cells to keep from the original board
        sampled_cells = random.sample(
            range(len(board) * len(board[0])), num_sample_hints
        )
        for cell in sampled_cells:
            row = cell // len(board[0])
            col = cell % len(board[0])
            new_board[row][col] = board[row][
                col
            ]  # Copy only the sampled cells from original board
        return new_board

    def check(self, game_state: dict[str, Any]) -> bool:
        for constraint in self.constraints:
            if not constraint.check(game_state):
                return False
        return True

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[Any]:
        pass


# =============================================================================
# Solvers
# =============================================================================


def solve_puzzle_backtrack(
    factory, game_state: dict[str, Any], max_attempts: int = 1000
) -> bool:
    """Simple backtracking solver."""
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
                random.shuffle(possible_values)

                for value in possible_values:
                    board[row][col] = value
                    if solve_puzzle_backtrack(factory, game_state, max_attempts):
                        return True
                    board[row][col] = 0
                return False

    return factory.check(game_state)  # All cells filled, verify final state


def generate_puzzle(
    factory,
    size: int,
    num_hints: int,
    max_attempts: int = 100,
    initial_board: list[list[Any]] | None = None,
    **extra_state,
) -> tuple[list[list[Any]], list[list[Any]]] | None:
    """Generate a puzzle with solution using the factory.

    Args:
        factory: The PuzzleFactory to use.
        size: Grid size.
        num_hints: Number of hints to reveal in the puzzle (if applicable).
        max_attempts: Max attempts to generate a valid solution.
        initial_board: Optional initial board state (e.g. containing trees).
        **extra_state: Additional state for the constraints (e.g. regions, clues).
    """
    for _attempt in range(max_attempts):
        # Create board
        if initial_board is not None:
            board = copy.deepcopy(initial_board)
        else:
            board = [[0 for _ in range(size)] for _ in range(size)]

        game_state = {"board": board, **extra_state}

        # Reset solve attempts counter
        factory._solve_attempts = 0

        # Try to fill the board
        if solve_puzzle_backtrack(factory, game_state):
            # Save solution (deep copy to avoid reference issues)
            solution = copy.deepcopy(board)

            # Create puzzle by sampling hints
            if num_hints > 0:
                puzzle = factory.sample_hints(solution, num_hints)
            else:
                puzzle = [[0 for _ in range(size)] for _ in range(size)]

            return puzzle, solution

    return None


# =============================================================================
# Common Constraints
# =============================================================================


class ConstraintRowNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        for row in board:
            values = [x for x in row if x != 0]
            if len(set(values)) != len(values):
                return False
        return True


class ConstraintColNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for col in range(size):
            values = [board[row][col] for row in range(size) if board[row][col] != 0]
            if len(set(values)) != len(values):
                return False
        return True


# =============================================================================
# Binairo
# =============================================================================


class ConstraintRowBalance(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_balance"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        expected_count = size // 2

        assert all(
            all(cell != "*" for cell in row) for row in board
        ), "'*' should be replaced by '0' in the initialization board"

        for row in board:
            if 0 not in row:  # Only check completed rows
                white_count = sum(1 for x in row if x == "w")
                black_count = sum(1 for x in row if x == "b")
                if white_count != black_count or white_count != expected_count:
                    return False
        return True


class ConstraintColBalance(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_balance"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        expected_count = size // 2

        for col in range(size):
            column = [board[row][col] for row in range(size)]
            if 0 not in column and "*" not in column:  # Only check completed columns
                white_count = sum(1 for x in column if x == "w")
                black_count = sum(1 for x in column if x == "b")
                if white_count != black_count or white_count != expected_count:
                    return False
        return True


class ConstraintNoTripleAdjacent(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_no_triple_adjacent"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Check rows
        for row in range(size):
            for col in range(size - 2):
                if (
                    board[row][col] != 0
                    and board[row][col] == board[row][col + 1] == board[row][col + 2]
                ):
                    return False

        # Check columns
        for col in range(size):
            for row in range(size - 2):
                if (
                    board[row][col] != 0
                    and board[row][col] == board[row + 1][col] == board[row + 2][col]
                ):
                    return False
        return True


class BinairoPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4 or size % 2 != 0:
            raise ValueError("Size must be an even number greater than or equal to 4")

        self.game_name = "binairo"
        self.size = size
        self.constraints = [
            ConstraintRowBalance(),
            ConstraintColBalance(),
            ConstraintNoTripleAdjacent(),
        ]

        self.all_possible_values = ["w", "b"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Battleships
# =============================================================================


class ConstraintBattleships(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_battleships"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Check if ships touch diagonally or orthogonally
        for i in range(size):
            for j in range(size):
                if isinstance(board[i][j], tuple):
                    # Revealed ship with direction logic (omitted for brevity if simpler logic suffices,
                    # but keeping structure)
                    _, direction = board[i][j]
                    if direction in "<>-":
                        for di in [-1, 1]:
                            if 0 <= i + di < size and board[i + di][j] == "s":
                                return False
                    elif direction in "^V|":
                        for dj in [-1, 1]:
                            if 0 <= j + dj < size and board[i][j + dj] == "s":
                                return False
                elif board[i][j] == "s":
                    # Regular ship cell checks
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < size
                                and 0 <= nj < size
                                and (
                                    board[ni][nj] == "s"
                                    or (
                                        isinstance(board[ni][nj], tuple)
                                        and board[ni][nj][0] == "s"
                                    )
                                )
                                and (di != 0 and dj != 0)
                            ):  # Diagonal check
                                return False
        return True


class ConstraintBattleshipsHints(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_battleships_hints"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state["hints"]

        row_hints = hints["row_hints"]
        col_hints = hints["col_hints"]
        ships = hints["ships"]
        size = len(board)

        # Calculate total required ship cells
        total_ship_cells_required = sum(
            int(length) * int(count) for length, count in ships.items()
        )
        total_ship_cells_selected = sum(
            1 for i in range(size) for j in range(size) if board[i][j] == "s"
        )
        total_undefined_cells = sum(
            1 for i in range(size) for j in range(size) if board[i][j] == 0
        )

        if (
            total_ship_cells_selected + total_undefined_cells
            < total_ship_cells_required
        ):
            return False

        if total_ship_cells_selected > total_ship_cells_required:
            return False

        # Check row hints
        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)
            if all(cell != 0 and cell != -1 for cell in board[i]):
                if row_selected != row_hints[i]:
                    return False
            else:
                if row_selected > row_hints[i]:
                    return False
                if row_selected + row_undefined < row_hints[i]:
                    return False

        # Check col hints
        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 and board[i][j] != -1 for i in range(size)):
                if col_selected != col_hints[j]:
                    return False
            else:
                if col_selected > col_hints[j]:
                    return False
                if col_selected + col_undefined < col_hints[j]:
                    return False

        # Check ship shapes when full
        if total_undefined_cells == 0:
            visited = [[False] * size for _ in range(size)]
            ship_lengths = []

            def get_ship_length(i: int, j: int) -> int:
                if (
                    i < 0
                    or i >= size
                    or j < 0
                    or j >= size
                    or visited[i][j]
                    or board[i][j] != "s"
                ):
                    return 0
                visited[i][j] = True
                length = 1
                if j + 1 < size and board[i][j + 1] == "s":
                    for col in range(j + 1, size):
                        if board[i][col] != "s":
                            break
                        visited[i][col] = True
                        length += 1
                elif i + 1 < size and board[i + 1][j] == "s":
                    for row in range(i + 1, size):
                        if board[row][j] != "s":
                            break
                        visited[row][j] = True
                        length += 1
                return length

            for i in range(size):
                for j in range(size):
                    if not visited[i][j] and board[i][j] == "s":
                        ship_lengths.append(get_ship_length(i, j))

            ship_counts = {}
            for length in ship_lengths:
                ship_counts[length] = ship_counts.get(length, 0) + 1

            for length, count in ships.items():
                if ship_counts.get(int(length), 0) != int(count):
                    return False

        return True


class BattleshipsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "battleships"
        self.size = size
        self.constraints = [ConstraintBattleships(), ConstraintBattleshipsHints()]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        board = game_state["board"]
        if board[row][col] != 0:
            return []
        possible_values = []
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Futoshiki
# =============================================================================


class ConstraintInequality(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_inequality"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        inequalities = game_state.get("inequalities", {"row": [], "col": []})

        row_ineq = inequalities.get(
            "row", [["" for _ in range(size - 1)] for _ in range(size)]
        )
        if not row_ineq:
            row_ineq = [["" for _ in range(size - 1)] for _ in range(size)]

        for row in range(size):
            for col in range(size - 1):
                if row_ineq[row][col] == "<":
                    if board[row][col] != 0 and board[row][col + 1] != 0:
                        if board[row][col] >= board[row][col + 1]:
                            return False
                elif row_ineq[row][col] == ">":
                    if board[row][col] != 0 and board[row][col + 1] != 0:
                        if board[row][col] <= board[row][col + 1]:
                            return False

        col_ineq = inequalities.get(
            "col", [["" for _ in range(size)] for _ in range(size - 1)]
        )
        if not col_ineq:
            col_ineq = [["" for _ in range(size)] for _ in range(size - 1)]

        for row in range(size - 1):
            for col in range(size):
                if col_ineq[row][col] == "^":
                    if board[row][col] != 0 and board[row + 1][col] != 0:
                        if board[row][col] >= board[row + 1][col]:
                            return False
                elif col_ineq[row][col] == "v":
                    if board[row][col] != 0 and board[row + 1][col] != 0:
                        if board[row][col] <= board[row + 1][col]:
                            return False
        return True


class FutoshikiPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "futoshiki"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintInequality(),
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Hitori
# =============================================================================


class ConstraintHitoriNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        numbers = game_state.get("numbers", [])
        size = len(board)
        for i in range(size):
            row_values = [numbers[i][j] for j in range(size) if board[i][j] == "e"]
            col_values = [numbers[j][i] for j in range(size) if board[j][i] == "e"]
            if len(row_values) != len(set(row_values)) or len(col_values) != len(
                set(col_values)
            ):
                return False
        return True


class ConstraintHitoriAdjacent(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_adjacent"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for row in range(size):
            for col in range(size):
                if board[row][col] == "s":
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == "s":
                            return False
        return True


class ConstraintHitoriConnected(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_connected"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        start = None
        for r in range(size):
            for c in range(size):
                if board[r][c] in ["e", 0]:
                    start = (r, c)
                    break
            if start:
                break
        if not start:
            return False
        visited = [[False] * size for _ in range(size)]
        queue = [start]
        visited[start[0]][start[1]] = True
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < size
                    and 0 <= nc < size
                    and not visited[nr][nc]
                    and board[nr][nc] in ["e", 0]
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        for r in range(size):
            for c in range(size):
                if board[r][c] in ["e", 0] and not visited[r][c]:
                    return False
        return True


class HitoriPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "hitori"
        self.size = size
        self.constraints = [
            ConstraintHitoriNoRepeat(),
            ConstraintHitoriAdjacent(),
            ConstraintHitoriConnected(),
        ]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Renzoku
# =============================================================================


class ConstraintAdjacency(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacency"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        hints = game_state.get("hints")
        if not hints:
            return True

        if len(hints.get("row", [])) < size:
            hints["row"] = [["0" for _ in range(size - 1)] for _ in range(size)]
        if len(hints.get("col", [])) < size - 1:
            hints["col"] = [["0" for _ in range(size)] for _ in range(size - 1)]

        # Check row adjacency
        for row in range(size):
            for col in range(size - 1):
                if hints["row"][row][col] == "1":
                    val1 = board[row][col]
                    val2 = board[row][col + 1]
                    if val1 == 0 or val2 == 0:
                        continue
                    if abs(val1 - val2) != 1:
                        return False

        # Check col adjacency
        for row in range(size - 1):
            for col in range(size):
                if hints["col"][row][col] == "1":
                    val1 = board[row][col]
                    val2 = board[row + 1][col]
                    if val1 == 0 or val2 == 0:
                        continue
                    if abs(val1 - val2) != 1:
                        return False
        return True


class RenzokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "renzoku"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintAdjacency(),
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Thermometers
# =============================================================================


class ConstraintThermometerFill(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_fill"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        thermometers = game_state.get("clues", {}).get("thermometers", [])
        thermometer_positions = {(r, c) for therm in thermometers for r, c in therm}

        for i in range(len(board)):
            for j in range(len(board[i])):
                if (i, j) not in thermometer_positions and board[i][j] == "s":
                    return False

        for thermometer in thermometers:
            first_empty = -1
            for i, (r, c) in enumerate(thermometer):
                if board[r][c] == "e":
                    first_empty = i
                    break
            if first_empty != -1:
                for i, (r, c) in enumerate(thermometer):
                    if i > first_empty and board[r][c] == "s":
                        return False
        return True


class ConstraintThermometerCount(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_count"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True
        size = len(board)
        row_counts = clues.get("row_counts")
        col_counts = clues.get("col_counts")

        if row_counts is None or col_counts is None:
            return True

        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)
            if 0 not in board[i]:
                if row_selected != row_counts[i]:
                    return False
            else:
                if row_selected > row_counts[i]:
                    return False
                if row_selected + row_undefined < row_counts[i]:
                    return False

        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 for i in range(size)):
                if col_selected != col_counts[j]:
                    return False
            else:
                if col_selected > col_counts[j]:
                    return False
                if col_selected + col_undefined < col_counts[j]:
                    return False
        return True


class ThermometersPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "thermometers"
        self.size = size
        self.constraints = [ConstraintThermometerFill(), ConstraintThermometerCount()]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Star Battle
# =============================================================================


class ConstraintRowStar(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_star"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        for row in board:
            if 0 not in row:
                if sum(1 for cell in row if cell == "s") != 1:
                    return False
            else:
                if sum(1 for cell in row if cell == "s") > 1:
                    return False
        return True


class ConstraintColStar(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_star"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for col in range(size):
            col_values = [board[row][col] for row in range(size)]
            if 0 not in col_values:
                if sum(1 for val in col_values if val == "s") != 1:
                    return False
            else:
                if sum(1 for val in col_values if val == "s") > 1:
                    return False
        return True


class ConstraintRegionStar(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_region_star"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        regions = game_state["regions"]
        size = len(board)
        region_counts = {}
        for i in range(size):
            for j in range(size):
                if board[i][j] == "s":
                    region = regions[i][j]
                    region_counts[region] = region_counts.get(region, 0) + 1
                    if region_counts[region] > 1:
                        return False
        for region_num in set(cell for row in regions for cell in row):
            region_cells = [
                (i, j)
                for i in range(size)
                for j in range(size)
                if regions[i][j] == region_num
            ]
            if all(board[i][j] != 0 for i, j in region_cells):
                if region_counts.get(region_num, 0) != 1:
                    return False
        return True


class ConstraintAdjacentStar(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacent_star"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for row in range(size):
            for col in range(size):
                if board[row][col] == "s":
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            new_row, new_col = row + dr, col + dc
                            if (
                                0 <= new_row < size
                                and 0 <= new_col < size
                                and board[new_row][new_col] == "s"
                            ):
                                return False
        return True


class StarBattlePuzzleFactory(PuzzleFactory):
    def __init__(self, size: int, num_stars: int = 1) -> None:
        super().__init__()
        self.game_name = "starbattle"
        self.size = size
        self.constraints = [
            ConstraintRowStar(),
            ConstraintColStar(),
            ConstraintAdjacentStar(),
            ConstraintRegionStar(),
        ]
        self.all_possible_values = ["s", "e"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        board = game_state["board"]
        if board[row][col] in ["s", "e"]:
            return []
        possible = []
        for val in ["s", "e"]:
            board[row][col] = val
            if self.check(game_state):
                possible.append(val)
            board[row][col] = 0
        return possible


# =============================================================================
# Trees and Tents
# =============================================================================


class ConstraintRowTents(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_tents"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True
        for i, row in enumerate(board):
            if 0 not in row:
                if row.count("tt") != clues["row_clues"][i]:
                    return False
            else:
                if row.count("tt") > clues["row_clues"][i]:
                    return False
        return True


class ConstraintColTents(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_tents"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True
        size = len(board)
        for j in range(size):
            col = [board[i][j] for i in range(size)]
            if 0 not in col:
                if col.count("tt") != clues["col_clues"][j]:
                    return False
            else:
                if col.count("tt") > clues["col_clues"][j]:
                    return False
        return True


class ConstraintTentTree(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_tent_tree"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tt":
                    adjacent_trees = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board[ni][nj] == "tr":
                                adjacent_trees.append((ni, nj))
                    if len(adjacent_trees) != 1:
                        return False
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tr":
                    adjacent_tents = 0
                    adjacent_non_allocated = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board[ni][nj] == "tt":
                                adjacent_tents += 1
                            elif board[ni][nj] == 0:
                                adjacent_non_allocated += 1
                    if adjacent_tents > 1:
                        return False
                    if adjacent_tents == 0 and adjacent_non_allocated == 0:
                        return False
        return True


class ConstraintAdjacentTents(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacent_tents"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tt":
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size:
                                if board[ni][nj] == "tt":
                                    return False
        return True


class ConstraintTentTreeCount(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_tent_tree_count"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        num_trees = sum(row.count("tr") for row in board)
        num_tents = sum(row.count("tt") for row in board)
        num_unallocated = sum(row.count(0) for row in board)
        if num_unallocated == 0:
            return num_tents == num_trees
        return (num_tents + num_unallocated) >= num_trees


class TreesAndTentsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "treesandtents"
        self.size = size
        self.constraints = [
            ConstraintRowTents(),
            ConstraintColTents(),
            ConstraintTentTree(),
            ConstraintAdjacentTents(),
            ConstraintTentTreeCount(),
        ]
        self.all_possible_values = ["tt", "e"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        board = game_state["board"]
        if board[row][col] != 0:
            return []
        possible = []
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible.append(value)
        board[row][col] = original_value
        return possible
