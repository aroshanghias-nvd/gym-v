"""Light Up single-turn environment backed by VGRP-Bench."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger
from gym_v.envs.vgrp.puzzle_generator import generate_puzzle

from .vgrp_factories import LightUpPuzzleFactory

logger = get_logger()


class VGRPLightUpEnv(Env):
    """Light Up puzzle using VGRP-Bench's LightUp puzzle generator.

    Place light bulbs to illuminate the entire grid.

    Args:
        size: Grid size (default 5)
        num_hints: Not used
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        size: int = 5,
        num_hints: int = 0,
        cell_px: int = 60,
        padding: int = 30,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._size = size
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._factory = LightUpPuzzleFactory(size)
        self._puzzle_board: list[list[str]] | None = None
        self._solution_board: list[list[str]] | None = None
        self._wall_numbers: list[list[int]] | None = (
            None  # -1 for no number, 0-4 for number
        )

    @property
    def description(self) -> str:
        """Return description for Light Up puzzle."""
        return dedent(f"""
            Solve this {self._size}x{self._size} Light Up puzzle.

            In the image:
            - Black cells are walls (some may have numbers)
            - White cells need to be illuminated
            - Numbers on walls indicate how many bulbs must be adjacent (orthogonally)

            Rules:
            1. Place light bulbs to illuminate all white cells
            2. Light bulbs illuminate their entire row and column until blocked by a wall
            3. Light bulbs cannot illuminate each other
            4. Numbered walls must have exactly that many adjacent bulbs

            Output format: A {self._size}x{self._size} grid where:
            - 's' = light bulb
            - 'e' = empty white cell
            - 'w' = wall (given, include in output)
            separated by spaces within rows, and newlines separating rows.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Generate walls and wall numbers
        self._wall_numbers = self._generate_walls()

        # Generate solution
        result = generate_puzzle(
            self._factory, self._size, 0, wall_numbers=self._wall_numbers
        )

        if result is None:
            raise RuntimeError(
                f"Failed to generate LightUp puzzle with size {self._size}"
            )

        _, self._solution_board = result

        # Ensure walls are marked in solution
        for i in range(self._size):
            for j in range(self._size):
                if self._wall_numbers[i][j] >= 0:
                    self._solution_board[i][j] = "w"

        # Puzzle board shows walls
        self._puzzle_board = [
            ["e" if self._wall_numbers[i][j] == -1 else "w" for j in range(self._size)]
            for i in range(self._size)
        ]

        logger.info("Reset VGRP Light Up.")

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_with_walls(),
            metadata={"size": self._size},
        )
        info = {
            "oracle_answer": self._board_to_text(self._solution_board),
        }
        return obs, info

    def _generate_walls(self) -> list[list[int]]:
        """Generate random wall configuration with numbers.

        Returns:
            2D array where -1 = empty cell, 0-4 = numbered wall, 5 = unnumbered wall
        """
        walls = [[-1 for _ in range(self._size)] for _ in range(self._size)]

        # Place some walls (20-30% of cells)
        num_walls = int(self._size * self._size * 0.25)

        for _ in range(num_walls):
            r = np.random.randint(0, self._size)
            c = np.random.randint(0, self._size)

            if walls[r][c] == -1:
                # 50% chance of numbered wall, 50% unnumbered
                if np.random.random() < 0.5:
                    walls[r][c] = np.random.randint(0, 5)  # 0-4
                else:
                    walls[r][c] = -1  # Will be treated as wall without number

        return walls

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        try:
            answer_board = self._text_to_board(action)
            reward = 1.0 if self._check_solution(answer_board) else 0.0
        except Exception as e:
            logger.warning(f"Failed to parse answer: {e}")
            reward = 0.0

        obs = Observation(
            image=self.render(),
            text=self._board_to_text_with_walls(),
            metadata={"size": self._size},
        )
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _check_solution(self, answer_board: list[list[str]]) -> bool:
        """Check if the answer matches the solution."""
        if len(answer_board) != self._size:
            return False
        for i in range(self._size):
            if len(answer_board[i]) != self._size:
                return False
            for j in range(self._size):
                if answer_board[i][j] != self._solution_board[i][j]:
                    return False
        return True

    def _board_to_text_with_walls(self) -> str:
        """Convert board to text showing walls."""
        lines = []
        lines.append("Grid (w=wall, numbers on walls show required adjacent bulbs):")
        for i, row in enumerate(self._puzzle_board):
            row_str = []
            for j, cell in enumerate(row):
                if cell == "w" and self._wall_numbers[i][j] >= 0:
                    row_str.append(str(self._wall_numbers[i][j]))
                else:
                    row_str.append(cell)
            lines.append(" ".join(row_str))
        return "\n".join(lines)

    def _board_to_text(self, board: list[list[str]]) -> str:
        """Convert board to text."""
        lines = []
        for row in board:
            lines.append(" ".join(row))
        return "\n".join(lines)

    def _text_to_board(self, text: str) -> list[list[str]]:
        """Parse text to board."""
        lines = text.strip().split("\n")
        board = []
        for line in lines:
            line = line.strip()
            if not line or ":" in line or "Grid" in line:
                continue
            row = []
            for val in line.split():
                val = val.strip().lower()
                # Handle numbers as walls
                if val.isdigit():
                    row.append("w")
                elif val in ["s", "e", "w"]:
                    row.append(val)
                else:
                    row.append("e")
            if row:
                board.append(row)
        return board

    def render(self) -> Image.Image:
        return self._render_lightup(
            self._puzzle_board,
            self._wall_numbers,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_lightup(
        self,
        puzzle: list[list[str]],
        wall_numbers: list[list[int]],
        cell_px: int = 60,
        padding: int = 30,
        bg: tuple[int, int, int] = (40, 40, 50),  # Dark background (unlit room)
        fg: tuple[int, int, int] = (20, 20, 20),
        wall_color: tuple[int, int, int] = (60, 60, 70),
        grid: tuple[int, int, int] = (100, 100, 110),
    ) -> Image.Image:
        n = self._size
        size = padding * 2 + cell_px * n
        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            font = ImageFont.truetype(
                str(font_path), int(cell_px * 0.6)
            )  # Larger font for wall numbers
        else:
            font = ImageFont.load_default()

        # First pass: Draw base cells and light effects
        for r in range(n):
            for c in range(n):
                x = padding + c * cell_px
                y = padding + r * cell_px

                if puzzle[r][c] == "w":
                    # Check if wall has a number
                    has_number = wall_numbers[r][c] >= 0

                    if has_number:
                        # Numbered wall - make it stand out with different color
                        # Shadow
                        draw.rectangle(
                            [x + 3, y + 3, x + cell_px - 1, y + cell_px - 1],
                            fill=(50, 40, 30),
                        )
                        # Main wall (darker brown for numbered walls)
                        draw.rectangle(
                            [x + 1, y + 1, x + cell_px - 3, y + cell_px - 3],
                            fill=(70, 60, 50),
                        )
                        # Highlight border to make it obvious
                        draw.rectangle(
                            [x + 1, y + 1, x + cell_px - 3, y + cell_px - 3],
                            outline=(100, 80, 60),
                            width=3,
                        )
                    else:
                        # Unnumbered wall - solid black
                        # Shadow
                        draw.rectangle(
                            [x + 3, y + 3, x + cell_px - 1, y + cell_px - 1],
                            fill=(30, 30, 35),
                        )
                        # Main wall (very dark)
                        draw.rectangle(
                            [x + 1, y + 1, x + cell_px - 3, y + cell_px - 3],
                            fill=(45, 45, 55),
                        )

                        # Brick pattern for unnumbered walls
                        brick_rows = 3
                        brick_h = (cell_px - 4) // brick_rows
                        for i in range(brick_rows):
                            y_brick = y + 2 + i * brick_h
                            # Horizontal line
                            draw.line(
                                [(x + 2, y_brick), (x + cell_px - 4, y_brick)],
                                fill=(60, 60, 70),
                                width=1,
                            )

                    # Draw number if present (BIG and BRIGHT)
                    if has_number:
                        num_str = str(wall_numbers[r][c])
                        cx = x + cell_px // 2
                        cy = y + cell_px // 2

                        # Draw circular background for number
                        circle_r = cell_px // 3
                        draw.ellipse(
                            [
                                cx - circle_r,
                                cy - circle_r,
                                cx + circle_r,
                                cy + circle_r,
                            ],
                            fill=(255, 200, 100),
                            outline=(200, 150, 50),
                            width=3,
                        )

                        # Draw number (centered, bold, dark)
                        draw.text(
                            (cx, cy), num_str, fill=(40, 30, 20), font=font, anchor="mm"
                        )
                else:
                    # Empty cell - dark room
                    draw.rectangle(
                        [x + 1, y + 1, x + cell_px - 1, y + cell_px - 1],
                        fill=(50, 50, 60),
                    )

        # Second pass: Draw light bulb placeholders and socket indicators
        for r in range(n):
            for c in range(n):
                x = padding + c * cell_px
                y = padding + r * cell_px

                # For empty cells, draw ghost light bulb placeholder
                if puzzle[r][c] == "e":
                    cx = x + cell_px // 2
                    cy = y + cell_px // 2
                    bulb_r = cell_px // 4

                    # Draw translucent bulb shape 💡
                    # Bulb glass (circle)
                    draw.ellipse(
                        [
                            cx - bulb_r,
                            cy - bulb_r - bulb_r // 3,
                            cx + bulb_r,
                            cy + bulb_r - bulb_r // 3,
                        ],
                        fill=(100, 100, 110),
                        outline=(140, 140, 150),
                        width=2,
                    )

                    # Bulb base/socket (trapezoid shape)
                    base_h = bulb_r // 2
                    points = [
                        (cx - bulb_r // 2, cy + bulb_r - bulb_r // 3),
                        (cx + bulb_r // 2, cy + bulb_r - bulb_r // 3),
                        (cx + bulb_r // 3, cy + bulb_r - bulb_r // 3 + base_h),
                        (cx - bulb_r // 3, cy + bulb_r - bulb_r // 3 + base_h),
                    ]
                    draw.polygon(points, fill=(90, 90, 100), outline=(120, 120, 130))

                    # Draw + symbol in center to indicate "place here"
                    plus_size = bulb_r // 3
                    draw.line(
                        [
                            (cx - plus_size, cy - bulb_r // 3),
                            (cx + plus_size, cy - bulb_r // 3),
                        ],
                        fill=(150, 150, 160),
                        width=2,
                    )
                    draw.line(
                        [
                            (cx, cy - bulb_r // 3 - plus_size),
                            (cx, cy - bulb_r // 3 + plus_size),
                        ],
                        fill=(150, 150, 160),
                        width=2,
                    )

        # Draw grid
        for i in range(n + 1):
            x = padding + i * cell_px
            y = padding + i * cell_px
            draw.line([(padding, y), (padding + n * cell_px, y)], fill=grid, width=2)
            draw.line([(x, padding), (x, padding + n * cell_px)], fill=grid, width=2)

        return img
