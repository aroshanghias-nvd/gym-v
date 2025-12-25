"""Knight Swap single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class ReasoningGymKnightSwapEnv(Env):
    """Knight Swap puzzle using reasoning-gym's Knight Swap dataset.

    The player must swap white and black knights' positions through valid knight moves.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the board in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 64,
        padding: int = 24,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._cell_px = cell_px
        self._padding = padding

        self._seed: int | None = None
        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._board: dict[str, list[str]] | None = None
        self._pieces: dict[str, str | None] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description with current puzzle parameters.

        Original reasoning-gym question format:
        ```
        Knight Swap Challenge:

        ```
            A   B   C   D
           ----------------
        3 |   | . |   | . |
           ----------------
        2 | B | w |   |   |
           ----------------
        1 |   |   | B | w |
           ----------------
        ```

        Legend:
        - 'w' = White Knight
        - 'B' = Black Knight
        - Empty squares are marked with '.'

        Objective:
        Swap the positions of all white knights with all black knights through valid moves.

        Rules:
        1. Knights move in L-shape (2 squares + 1 square perpendicular)
        2. Knights can only move to empty squares
        3. w moves first, then players alternate
        4. All knights must reach their target positions (white ↔ black)

        Question:
        Is it possible to swap all knights' positions? If yes, list the moves.

        Answer Format:
        - For impossible puzzles: "No"
        - For possible puzzles: List moves as ["color,from,to", ...]
          Example: ["w,A1,B3"] means white knight moves A1→B3
        ```

        Original reasoning-gym answer format:
        - For impossible puzzles: "No"
        - For possible puzzles: '["w,A1,C2", "B,D1,B2", ...]'
          (JSON array of move strings)
        """
        start_turn = self._metadata.get("start_turn", "w") if self._metadata else "w"
        start_color = "White" if start_turn == "w" else "Black"
        return dedent(f"""
            Knight Swap Challenge:

            In the image:
            - White knights are shown in white/light color
            - Black knights are shown in black/dark color
            - Light/green squares are valid positions (knights can move here)
            - Dark gray squares are invalid positions (knights cannot move here)
            - Empty valid squares have no knight piece

            Objective:
            Swap the positions of all white knights with all black knights through valid moves.

            Rules:
            1. Knights move in L-shape (2 squares + 1 square perpendicular)
            2. Knights can only move to empty squares
            3. {start_color} ('{start_turn}') moves first, then players alternate
            4. All knights must reach their target positions (white ↔ black)

            Question:
            Is it possible to swap all knights' positions? If yes, list the moves.

            Answer Format:
            - For impossible puzzles: "No"
            - For possible puzzles: List moves as a JSON array ["color,from,to", ...]
              where color is 'w' for white or 'B' for black
              Example: ["w,A1,C2", "B,D1,B2"] means white knight A1→C2, then black knight D1→B2
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if seed is not None and "seed" not in kwargs:
            kwargs["seed"] = seed
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("knight_swap", **kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        self._dataset = self._make_dataset(seed=self._seed)
        self._entry_idx = int(self.np_random.integers(0, len(self._dataset)))
        self._entry = self._dataset[self._entry_idx]

        self._oracle_answer = self._entry["answer"]
        self._metadata = self._entry.get("metadata", {})
        self._board = self._metadata.get("board", {})
        self._pieces = self._metadata.get("pieces", {})

        logger.info("Reset ReasoningGym Knight Swap.")

        # obs.text = only the board state (caption)
        board_text = self._format_board(self._board, self._pieces)

        obs = Observation(
            image=self.render(),
            text=board_text,
            metadata=self._metadata,
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        answer = action
        reward = self._dataset.score_answer(answer=answer, entry=self._entry)

        obs = Observation(
            image=self.render(),
            text=None,
            metadata=self._metadata,
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }

        return obs, reward, True, False, info

    def _format_board(
        self, board: dict[str, list[str]], pieces: dict[str, str | None]
    ) -> str:
        """Format the board state as a string."""
        if not board:
            return ""

        positions = list(board.keys())
        columns = sorted(set(pos[0] for pos in positions))
        rows = sorted(set(int(pos[1:]) for pos in positions), reverse=True)

        lines = []
        # Header
        lines.append("    " + "   ".join(columns))
        lines.append("   " + "----" * len(columns))

        # Board rows
        for row in rows:
            line = f"{row} |"
            for col in columns:
                pos = col + str(row)
                if pos in pieces:
                    piece = pieces[pos] if pieces[pos] is not None else "."
                    line += f" {piece} |"
                else:
                    line += "   |"
            lines.append(line)
            lines.append("   " + "----" * len(columns))

        return "\n".join(lines)

    def render(self) -> Image.Image:
        return self._render_knight_board(
            self._board,
            self._pieces,
            cell_px=self._cell_px,
            padding=self._padding,
        )

    def _render_knight_board(
        self,
        board: dict[str, list[str]],
        pieces: dict[str, str | None],
        cell_px: int = 64,
        padding: int = 24,
        light_sq: tuple[int, int, int] = (238, 238, 210),
        dark_sq: tuple[int, int, int] = (118, 150, 86),
        white_knight_color: tuple[int, int, int] = (255, 255, 255),
        black_knight_color: tuple[int, int, int] = (30, 30, 30),
    ) -> Image.Image:
        if not board:
            return Image.new("RGB", (200, 200), (250, 250, 250))

        positions = list(board.keys())
        columns = sorted(set(pos[0] for pos in positions))
        rows = sorted(set(int(pos[1:]) for pos in positions), reverse=True)

        n_cols = len(columns)
        n_rows = len(rows)

        # Extra margin for coordinate labels on left and bottom
        label_margin = int(cell_px * 0.5)
        board_width = cell_px * n_cols
        board_height = cell_px * n_rows
        width = padding * 2 + label_margin + board_width
        height = padding * 2 + label_margin + board_height

        bg_color = (48, 46, 43)
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        font_path = self.assets_dir / "DejaVuSans.ttf"
        if font_path.exists():
            knight_font = ImageFont.truetype(str(font_path), int(cell_px * 0.65))
            label_font = ImageFont.truetype(str(font_path), int(cell_px * 0.28))
        else:
            logger.warning(f"Font file not found: {font_path}, using default font")
            knight_font = ImageFont.load_default()
            label_font = knight_font

        # Board origin (top-left of the chess grid)
        board_x0 = padding + label_margin
        board_y0 = padding

        # Draw row labels on the left (numbers: 1, 2, 3, ...)
        for r_idx, row in enumerate(rows):
            label = str(row)
            bbox = draw.textbbox((0, 0), label, font=label_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx = padding + (label_margin - tw) // 2
            ly = board_y0 + r_idx * cell_px + (cell_px - th) // 2
            draw.text((lx, ly), label, fill=(200, 200, 200), font=label_font)

        # Draw column labels at the bottom (letters: A, B, C, ...)
        for c_idx, col in enumerate(columns):
            label = col
            bbox = draw.textbbox((0, 0), label, font=label_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx = board_x0 + c_idx * cell_px + (cell_px - tw) // 2
            ly = board_y0 + board_height + (label_margin - th) // 2
            draw.text((lx, ly), label, fill=(200, 200, 200), font=label_font)

        # Draw board squares and pieces
        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(columns):
                pos = col + str(row)
                x = board_x0 + c_idx * cell_px
                y = board_y0 + r_idx * cell_px

                # Check if position is valid in the board
                if pos not in board:
                    # Draw as invalid/non-existent cell (grayed out)
                    draw.rectangle(
                        [x, y, x + cell_px - 1, y + cell_px - 1],
                        fill=(80, 80, 80),
                        outline=(60, 60, 60),
                        width=1,
                    )
                    continue

                # Alternating square colors (chess.com style)
                sq_color = light_sq if (r_idx + c_idx) % 2 == 0 else dark_sq
                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=sq_color,
                )

                # Draw piece if present
                piece = pieces.get(pos)
                if piece == "w":
                    self._draw_knight(
                        draw,
                        x,
                        y,
                        cell_px,
                        white_knight_color,
                        knight_font,
                        outline=(0, 0, 0),
                    )
                elif piece == "B":
                    self._draw_knight(
                        draw,
                        x,
                        y,
                        cell_px,
                        black_knight_color,
                        knight_font,
                        outline=(180, 180, 180),
                    )

        # Draw board border
        draw.rectangle(
            [
                board_x0 - 2,
                board_y0 - 2,
                board_x0 + board_width + 1,
                board_y0 + board_height + 1,
            ],
            outline=(80, 80, 80),
            width=2,
        )

        return img

    def _draw_knight(
        self,
        draw: ImageDraw.Draw,
        x: int,
        y: int,
        cell_px: int,
        color: tuple[int, int, int],
        font: ImageFont.FreeTypeFont,
        outline: tuple[int, int, int] = (0, 0, 0),
    ):
        """Draw a knight symbol (♞) at the specified cell."""
        knight_symbol = "♞"
        bbox = draw.textbbox((0, 0), knight_symbol, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = x + cell_px // 2
        cy = y + cell_px // 2
        # Draw outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text(
                        (cx - tw // 2 + dx, cy - th // 2 + dy),
                        knight_symbol,
                        fill=outline,
                        font=font,
                    )
        # Draw knight
        draw.text((cx - tw // 2, cy - th // 2), knight_symbol, fill=color, font=font)
