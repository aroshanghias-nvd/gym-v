"""Symmetry Fill environment with icon shapes (original Sphinx style)."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.sphinx.utils import (
    compose_symmetry_fill_8_options,
    generate_extra_distractors,
    generate_symmetric_2x2_icons,
)

logger = get_logger()


class SphinxSymmetryFillPolyEnv(Env):
    """Symmetry Fill task with icon shapes (original Sphinx style).

    Given a 2x2 grid of icons with one missing cell, identify which of 8 options
    completes the grid to satisfy vertical + horizontal mirror symmetry.

    This version generates geometric icon shapes similar to the original
    Sphinx dataset.

    Args:
        cell_size: Pixel size of each cell/icon
        line_width: Width of icon lines
        option_size: Size of each option image in pixels
        padding: Padding between elements in the composed image
    """

    def __init__(
        self,
        cell_size: int = 200,
        line_width: int = 4,
        option_size: int = 200,
        padding: int = 15,
        style: str | None = None,
        difficulty: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the environment.

        Args:
            cell_size: Pixel size of each cell/icon
            line_width: Width of icon lines
            option_size: Size of each option image in pixels
            padding: Padding between elements in the composed image
            style: Visual style from ICON_STYLES ('simple', 'colored', 'nested',
                   'complex'), or 'random' for random selection each reset.
                   If None, uses difficulty to select style.
            difficulty: Difficulty level 1-4. Higher = more complex icons.
                       Only used if style is None.
        """
        super().__init__(**kwargs)
        self._cell_size = cell_size
        self._line_width = line_width
        self._option_size = option_size
        self._padding = padding
        self._style = style
        self._difficulty = difficulty

        self._cells: list[Image.Image] | None = None
        self._hidden_idx: int | None = None
        self._question: Image.Image | None = None
        self._correct_idx: int | None = None
        self._composed_image: Image.Image | None = None
        self._oracle_answer: str | None = None
        self._current_style: str | None = None

    @property
    def description(self) -> str:
        """Return task description for Symmetry Fill."""
        return dedent("""
            Look at the 2x2 grid of icons on the left side of the image.
            One cell is blank (shown in black).

            Your task: Identify which option (a)-(h) should fill the blank cell
            so that the completed grid exhibits vertical + horizontal mirror symmetry.

            In a grid with V+H symmetry:
            - Left and right halves are horizontal mirrors
            - Top and bottom halves are vertical mirrors

            Answer format: A single letter in parentheses, e.g., (a), (b), ..., (h)
        """).strip()

    def _compose_question_grid(
        self,
        cells: list[Image.Image],
        hidden_idx: int,
        gap: int = 4,
    ) -> Image.Image:
        """Compose a 2x2 question grid with one cell hidden (black).

        Args:
            cells: List of 4 cell images [TL, TR, BL, BR]
            hidden_idx: Index of the cell to hide (0-3)
            gap: Gap between cells in pixels

        Returns:
            Composed 2x2 grid image with one black cell
        """
        cell_w, cell_h = cells[0].size

        # Create canvas for 2x2 grid
        total_w = 2 * cell_w + gap
        total_h = 2 * cell_h + gap
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

        # Position mapping: [TL, TR, BL, BR]
        positions = [
            (0, 0),  # TL
            (cell_w + gap, 0),  # TR
            (0, cell_h + gap),  # BL
            (cell_w + gap, cell_h + gap),  # BR
        ]

        for i, (cell, pos) in enumerate(zip(cells, positions, strict=True)):
            if i == hidden_idx:
                # Draw black rectangle for hidden cell
                black_cell = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
                canvas.paste(black_cell, pos)
            else:
                canvas.paste(cell, pos)

        return canvas

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Generate symmetric 2x2 icon grid with style/difficulty
        self._cells, self._hidden_idx = generate_symmetric_2x2_icons(
            self.np_random,
            cell_size=self._cell_size,
            line_width=self._line_width,
            style=self._style,
            difficulty=self._difficulty,
        )

        # Track the selected style for metadata
        if self._style == "random" or (
            self._style is None and self._difficulty is None
        ):
            self._current_style = "random"
        elif self._style is not None:
            self._current_style = self._style
        else:
            self._current_style = f"difficulty_{self._difficulty}"

        # The correct answer is the hidden cell
        correct_cell = self._cells[self._hidden_idx]

        # Compose question grid (with hidden cell as black)
        self._question = self._compose_question_grid(self._cells, self._hidden_idx)

        # Generate 8 options:
        # - Option 0: correct answer
        # - Options 1-3: other cells from the grid (as partial distractors)
        # - Options 4-7: generated distractors via transformations

        # Get the other 3 cells as base distractors
        other_cells = [self._cells[i] for i in range(4) if i != self._hidden_idx]

        # Generate 4 extra distractors by transforming the correct answer
        extra_distractors = generate_extra_distractors(
            correct_cell,
            self._cells,  # All 4 cells for comparison
            self.np_random,
            num_extra=4,
        )

        # Combine: correct + 3 other cells + 4 generated distractors = 8 options
        all_options = [correct_cell] + other_cells + extra_distractors

        # Shuffle all 8 options and track correct answer position
        indices = list(range(8))
        self.np_random.shuffle(indices)

        shuffled_options = [all_options[i] for i in indices]
        # Find where the correct option (index 0) ended up
        self._correct_idx = indices.index(0)

        # Compose the final image
        self._composed_image = compose_symmetry_fill_8_options(
            self._question,
            shuffled_options,
            self._correct_idx,
            option_size=self._option_size,
            padding=self._padding,
        )

        # Set oracle answer
        labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        self._oracle_answer = labels[self._correct_idx]

        # Build observation text
        obs_text = (
            "Which option (a)-(h) completes the left 2×2 grid "
            "with vertical + horizontal mirror symmetry?"
        )

        logger.info(
            f"Reset Sphinx SymmetryFillPoly: hidden_idx={self._hidden_idx}, "
            f"answer={self._oracle_answer}, style={self._current_style}"
        )

        obs = Observation(
            image=self.render(),
            text=obs_text,
            metadata={
                "hidden_idx": self._hidden_idx,
                "style": self._current_style,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "hidden_idx": self._hidden_idx,
            "style": self._current_style,
        }
        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Evaluate the action against the correct answer.

        Args:
            action: The user's answer, e.g., "(a)", "a", "(h)"

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Normalize action
        action_clean = action.strip().lower()
        if not action_clean.startswith("("):
            action_clean = f"({action_clean})"
        if not action_clean.endswith(")"):
            action_clean = action_clean + ")"

        # Check if answer is correct
        correct = action_clean == self._oracle_answer.lower()
        reward = 1.0 if correct else 0.0

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "user_answer": action,
                "correct": correct,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "hidden_idx": self._hidden_idx,
            "correct": correct,
        }

        return obs, reward, True, False, info

    def render(self) -> Image.Image:
        """Return the composed image with 8 options."""
        if self._composed_image is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self._composed_image
