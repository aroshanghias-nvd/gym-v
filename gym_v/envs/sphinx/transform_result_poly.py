"""Transform Result Identify environment with polygon shapes (original Sphinx style)."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from PIL import Image

from gym_v import Env, Observation, get_logger
from gym_v.envs.sphinx.utils import (
    TRANSFORMS,
    apply_transform,
    compose_8_options,
    generate_random_polygon,
)

logger = get_logger()

# Human-readable descriptions for each transformation
TRANSFORM_DESCRIPTIONS = {
    "identity": "no transformation",
    "rot90_cw": "rotate 90° clockwise",
    "rot180": "rotate 180°",
    "rot90_ccw": "rotate 90° counterclockwise",
    "flip_h": "reflect across a vertical line",
    "flip_v": "reflect across a horizontal line",
    "flip_diag": "reflect across the main diagonal",
    "flip_antidiag": "reflect across the anti-diagonal",
}


class SphinxTransformResultPolyEnv(Env):
    """Transform Result Identify task with polygon shapes (original Sphinx style).

    Given an original polygon shape and a transformation description, identify which
    of 8 options shows the correct transformation result.

    This version generates irregular polygon shapes on a grid background,
    similar to the original Sphinx dataset.

    Args:
        img_size: Size of each shape image in pixels
        num_points: Number of vertices in the polygon (controls complexity)
        line_width: Width of polygon lines
        grid_divisions: Number of grid divisions in background
        option_size: Size of each option image in the composed output
        padding: Padding between elements in the composed image
    """

    def __init__(
        self,
        img_size: int = 300,
        num_points: int = 8,
        line_width: int = 3,
        grid_divisions: int = 8,
        option_size: int = 280,
        padding: int = 20,
        style: str | None = None,
        difficulty: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the environment.

        Args:
            img_size: Size of each shape image in pixels
            num_points: Number of vertices in the polygon (controls complexity)
            line_width: Width of polygon lines
            grid_divisions: Number of grid divisions in background
            option_size: Size of each option image in the composed output
            padding: Padding between elements in the composed image
            style: Visual style from POLY_STYLES ('outline', 'filled', 'nested',
                   'striped', 'gradient', '3d', 'composite', 'pixelated'),
                   or 'random' for random selection each reset.
                   If None, uses difficulty to select style.
            difficulty: Difficulty level 1-8 (maps to styles in order).
                       Higher = more complex visual style. Only used if style is None.
        """
        super().__init__(**kwargs)
        self._img_size = img_size
        self._num_points = num_points
        self._line_width = line_width
        self._grid_divisions = grid_divisions
        self._option_size = option_size
        self._padding = padding
        self._style = style
        self._difficulty = difficulty

        self._original: Image.Image | None = None
        self._correct_transform: str | None = None
        self._correct_idx: int | None = None
        self._composed_image: Image.Image | None = None
        self._oracle_answer: str | None = None
        self._current_style: str | None = None

    @property
    def description(self) -> str:
        """Return task description for Transform Result Identify."""
        return dedent("""
            Look at the original polygon shape at the top of the image.
            A geometric transformation has been applied to create one of the 8 options below.

            Your task: Identify which option (a)-(h) shows the correct transformation result.

            The transformation could be:
            - Rotation: 90° clockwise, 180°, or 90° counterclockwise
            - Reflection: across horizontal, vertical, main diagonal, or anti-diagonal axis
            - Identity: no change

            Answer format: A single letter in parentheses, e.g., (a), (b), ..., (h)
        """).strip()

    def _build_problem_text(self) -> str:
        """Build the problem text based on the correct transformation."""
        desc = TRANSFORM_DESCRIPTIONS[self._correct_transform]
        return (
            f"After performing {desc} on the top figure, "
            f"which option (a)–(h) shows the correct outcome?"
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Generate random polygon as the original shape with style/difficulty
        self._original = generate_random_polygon(
            self.np_random,
            img_size=self._img_size,
            num_points=self._num_points,
            line_width=self._line_width,
            grid_lines=True,
            grid_divisions=self._grid_divisions,
            style=self._style,
            difficulty=self._difficulty,
        )

        # Track the selected style for metadata
        if self._style == "random" or (
            self._style is None and self._difficulty is None
        ):
            # Style was randomly selected, we can infer it from the difficulty logic
            self._current_style = "random"
        elif self._style is not None:
            self._current_style = self._style
        else:
            self._current_style = f"difficulty_{self._difficulty}"

        # Randomly select a transformation as the correct answer
        transform_idx = int(self.np_random.integers(0, len(TRANSFORMS)))
        self._correct_transform = TRANSFORMS[transform_idx]

        # Generate 8 options with all transformations
        options_list, self._correct_idx = self._generate_options()

        # Compose the final image
        self._composed_image = compose_8_options(
            self._original,
            options_list,
            self._correct_idx,
            option_size=self._option_size,
            padding=self._padding,
        )

        # Set oracle answer
        labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
        self._oracle_answer = labels[self._correct_idx]

        # Build observation text
        obs_text = self._build_problem_text()

        logger.info(
            f"Reset Sphinx TransformResultPoly: transform={self._correct_transform}, "
            f"answer={self._oracle_answer}, num_points={self._num_points}, "
            f"style={self._current_style}"
        )

        obs = Observation(
            image=self.render(),
            text=obs_text,
            metadata={
                "transform": self._correct_transform,
                "num_points": self._num_points,
                "style": self._current_style,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "transform": self._correct_transform,
            "style": self._current_style,
        }
        return obs, info

    def _generate_options(self) -> tuple[list[Image.Image], int]:
        """Generate 8 option images with all transformations.

        Returns:
            Tuple of (list of 8 option images, index of correct answer)
        """
        # Generate all 8 transformations
        transformed = {t: apply_transform(self._original, t) for t in TRANSFORMS}

        # Create shuffled list
        transform_order = list(TRANSFORMS)
        self.np_random.shuffle(transform_order)

        options = [transformed[t] for t in transform_order]
        correct_idx = transform_order.index(self._correct_transform)

        return options, correct_idx

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
                "transform": self._correct_transform,
                "user_answer": action,
                "correct": correct,
            },
        )
        info = {
            "oracle_answer": self._oracle_answer,
            "transform": self._correct_transform,
            "correct": correct,
        }

        return obs, reward, True, False, info

    def render(self) -> Image.Image:
        """Return the composed image with 8 options."""
        if self._composed_image is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self._composed_image
