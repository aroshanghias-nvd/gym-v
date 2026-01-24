"""Roundtable Assignment environment for gym-v (self-contained)."""

from __future__ import annotations

import math
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVERoundtableAssignmentEnv(Env):
    """RLVE Roundtable Assignment as a single-turn environment.

    Given M groups of people and N tables, assign each person to a table such that:
    - No table contains more than one person from the same group
    - No table exceeds its total capacity
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""There are {M} groups of people and {N} tables.
- The i-th group consists of R[i] people. Array R: {R}
- The j-th table can seat up to C[j] people. Array C: {C}

You need to assign each person to a table such that:
- No table contains more than one person from the same group.
- No table exceeds its total capacity.

**Output Format:** Output {M} lines. The i-th line (0-indexed) should contain R[i] integers (separated by spaces), representing the table indices assigned to each person in the i-th group."""

    def __init__(
        self,
        max_n_m: int = 5,
        wrong_format: float = -1.0,
        invalid_solution: float = -0.5,
        rewarding_strategy: str = "(satisfied/all)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 5.0,
        cell_px: int = 100,
        padding: int = 40,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        if max_n_m < 3:
            raise ValueError("max_n_m must be >= 3")

        self._m: int | None = None
        self._n: int | None = None
        self._r: list[int] | None = None
        self._c: list[int] | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._m and self._n:
            size_hint = f"{self._m} groups, {self._n} tables"
        else:
            size_hint = "M groups, N tables"

        return dedent(
            f"""
            Roundtable Assignment Problem:

            Given {size_hint}, assign each person to a table such that:
            1. No table contains more than one person from the same group
            2. No table exceeds its total capacity

            In the visualization:
            - The image shows a circular seating arrangement (top-down view)
            - Each colored circle represents a different group
            - Group sizes are shown with labels (e.g., "Group 0: 3 people")
            - Tables are shown as circular positions around the center
            - Table capacities are labeled (e.g., "Table 0: cap=4")
            - Connections show the constraint that each group must spread across different tables
            - Color coding helps distinguish between groups

            Legend:
            - Each color represents a unique group
            - Numbers indicate group indices and table indices
            - Capacity values show the maximum number of people each table can seat

            Output format: M lines, where line i contains R[i] space-separated integers
            representing the table index assigned to each person in group i.
            """
        ).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        obs = Observation(
            image=self._last_image,
            text=self._prompt,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
        }
        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]
        reward = float(self._score_answer(action_str))
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
        }

        terminated = True
        truncated = False

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: terminated for agent_id in self._agent_ids},
                "__all__": terminated,
            },
            {
                **{agent_id: truncated for agent_id in self._agent_ids},
                "__all__": truncated,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def _generate(self) -> None:
        """Generate a roundtable assignment problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        MAX_N_M = self._max_n_m
        if MAX_N_M < 3:
            raise ValueError("MAX_N_M must be >= 3")

        M = int(self.np_random.integers(2, MAX_N_M + 1))
        R = []
        tables = [[] for _ in range(MAX_N_M)]
        for group_index in range(M):
            R.append(int(self.np_random.integers(2, MAX_N_M + 1)))
            # Use self.np_random.choice instead of random.sample
            table_indices = self.np_random.choice(
                MAX_N_M, size=R[-1], replace=False
            ).tolist()
            for table_index in table_indices:
                tables[table_index].append(group_index)
        tables = [table for table in tables if len(table) > 0]

        if len(R) != M:
            raise RuntimeError("R should have length M")

        N = len(tables)
        C = [len(table) for table in tables]
        if len(C) != N:
            raise RuntimeError("C should have length N")

        reference_answer = [[] for _ in range(M)]
        for table_index, table in enumerate(tables):
            for group_index in table:
                reference_answer[group_index].append(table_index)

        if not all(len(answer) == R[group_index] for group_index, answer in enumerate(reference_answer)):
            raise RuntimeError("Reference answer does not match the group sizes")

        self._m = M
        self._n = N
        self._r = R
        self._c = C
        self._reference_answer = "\n".join(
            " ".join(map(str, answer)) for answer in reference_answer
        )

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._m is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            M=self._m,
            N=self._n,
            R=" ".join(f"R[{i}]={Ri}" for i, Ri in enumerate(self._r)),
            C=" ".join(f"C[{i}]={Ci}" for i, Ci in enumerate(self._c)),
        )

    def _process(self, answer: str | None) -> list[list[int]] | None:
        """Process the answer string into a 2D list of integers."""
        if answer is not None:
            answer = answer.strip()
            try:
                matrix = []
                for line in answer.splitlines():
                    line = line.strip()
                    if line:
                        matrix.append(list(map(int, line.split())))
                return matrix
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on the RLVE scoring strategy.

        Returns:
            wrong_format: if answer cannot be parsed
            invalid_solution: if answer has wrong format or violates constraints
            rewarding_weight * score: based on rewarding_strategy
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if not isinstance(processed_result, list):
                raise RuntimeError("processed_result should be a list")

            if len(processed_result) != self._m:
                return self._rewards["invalid_solution"]

            countings = [0] * self._n
            for answer_row, Ri in zip(processed_result, self._r):
                if len(answer_row) != Ri:
                    return self._rewards["invalid_solution"]
                if not all(0 <= i < self._n for i in answer_row):
                    return self._rewards["invalid_solution"]
                if len(set(answer_row)) != Ri:
                    return self._rewards["invalid_solution"]
                for table_index in answer_row:
                    countings[table_index] += 1

            if len(countings) != len(self._c) != self._n:
                raise RuntimeError("countings should match the number of tables")
            satisfied = sum(int(counting <= Ci) for counting, Ci in zip(countings, self._c))
            if satisfied > self._n:
                raise RuntimeError("satisfied should not exceed N")

            if self._rewards["rewarding_strategy"] == "(satisfied/all)^beta":
                return self._rewards["rewarding_weight"] * (
                    (satisfied / self._n) ** self._rewards["rewarding_beta"]
                )
            elif self._rewards["rewarding_strategy"] == "satisfied=all":
                return self._rewards["rewarding_weight"] * (satisfied == self._n)
            else:
                raise NotImplementedError(
                    f"Unknown rewarding strategy: {self._rewards['rewarding_strategy']}"
                )
        else:
            return self._rewards["wrong_format"]

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the roundtable assignment problem as an image.

        Shows:
        - Circular seating arrangement (top-down view)
        - Clear labels for each group and table
        - Visual indicators of group sizes and table capacities
        - Color-coded groups
        - Legend for symbols and colors
        """
        if self._m is None or self._n is None:
            raise RuntimeError("No problem generated")

        padding = self._padding
        center_radius = 200
        table_radius = 50
        group_radius = 30

        # Calculate dimensions
        width = padding * 2 + center_radius * 2 + table_radius * 2 + 300
        height = padding * 2 + center_radius * 2 + table_radius * 2 + 200

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, 24)
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Center point
        cx = width // 2 - 100
        cy = height // 2

        # Draw title
        title = "Roundtable Assignment Problem"
        title_bbox = draw.textbbox((0, 0), title, font=font_large)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(
            ((width - title_width) // 2, padding // 2),
            title,
            fill=(30, 30, 30),
            font=font_large,
        )

        # Color palette for groups
        group_colors = [
            (220, 100, 100),  # Red
            (100, 150, 220),  # Blue
            (120, 200, 120),  # Green
            (220, 180, 100),  # Orange
            (180, 120, 200),  # Purple
            (200, 200, 100),  # Yellow
            (100, 200, 200),  # Cyan
            (200, 150, 180),  # Pink
        ]

        # Draw tables in a circle
        table_positions = []
        for i in range(self._n):
            angle = 2 * math.pi * i / self._n - math.pi / 2
            tx = cx + center_radius * math.cos(angle)
            ty = cy + center_radius * math.sin(angle)
            table_positions.append((tx, ty))

            # Draw table circle
            draw.ellipse(
                [
                    tx - table_radius,
                    ty - table_radius,
                    tx + table_radius,
                    ty + table_radius,
                ],
                fill=(240, 240, 240),
                outline=(80, 80, 80),
                width=3,
            )

            # Draw table label
            table_label = f"T{i}"
            bbox = draw.textbbox((0, 0), table_label, font=font_medium)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (tx - tw // 2, ty - th // 2 - 10),
                table_label,
                fill=(30, 30, 30),
                font=font_medium,
            )

            # Draw capacity
            cap_label = f"cap={self._c[i]}"
            bbox = draw.textbbox((0, 0), cap_label, font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (tx - tw // 2, ty - th // 2 + 10),
                cap_label,
                fill=(100, 100, 100),
                font=font_small,
            )

        # Draw center circle (representing the problem)
        draw.ellipse(
            [
                cx - 60,
                cy - 60,
                cx + 60,
                cy + 60,
            ],
            fill=(200, 200, 200),
            outline=(100, 100, 100),
            width=2,
        )
        center_text = f"{self._m} Groups\n{self._n} Tables"
        lines = center_text.split("\n")
        y_offset = cy - 15
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw // 2, y_offset), line, fill=(50, 50, 50), font=font_small)
            y_offset += th + 2

        # Draw legend on the right side
        legend_x = width - 280
        legend_y = padding + 60
        draw.text(
            (legend_x, legend_y),
            "Groups:",
            fill=(30, 30, 30),
            font=font_medium,
        )
        legend_y += 30

        for i in range(self._m):
            color = group_colors[i % len(group_colors)]
            # Draw color circle
            draw.ellipse(
                [legend_x, legend_y, legend_x + 20, legend_y + 20],
                fill=color,
                outline=(50, 50, 50),
                width=1,
            )
            # Draw group info
            group_text = f"Group {i}: {self._r[i]} people"
            draw.text(
                (legend_x + 30, legend_y + 2),
                group_text,
                fill=(30, 30, 30),
                font=font_small,
            )
            legend_y += 28

        # Draw constraint visualization (connections between groups and tables)
        # Parse reference answer to show valid assignments
        ref_lines = self._reference_answer.strip().split("\n")
        for group_idx, line in enumerate(ref_lines):
            table_indices = list(map(int, line.split()))
            color = group_colors[group_idx % len(group_colors)]

            # Draw connections from center to each table this group uses
            for table_idx in table_indices:
                tx, ty = table_positions[table_idx]
                # Draw a light line from center to table
                draw.line(
                    [(cx, cy), (tx, ty)],
                    fill=(*color, 128),  # Semi-transparent
                    width=2,
                )

        # Draw tables legend
        legend_y += 20
        draw.text(
            (legend_x, legend_y),
            "Constraints:",
            fill=(30, 30, 30),
            font=font_medium,
        )
        legend_y += 25

        constraint_texts = [
            "• No table can have 2+ people",
            "  from the same group",
            "• Each table has a capacity",
            "  limit (shown on tables)",
        ]
        for text in constraint_texts:
            draw.text(
                (legend_x, legend_y),
                text,
                fill=(60, 60, 60),
                font=font_small,
            )
            legend_y += 18

        return img
