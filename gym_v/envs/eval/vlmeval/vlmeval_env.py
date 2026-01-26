import os
import tempfile

from PIL import Image

from gym_v.core import Env, Observation
from gym_v.logger import get_logger
from vlmeval.dataset import build_dataset
from vlmeval.smp import dump

logger = get_logger()


class VLMEvalEnv(Env):
    """
    An environment that wraps VLMEvalKit datasets.

    On reset, it loads the dataset and creates an agent for each item.
    On step, it collects predictions, runs VLMEvalKit evaluation, and returns rewards.
    """

    def __init__(
        self,
        dataset_name: str = "MMBench_DEV_EN",
        judge_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset = build_dataset(dataset_name)
        self.data = self.dataset.data

        # Default judge kwargs
        if judge_kwargs is None:
            self.judge_kwargs = {"model": "exact_matching"}
        else:
            self.judge_kwargs = judge_kwargs

        # Determine agent IDs based on dataset size
        # We use the index column as the agent identifier suffix
        self.indices = self.data["index"].tolist()
        self._agent_ids = {f"agent_{i}" for i in self.indices}

        # Create index to question mapping for evaluation
        self.index2question = {
            row["index"]: row.get("question", "") for _, row in self.data.iterrows()
        }

        self.predictions = {}

    @property
    def description(self) -> str:
        return f"VLMEvalEnv wrapping {self.dataset_name} with {len(self.data)} samples."

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.predictions = {}

        obs_dict = {}
        info_dict = {}

        # We can optionally shuffle or select a subset using options if needed
        # But per instructions: "reset sends out a whole eval item"

        for i in range(len(self.data)):
            item = self.data.iloc[i]
            idx = item["index"]
            agent_id = f"agent_{idx}"

            # Use VLMEvalKit to build the prompt structure
            # This returns a list of dicts with 'type' (image/text) and 'value'
            struct = self.dataset.build_prompt(item)

            images = []
            text_parts = []

            for msg in struct:
                if msg["type"] == "image":
                    img_path = msg["value"]
                    try:
                        img = Image.open(img_path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        logger.error(f"Failed to load image {img_path}: {e}")
                elif msg["type"] == "text":
                    text_parts.append(msg["value"])

            # Combine text parts
            full_text = "\n".join(text_parts)

            # Select the first image if available, or list of images if supported
            # gym-v Observation supports list[Image.Image]
            obs_img = images if images else None
            if obs_img and len(obs_img) == 1:
                obs_img = obs_img[0]

            obs_dict[agent_id] = Observation(
                image=obs_img, text=full_text, metadata={"struct": struct, "index": idx}
            )

            info_dict[agent_id] = {
                "index": idx,
                "answer": item.get("answer"),
                "question": item.get("question"),
            }

        return obs_dict, info_dict

    def step(self, action_dict):
        # action_dict maps agent_id -> prediction string

        # Create a copy of the original data to ensure all metadata columns (A, B, C, answer, etc.) are present
        # VLMEvalKit's evaluate function often requires these columns to be present in the input file
        df_predictions = self.data.copy()

        # Create a map from index to prediction
        pred_map = {}
        for agent_id, pred in action_dict.items():
            idx = int(agent_id.split("_")[1])
            pred_map[idx] = pred

        # Add prediction column
        # Map predictions to the dataframe based on index
        df_predictions["prediction"] = df_predictions["index"].map(pred_map)

        # Use .xlsx suffix which is robustly handled by VLMEvalKit
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_file = tmp.name

        # Dump predictions to file
        dump(df_predictions, tmp_file)

        # Run evaluation
        judge_kwargs = self.judge_kwargs

        # This is the "copy logic" part - calling the original evaluate
        metrics = self.dataset.evaluate(tmp_file, **judge_kwargs)

        rewards = {f"agent_{k}": 0.0 for k in action_dict.keys()}
        # TODO: Parse metrics to populate rewards if needed (usually metrics is aggregate)

        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        # All done in one step
        terminated = {f"agent_{k}": True for k in action_dict.keys()}
        truncated = {f"agent_{k}": False for k in action_dict.keys()}
        terminated["__all__"] = True
        truncated["__all__"] = False

        return {}, rewards, terminated, truncated, {"metrics": metrics}
