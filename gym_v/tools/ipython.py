"""IPython tool for code execution in isolated kernel."""

from __future__ import annotations

import ast
import base64
from io import BytesIO

from PIL import Image

from gym_v.core import Observation
from gym_v.tools.base import Tool


class IPythonTool(Tool):
    """IPython code execution tool with isolated kernel process.

    Executes Python code in a separate Jupyter kernel process for safety.
    Automatically captures matplotlib figures and includes them in the observation.

    Attributes:
        name: "ipython"
        description: Tool description for LLM prompts.
    """

    name = "ipython"
    description = (
        "Execute Python code in an isolated IPython kernel. "
        "Returns stdout, return values, and any matplotlib figures. "
        "Variables persist across calls within the same episode."
    )

    def __init__(self, timeout: float = 30.0):
        """Initialize IPython tool.

        Args:
            timeout: Code execution timeout in seconds.
        """
        self._timeout = timeout
        self._km = None  # KernelManager
        self._kc = None  # KernelClient

    def _ensure_kernel(self):
        """Ensure the kernel is started."""
        if self._km is None:
            from jupyter_client import KernelManager

            self._km = KernelManager(kernel_name="python3")
            self._km.start_kernel()
            self._kc = self._km.client()
            self._kc.start_channels()
            self._kc.wait_for_ready(timeout=60)
            # Pre-import matplotlib with non-interactive backend
            self._kc.execute(
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt"
            )
            # Wait for initialization to complete
            self._wait_for_idle()

    def _wait_for_idle(self):
        """Wait for kernel to become idle."""
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=self._timeout)
                if (
                    msg["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    break
            except Exception:
                break

    def execute(self, code: str) -> Observation:
        """Execute Python code in the isolated kernel.

        Args:
            code: Python code to execute.

        Returns:
            Observation with:
            - text: stdout, return value, or error traceback
            - image: list of matplotlib figures (if any)
        """
        self._ensure_kernel()

        # Execute the code
        self._kc.execute(code)

        text_parts = []
        images = []

        # Collect outputs from iopub channel
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=self._timeout)
            except Exception:
                break

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                # stdout/stderr
                text_parts.append(content["text"])

            elif msg_type == "execute_result":
                # Return value
                text_parts.append(content["data"].get("text/plain", ""))

            elif msg_type == "error":
                # Exception traceback
                text_parts.append("\n".join(content["traceback"]))

            elif msg_type == "display_data":
                # Rich output - capture images
                if "image/png" in content["data"]:
                    img_data = base64.b64decode(content["data"]["image/png"])
                    img = Image.open(BytesIO(img_data))
                    images.append(img)

            elif msg_type == "status" and content["execution_state"] == "idle":
                break

        # Capture any undisplayed matplotlib figures
        images.extend(self._capture_figures())

        text = "".join(text_parts).strip() or "(no output)"

        return Observation(
            text=text,
            image=images if images else None,
        )

    def _capture_figures(self) -> list[Image.Image]:
        """Capture all unclosed matplotlib figures."""
        capture_code = """
import io as _io
import base64 as _base64
_captured_figs = []
for _fig_num in plt.get_fignums():
    _buf = _io.BytesIO()
    plt.figure(_fig_num).savefig(_buf, format='png', bbox_inches='tight')
    _buf.seek(0)
    _captured_figs.append(_base64.b64encode(_buf.read()).decode())
plt.close('all')
_captured_figs
"""
        self._kc.execute(capture_code)

        images = []
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=self._timeout)
            except Exception:
                break

            if msg["msg_type"] == "execute_result":
                # Parse the list of base64-encoded figures
                try:
                    fig_list = ast.literal_eval(msg["content"]["data"]["text/plain"])
                    for b64_data in fig_list:
                        img_data = base64.b64decode(b64_data)
                        images.append(Image.open(BytesIO(img_data)))
                except Exception:
                    pass

            if (
                msg["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                break

        return images

    def reset(self) -> None:
        """Restart the kernel to clear all state."""
        if self._km is not None:
            self._km.restart_kernel()
            self._kc.wait_for_ready(timeout=60)
            # Re-initialize matplotlib
            self._kc.execute(
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt"
            )
            self._wait_for_idle()

    def close(self) -> None:
        """Shutdown the kernel and release resources."""
        if self._km is not None:
            self._kc.stop_channels()
            self._km.shutdown_kernel()
            self._km = None
            self._kc = None
