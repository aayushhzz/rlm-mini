"""rlm-mini: a small Recursive Language Model harness with a Continual Harness."""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


def home_dir(home=None):
    """Where runs/, harness_state/ and corpus/ live. RLM_HOME overrides the project dir."""
    return Path(home or os.environ.get("RLM_HOME") or PROJECT_DIR).resolve()
