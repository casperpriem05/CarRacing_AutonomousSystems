"""Baseline agent — uniformly random action selection.

Serves as the performance floor against which rule-based agents are compared.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Picks a uniformly random discrete action every step."""

    def __init__(self) -> None:
        super().__init__(name="Random Baseline")

    def act(self, features: Dict[str, Any]) -> int:  # noqa: ARG002
        return int(np.random.randint(0, 5))
