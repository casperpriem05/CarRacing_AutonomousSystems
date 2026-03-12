"""Abstract base class for all CarRacing-v3 rule-based agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Interface that every agent must implement.

    Attributes:
        name: Human-readable identifier shown in logs and plots.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def act(self, features: Dict[str, Any]) -> int:
        """Choose a discrete action given preprocessed features.

        Args:
            features: Dictionary produced by
                :class:`environment.ObservationProcessor`.

        Returns:
            An integer action in ``{0, 1, 2, 3, 4}``
            (do-nothing / left / right / gas / brake).
        """

    def reset(self) -> None:
        """Called at the start of each episode to clear internal state."""
