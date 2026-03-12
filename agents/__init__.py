"""Agent package — all rule-based agents for CarRacing-v3."""

from .baseline_random import RandomAgent
from .agent_cautious import CautiousCruiser
from .agent_apex import ApexRacer
from .agent_traction import TractionManager
from .agent_superbike import MotoGP
from .agent_rain import RainRider
from .agent_line_hunter import LineHunter

ALL_AGENTS = [
    RandomAgent,
    CautiousCruiser,
    ApexRacer,
    TractionManager,
    MotoGP,
    RainRider,
    LineHunter,
]

__all__ = [
    "RandomAgent",
    "CautiousCruiser",
    "ApexRacer",
    "TractionManager",
    "MotoGP",
    "RainRider",
    "LineHunter",
    "ALL_AGENTS",
]
