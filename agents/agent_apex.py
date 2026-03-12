"""Agent 2 -- Apex Racer.

Motorcycle metaphor: an experienced track rider who follows the racing line
using the classic "slow in, fast out" cornering technique.

Priority order: Speed > Completion > Safety.

Uses far-offset for braking decisions and near-offset for in-corner steering.
Phases: STRAIGHT -> BRAKING -> TURNING -> ACCELERATING.
Lower steering thresholds and shorter cooldowns for responsive cornering.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Any, Dict

from .base_agent import BaseAgent

DO_NOTHING = 0
LEFT = 1
RIGHT = 2
GAS = 3
BRAKE = 4


class _Phase(Enum):
    STRAIGHT = auto()
    BRAKING = auto()
    TURNING = auto()
    ACCELERATING = auto()


class ApexRacer(BaseAgent):
    """Racing-line rider -- brakes late, accelerates hard out of corners."""

    def __init__(self) -> None:
        super().__init__(name="Max Verstappen")
        self._phase = _Phase.STRAIGHT
        self._turn_steps = 0
        self._steer_cooldown = 0
        self._last_actions: deque[int] = deque(maxlen=4)
        self._recovery_frames: int = 0
        self._recovery_dir: int = DO_NOTHING

    def reset(self) -> None:
        self._phase = _Phase.STRAIGHT
        self._turn_steps = 0
        self._steer_cooldown = 0
        self._last_actions.clear()
        self._recovery_frames = 0
        self._recovery_dir = DO_NOTHING

    def _would_oscillate(self, action: int) -> bool:
        if len(self._last_actions) < 2:
            return False
        prev = self._last_actions[-1]
        return (action == LEFT and prev == RIGHT) or (action == RIGHT and prev == LEFT)

    def _pick(self, action: int) -> int:
        self._last_actions.append(action)
        return action

    def _steer_toward(self, offset: float) -> int:
        return RIGHT if offset < 0 else LEFT

    def act(self, features: Dict[str, Any]) -> int:
        if features.get("warmup", False):
            return self._pick(GAS)

        if self._steer_cooldown > 0:
            self._steer_cooldown -= 1

        speed = features["speed"]
        on_track = features["on_track"]
        off_near = features.get("offset_near", 0.0)
        off_mid = features.get("offset_mid", 0.0)
        off_far = features.get("offset_far", 0.0)
        sharpness = features.get("curve_sharpness", 0.0)
        curvature = abs(off_far - off_near)

        # Recovery mode
        if self._recovery_frames > 0:
            self._recovery_frames -= 1
            if on_track and abs(off_near) < 0.04:
                self._recovery_frames = 0
            else:
                return self._pick(self._recovery_dir)

        # Off-track emergency
        if not on_track:
            self._phase = _Phase.STRAIGHT
            # Brake first if fast to prevent overshoot
            if speed > 0.30:
                return self._pick(BRAKE)
            # Multi-frame recovery commitment
            if abs(off_near) > 0.03:
                self._recovery_dir = self._steer_toward(off_near)
                self._recovery_frames = 4
                return self._pick(self._recovery_dir)
            if abs(off_mid) > 0.03:
                self._recovery_dir = self._steer_toward(off_mid)
                self._recovery_frames = 3
                return self._pick(self._recovery_dir)
            return self._pick(GAS)

        if self._phase == _Phase.STRAIGHT:
            # Steer to stay on line even on straights
            if abs(off_near) > 0.05 and self._steer_cooldown == 0:
                steer = self._steer_toward(off_near)
                if not self._would_oscillate(steer):
                    # Proportional cooldown
                    self._steer_cooldown = 1 if abs(off_near) > 0.10 else 2
                    return self._pick(steer)

            # Use curvature as primary brake trigger
            if curvature > 0.18 and speed > 0.30:
                self._phase = _Phase.BRAKING
            elif abs(off_far) > 0.22 and speed > 0.30:
                self._phase = _Phase.BRAKING

            # Straight boost: gas hard when road is clearly straight
            return self._pick(GAS)

        if self._phase == _Phase.BRAKING:
            target = max(0.10, 0.30 - sharpness * 0.3)
            if speed <= target:
                self._phase = _Phase.TURNING
                self._turn_steps = 0
            return self._pick(BRAKE)

        if self._phase == _Phase.TURNING:
            self._turn_steps += 1
            if curvature < 0.06 and self._turn_steps > 2:
                self._phase = _Phase.ACCELERATING
                return self._pick(GAS)

            if abs(off_near) > 0.04 and self._steer_cooldown == 0:
                steer = self._steer_toward(off_near)
                if not self._would_oscillate(steer):
                    # Proportional: bigger offset = shorter cooldown for faster correction
                    self._steer_cooldown = 1 if abs(off_near) > 0.10 else max(1, int(speed * 2))
                    return self._pick(steer)
            return self._pick(GAS)

        if self._phase == _Phase.ACCELERATING:
            if curvature < 0.04:
                self._phase = _Phase.STRAIGHT
            # Still steer while accelerating
            if abs(off_near) > 0.05 and self._steer_cooldown == 0:
                steer = self._steer_toward(off_near)
                if not self._would_oscillate(steer):
                    self._steer_cooldown = 1 if abs(off_near) > 0.10 else 2
                    return self._pick(steer)
            return self._pick(GAS)

        return self._pick(GAS)
