"""Agent 5 -- Rain Rider.

Motorcycle metaphor: a skilled wet-weather rider who adapts aggression
based on recent performance -- backs off when things go wrong, pushes
harder when confident.

Architecture: adaptive aggression scaling.

    - Tracks a rolling 'confidence' score based on recent rewards
    - High confidence -> tighter lines, shorter cooldowns, more gas
    - Low confidence -> wider margins, longer cooldowns, more caution
    - Emergency recovery when off-track uses braking + directional correction

Key advantages:
    * Self-tuning -- automatically finds the right aggression level
    * Recovers from mistakes by braking then committing to multi-frame correction
    * Builds speed progressively as confidence grows
    * Uses all three offset distances for smooth trajectory
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np

from .base_agent import BaseAgent

DO_NOTHING = 0
LEFT = 1
RIGHT = 2
GAS = 3
BRAKE = 4


class RainRider(BaseAgent):
    """Adaptive-aggression rider -- backs off in trouble, pushes when safe."""

    def __init__(self) -> None:
        super().__init__(name="Rain Rider")
        self._confidence: float = 0.5  # 0 = cautious, 1 = aggressive
        self._reward_window: deque[float] = deque(maxlen=30)
        self._last_actions: deque[int] = deque(maxlen=6)
        self._steer_cooldown: int = 0
        self._recovery_frames: int = 0
        self._recovery_dir: int = DO_NOTHING

    def reset(self) -> None:
        self._confidence = 0.5
        self._reward_window.clear()
        self._last_actions.clear()
        self._steer_cooldown = 0
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

    def _update_confidence(self, on_track: bool) -> None:
        """Adjust confidence based on recent performance."""
        if len(self._reward_window) >= 10:
            recent_avg = np.mean(list(self._reward_window)[-10:])
            if recent_avg > 0:
                self._confidence = min(1.0, self._confidence + 0.02)
            elif recent_avg < -0.5:
                self._confidence = max(0.1, self._confidence - 0.02)

        if not on_track:
            self._confidence = max(0.1, self._confidence - 0.04)
        elif self._confidence < 0.5:
            self._confidence = min(1.0, self._confidence + 0.003)

    def act(self, features: Dict[str, Any]) -> int:
        if features.get("warmup", False):
            return self._pick(GAS)

        if self._steer_cooldown > 0:
            self._steer_cooldown -= 1

        speed: float = features["speed"]
        on_track: bool = features["on_track"]
        off_near: float = features.get("offset_near", 0.0)
        off_mid: float = features.get("offset_mid", 0.0)
        off_far: float = features.get("offset_far", 0.0)
        curvature = abs(off_far - off_near)

        self._update_confidence(on_track)

        # Adaptive thresholds
        steer_thresh = 0.08 - self._confidence * 0.04   # 0.04 to 0.08
        brake_speed = 0.45 - self._confidence * 0.10     # 0.35 to 0.45

        # Recovery mode
        if self._recovery_frames > 0:
            self._recovery_frames -= 1
            if on_track and abs(off_near) < 0.04:
                self._recovery_frames = 0
            else:
                return self._pick(self._recovery_dir)

        # Off-track emergency
        if not on_track:
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

        # Braking
        if curvature > 0.18 and speed > brake_speed:
            return self._pick(BRAKE)
        if abs(off_far) > 0.22 and speed > brake_speed:
            return self._pick(BRAKE)

        # Steering
        if self._steer_cooldown == 0:
            # Blend near and mid for smooth corrections
            blended = off_near * 0.7 + off_mid * 0.3

            if abs(blended) > steer_thresh:
                steer = self._steer_toward(blended)
                if not self._would_oscillate(steer):
                    # Proportional cooldown: bigger offset = shorter cooldown
                    if abs(blended) > 0.15:
                        cooldown = 1
                    else:
                        cooldown = max(1, int(3 - self._confidence * 1.5))
                    self._steer_cooldown = cooldown
                    return self._pick(steer)

            # Anticipatory: mid offset alone
            if abs(off_mid) > 0.10 and abs(off_near) < 0.05:
                steer = self._steer_toward(off_mid)
                if not self._would_oscillate(steer):
                    self._steer_cooldown = max(1, int(3 - self._confidence * 1.5))
                    return self._pick(steer)

        # Speed management
        # Straight boost: gas hard when road is straight
        if curvature < 0.05 and abs(off_near) < 0.08:
            return self._pick(GAS)

        if speed > 0.40 and curvature > 0.08:
            return self._pick(DO_NOTHING)

        return self._pick(GAS)

    def observe_reward(self, reward: float) -> None:
        self._reward_window.append(reward)
