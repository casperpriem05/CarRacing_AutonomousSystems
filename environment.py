"""Observation preprocessing wrapper for CarRacing-v3.

Converts the raw 96x96x3 RGB frame into a dictionary of structured features
that rule-based agents can reason about:
  - track geometry (center offset at near/mid/far distances)
  - curve direction and sharpness derived from track-center trajectory
  - estimated speed
  - on-track flag

Detection strategy: grass (green) is the easiest colour to isolate, so we
define "road" as everything that is NOT grass AND has minimum brightness
(to exclude the black viewport border).  Track center at each row is the
midpoint of the leftmost and rightmost road pixels.

The car is at a fixed screen position because the camera follows it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import gymnasium as gym
import numpy as np

import config

PLAYFIELD_ROWS = 84
WARMUP_FRAMES = 50

CAR_ROW = 66
CAR_COL = 48


def make_env(render: bool = False, seed: Optional[int] = None) -> gym.Env:
    """Instantiate CarRacing-v3 with discrete actions."""
    mode = "human" if render else None
    return gym.make(
        config.ENV_ID,
        continuous=False,
        render_mode=mode,
        max_episode_steps=config.MAX_EPISODE_STEPS,
    )


def _grass_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Binary mask (0/255) where grass pixels are white."""
    r = frame_rgb[:, :, 0].astype(np.int16)
    g = frame_rgb[:, :, 1].astype(np.int16)
    b = frame_rgb[:, :, 2].astype(np.int16)
    grass = (g > 80) & (g > r + 15) & (g > b + 15)
    mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    mask[grass] = 255
    mask[PLAYFIELD_ROWS:, :] = 0
    return mask


def _road_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Binary mask where road pixels are white.

    Road = not grass AND has minimum brightness (excludes black border).
    """
    grass = _grass_mask(frame_rgb)
    brightness = np.min(frame_rgb[:, :, :3], axis=2)
    road = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    road[:PLAYFIELD_ROWS, :] = 255
    road[grass > 0] = 0
    road[brightness < 40] = 0
    return road


def _estimate_speed(frame_rgb: np.ndarray, prev_frame: Optional[np.ndarray]) -> float:
    """Estimate speed from the HUD bar and frame differencing."""
    bar = frame_rgb[87:95, 12:77, :]
    g = bar[:, :, 1].astype(np.int16)
    r = bar[:, :, 0].astype(np.int16)
    b = bar[:, :, 2].astype(np.int16)
    green_cols = np.any((g > 150) & (g > r + 40) & (g > b + 40), axis=0)
    speed_bar = float(np.sum(green_cols)) / max(bar.shape[1], 1)

    speed_diff = 0.0
    if prev_frame is not None:
        gray_cur = cv2.cvtColor(frame_rgb[:PLAYFIELD_ROWS], cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(prev_frame[:PLAYFIELD_ROWS], cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray_cur, gray_prev)
        speed_diff = float(np.mean(diff)) / 8.0

    return float(np.clip(max(speed_bar, speed_diff), 0.0, 1.0))


def _track_center_at_row(road_mask: np.ndarray, row: int) -> Optional[float]:
    """Return the horizontal center of the road at the given row, or None."""
    if row < 0 or row >= road_mask.shape[0]:
        return None
    cols = np.where(road_mask[row, :] > 0)[0]
    if len(cols) < 3:
        return None
    return float((cols[0] + cols[-1]) / 2.0)


def _analyse_track_ahead(
    road_mask: np.ndarray,
    car_col: int,
) -> Dict[str, float]:
    """Measure where the track center is at near / mid / far distances.

    Returns offsets normalised to roughly [-1, 1], where negative means the
    road center is to the left of the car and positive means right.
    """
    w = road_mask.shape[1]
    half_w = w / 2.0

    distances = [10, 25, 40]
    offsets = []
    for d in distances:
        row = CAR_ROW - d
        center = _track_center_at_row(road_mask, row)
        if center is not None:
            offsets.append((center - car_col) / half_w)
        else:
            offsets.append(0.0)

    off_near, off_mid, off_far = offsets

    avg_offset = (off_near + off_mid + off_far) / 3.0
    if abs(avg_offset) < 0.04:
        curve_direction = 0.0
    else:
        curve_direction = 1.0 if avg_offset > 0 else -1.0

    curve_sharpness = float(np.clip(abs(off_far - off_near) / 0.5, 0.0, 1.0))

    track_left = max(0.0, -avg_offset * 2 + 0.5)
    track_right = max(0.0, avg_offset * 2 + 0.5)
    track_center_val = 1.0 - abs(avg_offset) * 2

    return {
        "track_left": float(np.clip(track_left, 0, 1)),
        "track_center": float(np.clip(track_center_val, 0, 1)),
        "track_right": float(np.clip(track_right, 0, 1)),
        "curve_direction": curve_direction,
        "curve_sharpness": curve_sharpness,
        "offset_near": off_near,
        "offset_mid": off_mid,
        "offset_far": off_far,
    }


class ObservationProcessor:
    """Stateful processor that converts raw RGB frames into feature dicts."""

    def __init__(self) -> None:
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_idx: int = 0

    def reset(self) -> None:
        self._prev_frame = None
        self._frame_idx = 0

    def process(self, frame: np.ndarray, debug: bool = False) -> Dict[str, Any]:
        """Extract structured features from a single 96x96x3 RGB frame.

        Args:
            frame: Raw observation from CarRacing-v3 (uint8, 96x96x3).
            debug: If True, print feature values every 50 frames.
        """
        frame_uint8 = frame.astype(np.uint8) if frame.dtype != np.uint8 else frame

        road = _road_mask(frame_uint8)
        ahead = _analyse_track_ahead(road, CAR_COL)
        speed = _estimate_speed(frame_uint8, self._prev_frame)

        grass = _grass_mask(frame_uint8)
        margin = 6
        r0 = max(CAR_ROW - margin, 0)
        r1 = min(CAR_ROW + margin + 1, PLAYFIELD_ROWS)
        c0 = max(CAR_COL - margin, 0)
        c1 = min(CAR_COL + margin + 1, frame_uint8.shape[1])
        car_grass = grass[r0:r1, c0:c1]
        on_track = bool(np.mean(car_grass) < 60)

        self._prev_frame = frame_uint8.copy()
        self._frame_idx += 1

        features: Dict[str, Any] = {
            **ahead,
            "speed": speed,
            "on_track": on_track,
            "car_row": CAR_ROW,
            "car_col": CAR_COL,
            "warmup": self._frame_idx <= WARMUP_FRAMES,
        }

        if debug and self._frame_idx % 50 == 0:
            print(
                f"  [debug] frame={self._frame_idx} speed={speed:.3f} "
                f"on_track={on_track} off_n={ahead['offset_near']:+.3f} "
                f"off_m={ahead['offset_mid']:+.3f} off_f={ahead['offset_far']:+.3f} "
                f"sharpness={ahead['curve_sharpness']:.3f}"
            )

        return features
