"""Championship Race Game — Race against AI agent ghosts across 10 maps.

Controls  (simultaneous keys work):
    Arrow UP    = GAS
    Arrow DOWN  = BRAKE
    Arrow LEFT  = STEER LEFT
    Arrow RIGHT = STEER RIGHT
    (no key)    = COAST

Usage:
    python race_game.py                  # Run championship (generates ghosts if needed)
    python race_game.py --generate       # Only generate/refresh ghost data
    python race_game.py --maps 5         # Race on 5 maps instead of 10
    python race_game.py --seed 100       # Use different base seed
    python race_game.py --no-random      # Exclude the Random Baseline agent
    python race_game.py --refresh        # Delete & regenerate ghost data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np

try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    print("pygame is required.  pip install pygame")
    sys.exit(1)

import config
from agents import ALL_AGENTS
from environment import ObservationProcessor

GHOST_DIR = os.path.join(config.RESULTS_DIR, "ghosts")
NUM_MAPS = 10

RENDER_W, RENDER_H = 600, 400
VIEW_SCALE = 1.2
GAME_W = int(RENDER_W * VIEW_SCALE)   # 720
GAME_H = int(RENDER_H * VIEW_SCALE)   # 480
GAME_X, GAME_Y = 10, 60

HUD_X = GAME_X + GAME_W + 15          # 745
HUD_W = 340
WINDOW_W = HUD_X + HUD_W + 10         # ~1095
WINDOW_H = GAME_Y + GAME_H + 10       # ~550

CR_WINDOW_W, CR_WINDOW_H = 1000, 800
CR_SCALE = 6.0
CR_ZOOM = 2.7
CR_ZOOM_STEADY = CR_ZOOM * CR_SCALE   # 16.2

SX = RENDER_W / CR_WINDOW_W           # 0.6
SY = RENDER_H / CR_WINDOW_H           # 0.5

CAR_SCREEN_X = CR_WINDOW_W / 2 * SX   # 300
CAR_SCREEN_Y = (CR_WINDOW_H * 3 / 4) * SY   # 300

GAME_FPS = 50  # match env physics FPS

POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

ACTION_NAMES = {0: "COAST", 1: "LEFT", 2: "RIGHT", 3: "GAS", 4: "BRAKE"}

AGENT_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "Random Baseline": (140, 140, 140),
    "Cautious (My Sister)": (80, 130, 255),
    "Max Verstappen": (255, 200, 50),
    "Traction Focused": (50, 200, 200),
    "MotoGP": (255, 70, 70),
    "Rain Rider": (200, 80, 220),
    "Line Hunter": (80, 220, 80),
    "YOU": (255, 255, 255),
}

GHOST_CAR_POLY = [
    (0, -12),   # nose
    (-6, -4),
    (-6,  10),
    (-4,  12),
    ( 4,  12),
    ( 6,  10),
    ( 6, -4),
]


def ghost_path(agent_name: str, seed: int) -> str:
    safe = agent_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    return os.path.join(GHOST_DIR, f"{safe}_seed{seed}.json")


def load_ghosts(seed: int, exclude_random: bool = False) -> List[Dict[str, Any]]:
    ghosts = []
    for cls in ALL_AGENTS:
        agent = cls()
        if exclude_random and "Random" in agent.name:
            continue
        fp = ghost_path(agent.name, seed)
        if os.path.exists(fp):
            with open(fp) as f:
                ghosts.append(json.load(f))
    return ghosts


def ghosts_have_positions(seed: int) -> bool:
    """Check if ghost data contains world position fields."""
    for cls in ALL_AGENTS:
        fp = ghost_path(cls().name, seed)
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
            if data["steps"] and "car_x" in data["steps"][0]:
                return True
            return False
    return False


def generate_all_ghosts(base_seed: int, num_maps: int, exclude_random: bool = False) -> None:
    from game.generate_ghosts import main as _generate
    _generate()


def init_pygame() -> Tuple[pygame.Surface, Dict[str, pygame.font.Font]]:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("CarRacing Championship — Race the Ghosts!")
    fonts = {
        "big": pygame.font.SysFont("consolas", 24, bold=True),
        "med": pygame.font.SysFont("consolas", 16),
        "small": pygame.font.SysFont("consolas", 13),
        "title": pygame.font.SysFont("consolas", 32, bold=True),
        "countdown": pygame.font.SysFont("consolas", 52, bold=True),
    }
    return screen, fonts


def txt(surface, text, pos, font, colour=(255, 255, 255), bg=None):
    r = font.render(text, True, colour, bg)
    return surface.blit(r, pos)


def world_to_game_view(
    wx: float, wy: float,
    player_x: float, player_y: float, player_angle: float,
    zoom: float,
) -> Tuple[float, float]:
    """Convert a world point to game-view pixel coordinates."""
    dx = wx - player_x
    dy = wy - player_y
    angle = -player_angle
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    sx = CAR_SCREEN_X + rx * zoom * SX
    sy = CAR_SCREEN_Y - ry * zoom * SY
    return sx * VIEW_SCALE, sy * VIEW_SCALE


def _rotate_poly(poly, angle_rad):
    """Rotate a polygon (list of (x,y)) around origin by angle_rad."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [(x * c - y * s, x * s + y * c) for x, y in poly]


def draw_ghost_cars(
    game_surf: pygame.Surface,
    player_x: float, player_y: float, player_angle: float,
    zoom: float,
    ghosts: List[Dict[str, Any]],
    step_idx: int,
) -> None:
    """Draw semi-transparent ghost cars on the game surface."""
    for g in ghosts:
        steps = g["steps"]
        si = min(step_idx, len(steps) - 1)
        if si < 0:
            continue
        sd = steps[si]
        if "car_x" not in sd:
            continue  # old ghost format without positions

        gx, gy, ga = sd["car_x"], sd["car_y"], sd["car_angle"]
        sx, sy = world_to_game_view(gx, gy, player_x, player_y, player_angle, zoom)

        margin = 40
        if sx < -margin or sx > GAME_W + margin or sy < -margin or sy > GAME_H + margin:
            continue

        rel_angle = -(ga - player_angle)
        rotated = _rotate_poly(GHOST_CAR_POLY, rel_angle)
        translated = [(sx + x, sy + y) for x, y in rotated]

        colour = AGENT_COLOURS.get(g["agent"], (180, 180, 180))

        xs = [p[0] for p in translated]
        ys = [p[1] for p in translated]
        min_x, max_x = int(min(xs)) - 1, int(max(xs)) + 2
        min_y, max_y = int(min(ys)) - 1, int(max(ys)) + 2
        w = max_x - min_x
        h = max_y - min_y
        if w <= 0 or h <= 0:
            continue

        tmp = pygame.Surface((w, h), pygame.SRCALPHA)
        local_pts = [(int(x - min_x), int(y - min_y)) for x, y in translated]
        if len(local_pts) >= 3:
            alpha_colour = (*colour, 120)  # semi-transparent
            pygame.draw.polygon(tmp, alpha_colour, local_pts)
            outline_colour = (*colour, 200)
            pygame.draw.polygon(tmp, outline_colour, local_pts, 2)
        game_surf.blit(tmp, (min_x, min_y))

        name_short = g["agent"][:12]
        label = pygame.font.SysFont("consolas", 10).render(name_short, True, colour)
        label.set_alpha(160)
        game_surf.blit(label, (int(sx) - label.get_width() // 2, int(sy) + 14))


class PlayerInput:
    """Tracks keyboard state and produces smooth continuous actions."""

    # Discrete agents steer at ±0.6; continuous accepts ±1.0.
    # Cap to 0.6 so the player has the same steering strength as agents.
    STEER_MAX = 0.6

    def __init__(self):
        self.steer_target = 0.0
        self.steer_current = 0.0
        self.gas = 0.0
        self.brake = 0.0
        self.steer_speed = 4.0   # how fast steering responds (per second)
        self.steer_return = 6.0  # how fast steering centres (per second)

    def update(self, dt: float) -> np.ndarray:
        """Read current key state and return [steer, gas, brake] action."""
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            self.steer_target = -self.STEER_MAX
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            self.steer_target = +self.STEER_MAX
        else:
            self.steer_target = 0.0

        diff = self.steer_target - self.steer_current
        if abs(diff) < 0.01:
            self.steer_current = self.steer_target
        else:
            speed = self.steer_speed if self.steer_target != 0.0 else self.steer_return
            self.steer_current += diff * min(1.0, speed * dt)

        self.gas = 1.0 if keys[pygame.K_UP] else 0.0
        self.brake = 0.8 if keys[pygame.K_DOWN] else 0.0

        return np.array([self.steer_current, self.gas, self.brake], dtype=np.float32)

    def reset(self):
        self.steer_target = 0.0
        self.steer_current = 0.0
        self.gas = 0.0
        self.brake = 0.0

    @property
    def action_name(self) -> str:
        parts = []
        if self.gas > 0:
            parts.append("GAS")
        if self.brake > 0:
            parts.append("BRAKE")
        if self.steer_current < -0.15:
            parts.append("LEFT")
        elif self.steer_current > 0.15:
            parts.append("RIGHT")
        return "+".join(parts) if parts else "COAST"


def draw_hud(
    screen: pygame.Surface,
    fonts: Dict[str, pygame.font.Font],
    player_reward: float,
    player_step: int,
    ghosts: List[Dict[str, Any]],
    race_num: int,
    total_races: int,
    player_on_track: bool,
    player_speed: float,
    player_input: PlayerInput,
    championship: Dict[str, int],
) -> None:
    x0 = HUD_X
    pw = HUD_W - 10

    pygame.draw.rect(screen, (20, 20, 30), (HUD_X - 5, 0, HUD_W + 15, WINDOW_H))
    pygame.draw.line(screen, (50, 50, 70), (HUD_X - 6, 0), (HUD_X - 6, WINDOW_H), 2)

    y = 10
    txt(screen, f"RACE {race_num}/{total_races}", (x0, y), fonts["big"], (255, 220, 50))
    y += 28
    txt(screen, f"Step {player_step:4d} / {config.MAX_EPISODE_STEPS}", (x0, y), fonts["small"], (140, 140, 160))
    y += 20

    standings: List[Tuple[str, float, bool]] = [("YOU", player_reward, False)]
    for g in ghosts:
        si = min(player_step - 1, len(g["steps"]) - 1)
        gr = g["steps"][si]["total_reward"] if si >= 0 else 0.0
        standings.append((g["agent"], gr, player_step >= len(g["steps"])))
    standings.sort(key=lambda x: x[1], reverse=True)

    txt(screen, "LIVE STANDINGS", (x0, y), fonts["med"], (200, 200, 200))
    y += 20
    pygame.draw.line(screen, (50, 50, 70), (x0, y), (x0 + pw, y))
    y += 4

    for pos, (name, reward, finished) in enumerate(standings, 1):
        is_player = name == "YOU"
        col = AGENT_COLOURS.get(name, (180, 180, 180))
        if is_player:
            pygame.draw.rect(screen, (35, 35, 55), (HUD_X - 4, y - 1, HUD_W + 10, 19))
        pos_col = (255, 215, 0) if pos == 1 else (192, 192, 192) if pos == 2 else (205, 127, 50) if pos == 3 else (110, 110, 110)
        txt(screen, f"{pos:2d}.", (x0, y), fonts["small"], pos_col)
        dn = ">> YOU <<" if is_player else name[:15]
        txt(screen, dn, (x0 + 26, y), fonts["small"], col)
        rc = (80, 255, 80) if reward > 0 else (255, 80, 80)
        fin = " FIN" if finished else ""
        txt(screen, f"{reward:7.1f}{fin}", (x0 + pw - 80, y), fonts["small"], rc)
        y += 18

    y += 8
    pygame.draw.line(screen, (50, 50, 70), (x0, y), (x0 + pw, y))
    y += 8

    txt(screen, "YOUR STATUS", (x0, y), fonts["med"], (200, 200, 200))
    y += 20

    txt(screen, "Speed:", (x0, y), fonts["small"], (140, 140, 160))
    bx, bw = x0 + 55, pw - 60
    pygame.draw.rect(screen, (40, 40, 40), (bx, y + 2, bw, 11))
    sw = int(player_speed * bw)
    sc = (50, 255, 50) if player_speed < 0.6 else (255, 200, 50) if player_speed < 0.85 else (255, 80, 80)
    pygame.draw.rect(screen, sc, (bx, y + 2, sw, 11))
    y += 20

    act_name = player_input.action_name
    act_col = (50, 255, 50) if "GAS" in act_name else (255, 80, 80) if "BRAKE" in act_name else (255, 200, 50) if "LEFT" in act_name or "RIGHT" in act_name else (160, 160, 160)
    txt(screen, f"Action: {act_name}", (x0, y), fonts["small"], act_col)
    y += 18

    txt(screen, "Steer:", (x0, y), fonts["small"], (140, 140, 160))
    bar_cx = bx + bw // 2
    bar_half = bw // 2
    pygame.draw.rect(screen, (40, 40, 40), (bx, y + 2, bw, 8))
    pygame.draw.line(screen, (80, 80, 80), (bar_cx, y + 1), (bar_cx, y + 10))
    steer_px = int(player_input.steer_current * bar_half)
    if steer_px != 0:
        sx = min(bar_cx, bar_cx + steer_px)
        sw = abs(steer_px)
        pygame.draw.rect(screen, (255, 200, 50), (sx, y + 2, sw, 8))
    y += 18

    tc = (50, 255, 50) if player_on_track else (255, 50, 50)
    tt = "ON TRACK" if player_on_track else "OFF TRACK!"
    txt(screen, tt, (x0, y), fonts["small"], tc)
    y += 18

    rc = (80, 255, 80) if player_reward > 0 else (255, 80, 80)
    txt(screen, f"Reward: {player_reward:.1f}", (x0, y), fonts["med"], rc)
    y += 28

    pygame.draw.line(screen, (50, 50, 70), (x0, y), (x0 + pw, y))
    y += 6
    txt(screen, "CHAMPIONSHIP", (x0, y), fonts["med"], (255, 220, 50))
    y += 20
    champ = sorted(championship.items(), key=lambda x: x[1], reverse=True)
    for pos, (name, pts) in enumerate(champ[:8], 1):
        col = AGENT_COLOURS.get(name, (180, 180, 180))
        is_player = name == "YOU"
        if is_player:
            pygame.draw.rect(screen, (35, 35, 55), (HUD_X - 4, y - 1, HUD_W + 10, 17))
        dn = ">> YOU <<" if is_player else name[:13]
        txt(screen, f"{pos}. {dn}", (x0, y), fonts["small"], col)
        txt(screen, f"{pts:3d} pts", (x0 + pw - 52, y), fonts["small"], (200, 200, 200))
        y += 17


def draw_top_bar(screen, fonts, race_num, total_races, seed):
    pygame.draw.rect(screen, (15, 15, 25), (0, 0, WINDOW_W, GAME_Y - 5))
    txt(screen, f"RACE {race_num}/{total_races}", (GAME_X, 10), fonts["big"], (255, 220, 50))
    txt(screen, f"Track Seed: {seed}", (GAME_X + 200, 16), fonts["med"], (140, 140, 160))
    txt(screen, "Arrow Keys: Steer/Gas/Brake  |  ESC: Quit", (GAME_X, 38), fonts["small"], (100, 100, 120))


def draw_title_screen(screen, fonts, num_maps, num_agents) -> Optional[str]:
    screen.fill((15, 15, 25))
    cx = WINDOW_W // 2
    txt(screen, "CARRACING CHAMPIONSHIP", (cx - 220, 50), fonts["title"], (255, 220, 50))
    txt(screen, "Race Against the AI Ghosts!", (cx - 140, 95), fonts["med"], (180, 180, 220))
    txt(screen, f"{num_maps} Maps  |  {num_agents} AI Opponents", (cx - 140, 140), fonts["med"], (140, 140, 160))

    y = 190
    txt(screen, "Controls:", (cx - 170, y), fonts["med"], (200, 200, 200))
    for i, (k, a) in enumerate([("UP", "Gas"), ("DOWN", "Brake"), ("LEFT", "Steer Left"), ("RIGHT", "Steer Right")]):
        txt(screen, f"  {k:8s} — {a}", (cx - 150, y + 26 + i * 20), fonts["small"], (160, 160, 180))
    txt(screen, "  You can press multiple keys at once!", (cx - 150, y + 26 + 4 * 20), fonts["small"], (120, 200, 120))

    y = 350
    txt(screen, "[ENTER]  Start Championship", (cx - 140, y), fonts["med"], (80, 255, 80))
    txt(screen, "[G]      Generate / Refresh Ghosts", (cx - 140, y + 32), fonts["med"], (255, 200, 80))
    txt(screen, "[Q]      Quit", (cx - 140, y + 64), fonts["med"], (255, 80, 80))
    txt(screen, "Ghosts are recorded AI runs — beat their scores!", (cx - 200, WINDOW_H - 30), fonts["small"], (80, 80, 100))
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return "start"
            if event.key == pygame.K_g:
                return "generate"
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                return "quit"
    return None


def draw_countdown(screen, fonts, race_num, total_races, seed):
    cx, cy = WINDOW_W // 2, WINDOW_H // 2
    for label in ["3", "2", "1", "GO!"]:
        screen.fill((15, 15, 25))
        txt(screen, f"RACE {race_num}/{total_races}", (cx - 80, cy - 70), fonts["med"], (200, 200, 200))
        txt(screen, f"Track Seed: {seed}", (cx - 65, cy - 45), fonts["small"], (140, 140, 160))
        col = (255, 80, 80) if label != "GO!" else (80, 255, 80)
        ts = fonts["countdown"].render(label, True, col)
        screen.blit(ts, ts.get_rect(center=(cx, cy + 20)))
        pygame.display.flip()
        pygame.time.wait(600)
        for e in pygame.event.get():
            pass


def draw_race_results(screen, fonts, standings, race_num, championship, pts_awarded) -> bool:
    cx = WINDOW_W // 2
    while True:
        screen.fill((15, 15, 25))
        txt(screen, f"RACE {race_num} RESULTS", (cx - 120, 25), fonts["big"], (255, 220, 50))
        y = 70
        for pos, (name, reward) in enumerate(standings, 1):
            col = AGENT_COLOURS.get(name, (180, 180, 180))
            pc = (255, 215, 0) if pos == 1 else (192, 192, 192) if pos == 2 else (205, 127, 50) if pos == 3 else (150, 150, 150)
            is_p = name == "YOU"
            if is_p:
                pygame.draw.rect(screen, (35, 35, 60), (cx - 245, y - 2, 490, 23))
            dn = f">> {name} <<" if is_p else name
            txt(screen, f"{pos:2d}.", (cx - 235, y), fonts["med"], pc)
            txt(screen, f"{dn:24s}", (cx - 200, y), fonts["med"], col)
            rc = (80, 255, 80) if reward > 0 else (255, 80, 80)
            txt(screen, f"{reward:8.1f}", (cx + 80, y), fonts["med"], rc)
            pts = pts_awarded.get(name, 0)
            if pts > 0:
                txt(screen, f"+{pts}pt", (cx + 180, y), fonts["small"], (255, 220, 50))
            y += 24

        y += 15
        pygame.draw.line(screen, (50, 50, 70), (cx - 245, y), (cx + 245, y))
        y += 10
        txt(screen, "CHAMPIONSHIP STANDINGS", (cx - 125, y), fonts["med"], (255, 220, 50))
        y += 25
        for pos, (name, pts) in enumerate(sorted(championship.items(), key=lambda x: x[1], reverse=True), 1):
            col = AGENT_COLOURS.get(name, (180, 180, 180))
            is_p = name == "YOU"
            if is_p:
                pygame.draw.rect(screen, (35, 35, 60), (cx - 195, y - 1, 390, 18))
            dn = f">> {name} <<" if is_p else name
            txt(screen, f"{pos:2d}. {dn:20s} {pts:4d} pts", (cx - 175, y), fonts["small"], col)
            y += 18

        txt(screen, "ENTER = next race  |  Q = quit", (cx - 140, WINDOW_H - 35), fonts["small"], (100, 100, 120))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return True
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
        pygame.time.wait(20)


def draw_final_results(screen, fonts, championship):
    cx = WINDOW_W // 2
    champ = sorted(championship.items(), key=lambda x: x[1], reverse=True)
    winner = champ[0][0] if champ else "???"
    player_pos = next((i for i, (n, _) in enumerate(champ, 1) if n == "YOU"), 0)

    while True:
        screen.fill((15, 15, 25))
        txt(screen, "CHAMPIONSHIP FINAL RESULTS", (cx - 230, 20), fonts["title"], (255, 220, 50))
        y = 70
        if winner == "YOU":
            txt(screen, "YOU ARE THE CHAMPION!", (cx - 155, y), fonts["big"], (255, 215, 0))
        else:
            txt(screen, f"Champion: {winner}", (cx - 150, y), fonts["big"], AGENT_COLOURS.get(winner, (255, 255, 255)))
            txt(screen, f"Your position: {player_pos}", (cx - 85, y + 30), fonts["med"], (200, 200, 200))
        y += 70

        for pos, (name, pts) in enumerate(champ, 1):
            col = AGENT_COLOURS.get(name, (180, 180, 180))
            pc = (255, 215, 0) if pos == 1 else (192, 192, 192) if pos == 2 else (205, 127, 50) if pos == 3 else (150, 150, 150)
            is_p = name == "YOU"
            if is_p:
                pygame.draw.rect(screen, (35, 35, 60), (cx - 245, y - 2, 490, 25))
            medal = {1: " CHAMPION", 2: " 2nd", 3: " 3rd"}.get(pos, "")
            dn = f">> {name} <<" if is_p else name
            txt(screen, f"{pos:2d}.", (cx - 235, y), fonts["med"], pc)
            txt(screen, f"{dn:24s}", (cx - 200, y), fonts["med"], col)
            txt(screen, f"{pts:4d} pts", (cx + 100, y), fonts["med"], (200, 200, 200))
            if medal:
                txt(screen, medal, (cx + 175, y), fonts["small"], pc)
            y += 26

        txt(screen, "Press Q or ENTER to exit", (cx - 110, WINDOW_H - 35), fonts["small"], (100, 100, 120))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE, pygame.K_RETURN):
                    return
        pygame.time.wait(20)


def run_race(
    screen: pygame.Surface,
    fonts: Dict[str, pygame.font.Font],
    seed: int,
    race_num: int,
    total_races: int,
    ghosts: List[Dict[str, Any]],
    championship: Dict[str, int],
) -> Optional[List[Tuple[str, float]]]:
    """Run one race. Returns standings or None if user quit."""

    draw_countdown(screen, fonts, race_num, total_races, seed)

    env = gym.make(
        config.ENV_ID,
        continuous=True,
        render_mode="rgb_array",
        max_episode_steps=config.MAX_EPISODE_STEPS,
    )
    obs, info = env.reset(seed=seed)
    processor = ObservationProcessor()
    processor.reset()
    player_input = PlayerInput()

    total_reward = 0.0
    step = 0
    done = False
    clock = pygame.time.Clock()
    dt = 1.0 / GAME_FPS

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                env.close()
                return None

        action = player_input.update(dt)
        features = processor.process(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        car = env.unwrapped.car
        player_x = float(car.hull.position[0])
        player_y = float(car.hull.position[1])
        player_angle = float(car.hull.angle)
        t = env.unwrapped.t
        zoom = 0.1 * CR_SCALE * max(1 - t, 0) + CR_ZOOM_STEADY * min(t, 1)

        frame = env.render()
        if frame is None:
            continue

        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        game_surf = pygame.transform.smoothscale(surf, (GAME_W, GAME_H))

        draw_ghost_cars(game_surf, player_x, player_y, player_angle, zoom, ghosts, step - 1)

        screen.fill((10, 10, 18))
        draw_top_bar(screen, fonts, race_num, total_races, seed)
        screen.blit(game_surf, (GAME_X, GAME_Y))
        pygame.draw.rect(screen, (50, 50, 70), (GAME_X - 1, GAME_Y - 1, GAME_W + 2, GAME_H + 2), 1)

        draw_hud(
            screen, fonts,
            player_reward=total_reward,
            player_step=step,
            ghosts=ghosts,
            race_num=race_num,
            total_races=total_races,
            player_on_track=features["on_track"],
            player_speed=features["speed"],
            player_input=player_input,
            championship=championship,
        )

        pygame.display.flip()
        clock.tick(GAME_FPS)

    env.close()

    standings: List[Tuple[str, float]] = [("YOU", total_reward)]
    for g in ghosts:
        standings.append((g["agent"], g["final_reward"]))
    standings.sort(key=lambda x: x[1], reverse=True)
    return standings


def main():
    parser = argparse.ArgumentParser(description="CarRacing Championship")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--maps", type=int, default=NUM_MAPS)
    parser.add_argument("--seed", type=int, default=config.BASE_SEED)
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.refresh and os.path.exists(GHOST_DIR):
        import shutil
        shutil.rmtree(GHOST_DIR)
        print("  Cleared old ghost data.")

    if args.generate:
        print("\n  Generating ghost data...\n")
        generate_all_ghosts(args.seed, args.maps, args.no_random)
        print("\n  Done!")
        return

    screen, fonts = init_pygame()

    def ghosts_ready():
        for cls in ALL_AGENTS:
            a = cls()
            if args.no_random and "Random" in a.name:
                continue
            for i in range(args.maps):
                if not os.path.exists(ghost_path(a.name, args.seed + i)):
                    return False
        return True

    def need_positions():
        return not ghosts_have_positions(args.seed)

    running = True
    while running:
        choice = draw_title_screen(screen, fonts, args.maps, len(ALL_AGENTS) - (1 if args.no_random else 0))
        if choice == "quit":
            break
        elif choice == "generate" or choice == "start" and (not ghosts_ready() or need_positions()):
            screen.fill((15, 15, 25))
            msg = "Generating ghost data (with positions)..."
            if choice == "start" and not ghosts_ready():
                msg = "Ghost data missing — generating first..."
            elif choice == "start":
                msg = "Upgrading ghost data with positions..."
            txt(screen, msg, (WINDOW_W // 2 - 180, WINDOW_H // 2), fonts["med"], (255, 200, 80))
            txt(screen, "(check terminal for progress)", (WINDOW_W // 2 - 125, WINDOW_H // 2 + 28), fonts["small"], (120, 120, 140))
            pygame.display.flip()
            if need_positions() and os.path.exists(GHOST_DIR):
                import shutil
                shutil.rmtree(GHOST_DIR)
            generate_all_ghosts(args.seed, args.maps, args.no_random)
            if choice == "generate":
                continue

        if choice == "start":
            championship: Dict[str, int] = {"YOU": 0}
            for cls in ALL_AGENTS:
                a = cls()
                if args.no_random and "Random" in a.name:
                    continue
                championship[a.name] = 0

            for race_idx in range(args.maps):
                seed = args.seed + race_idx
                ghosts = load_ghosts(seed, args.no_random)
                standings = run_race(screen, fonts, seed, race_idx + 1, args.maps, ghosts, championship)
                if standings is None:
                    running = False
                    break

                pts_awarded: Dict[str, int] = {}
                for pos, (name, _) in enumerate(standings, 1):
                    pts = POINTS_TABLE.get(pos, 0)
                    championship[name] = championship.get(name, 0) + pts
                    if pts > 0:
                        pts_awarded[name] = pts

                if race_idx < args.maps - 1:
                    if not draw_race_results(screen, fonts, standings, race_idx + 1, championship, pts_awarded):
                        running = False
                        break
                else:
                    draw_race_results(screen, fonts, standings, race_idx + 1, championship, pts_awarded)

            if running:
                draw_final_results(screen, fonts, championship)

    pygame.quit()


if __name__ == "__main__":
    main()
