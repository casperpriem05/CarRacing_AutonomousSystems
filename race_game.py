#!/usr/bin/env python3
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

from game.race import main

if __name__ == "__main__":
    main()
