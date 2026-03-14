#!/usr/bin/env python3
"""Generate ghost data one-at-a-time in separate processes to avoid Box2D segfaults."""

from game.generate_ghosts import main

if __name__ == "__main__":
    main()
