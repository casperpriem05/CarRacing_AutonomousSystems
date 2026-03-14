#!/usr/bin/env python3
"""Generate a side-by-side comparison video of all agents' best episodes.

Usage:
    python compare_video.py
    python compare_video.py --tile-size 320
    python compare_video.py --fps 30
"""

from evaluation.compare_video import main

if __name__ == "__main__":
    main()
