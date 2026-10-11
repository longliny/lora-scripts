"""CLI entry point for the project's Krea 2 LoRA trainer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mikazuki.training.krea2 import main, setup_parser


if __name__ == "__main__":
    main()
