"""Project Anima trainer entry point."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts" / "dev"))

from mikazuki.training.anima import main, setup_parser

if __name__ == "__main__":
    main()
