#!/usr/bin/python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path:
    first_path = Path(sys.path[0] or ".").resolve()
    if first_path == SCRIPT_DIR:
        sys.path.pop(0)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vectorize.fuse_road_edges import main


if __name__ == "__main__":
    main()
