#!/usr/bin/python3
"""
This script is deprecated. Road fusion is now integrated into RoadEdgeProcess.
To fuse road records, simply run make_vector.py with process='road'.
The fused records will be saved automatically with '_fused.json' suffix.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    print("This standalone road fusion script is deprecated.")
    print("Road fusion is now integrated into RoadEdgeProcess.")
    print("To fuse road records, simply run make_vector.py with process='road'.")
    print("The fused records will be saved automatically with '_fused.json' suffix.")
    
    # For backward compatibility, we can still try to run the old fusion
    # But it's better to just exit
    sys.exit(1)

if __name__ == "__main__":
    main()
