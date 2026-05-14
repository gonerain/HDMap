"""Incremental pickle reader for outdoor.pkl-style streams.

outdoor_livox_ie.py writes one frame at a time via pickle.dump. Reading
back requires repeatedly calling pickle.load() until EOFError. This
helper centralizes that idiom.
"""
import pickle

import numpy as np


def load_frames(path, dtype=np.float64):
    """Read all pickle-frame records from `path` into a list of ndarrays."""
    frames = []
    with open(path, "rb") as f:
        while True:
            try:
                frames.append(np.asarray(pickle.load(f), dtype=dtype))
            except EOFError:
                break
    return frames
