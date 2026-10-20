#!/usr/bin/env python3
"""Export reproducible 0/1 ML-KEM RRAM weights; refuse changed existing weights."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from binary_rram import export_weights


if __name__ == "__main__":
    manifest = export_weights()
    print(f"Verified/exported {len(manifest['artifacts'])} binary RRAM weight artifacts")
    print("NTT/INTT: 12 shared 128x128 planes each, four 64x64 tiles per plane")
