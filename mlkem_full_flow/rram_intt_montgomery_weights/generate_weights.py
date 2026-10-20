#!/usr/bin/env python3
"""Generate/recheck the dedicated twelve-plane INTT-only weight bank."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rram_intt_montgomery import export_weights

if __name__ == "__main__":
    manifest = export_weights()
    print(f"Verified/exported {len(manifest['artifacts'])} INTT-only artifacts; R=2^32, unsigned12 weights")
