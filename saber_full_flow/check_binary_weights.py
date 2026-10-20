#!/usr/bin/env python3
"""Verify that every exported physical Saber weight is exactly binary.

CSV tokens are checked as text before conversion. In particular, a value such
as 0.1 is rejected rather than silently truncated to an integer. Dense matrices
under matrices/ contain modular coefficients; they are not conductance values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODULI = (3329, 7681)


def read_binary_csv(path: Path) -> np.ndarray:
    """Read exactly 64x64 literal 0/1 tokens, rejecting every other value."""
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 64 or any(len(row) != 64 for row in rows):
        raise ValueError(f"{path.name}: expected 64 rows and 64 columns")
    for row_index, row in enumerate(rows):
        for column_index, token in enumerate(row):
            if token.strip() not in ("0", "1"):
                raise ValueError(
                    f"{path.name}: non-binary token {token!r} at "
                    f"row {row_index}, column {column_index}"
                )
    return np.array([[int(token) for token in row] for row in rows], dtype=np.uint8)


def verify_weights(weights_dir: Path | None = None) -> dict:
    """Check all 50 CSV planes and the four matching NPZ plane stacks."""
    directory = ROOT / "rram_binary" if weights_dir is None else Path(weights_dir)
    expected_names = {
        f"{direction}_{p}_bit{bit:02d}.csv"
        for p in MODULI
        for direction in ("ntt", "intt")
        for bit in range((p - 1).bit_length())
    }
    actual_names = {path.name for path in directory.glob("*.csv")}
    if actual_names != expected_names:
        raise ValueError("Expected exactly the 50 named NTT/INTT binary CSV files")
    csv_arrays = {name: read_binary_csv(directory / name) for name in sorted(expected_names)}
    expected_keys = {
        f"{direction}_{p}_physical_planes"
        for p in MODULI for direction in ("ntt", "intt")
    }
    archive_path = directory / "saber_binary_weights.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys | {"moduli", "R32"}:
            raise ValueError("Unexpected binary weight NPZ keys")
        if not np.array_equal(archive["moduli"], np.array(MODULI)):
            raise ValueError("Unexpected auxiliary-prime metadata")
        if archive["R32"].shape != () or int(archive["R32"]) != 1 << 32:
            raise ValueError("Unexpected Montgomery-radix metadata")
        for p in MODULI:
            bits = (p - 1).bit_length()
            for direction in ("ntt", "intt"):
                key = f"{direction}_{p}_physical_planes"
                planes = archive[key]
                if planes.shape != (bits, 64, 64) or planes.dtype != np.uint8:
                    raise ValueError(f"{key}: expected uint8 planes of shape {(bits, 64, 64)}")
                if not np.all((planes == 0) | (planes == 1)):
                    raise ValueError(f"{key}: NPZ contains non-binary values")
                for bit, plane in enumerate(planes):
                    name = f"{direction}_{p}_bit{bit:02d}.csv"
                    if not np.array_equal(plane, csv_arrays[name]):
                        raise ValueError(f"{name}: CSV differs from its NPZ plane")
    paths = [directory / name for name in sorted(expected_names)] + [archive_path]
    return {
        "status": "passed",
        "csv_files": len(expected_names),
        "npz_plane_stacks": len(expected_keys),
        "planes": 50,
        "binary_cells": 50 * 64 * 64,
        "stored_weight_values": [0, 1],
        "csv_validation": "Literal 0/1 tokens; no floating-point to integer truncation",
        "orientation": "plane[weight_bit, input_row, output_column]",
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=None)
    args = parser.parse_args()
    report = verify_weights(args.weights_dir)
    destination = ROOT / "generated/results"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "binary_weights.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
