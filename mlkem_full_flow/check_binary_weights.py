#!/usr/bin/env python3
"""Check that every physical weight plane/tile/CSV contains only 0 and 1.

NPZ layout metadata (modulus, offsets, radix and digital bit weights) is not a
physical weight array. The dense mathematical transform matrices contain
ordinary residues modulo 3329; neither category represents conductance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from validation import require

ROOT = Path(__file__).resolve().parent


def verify_binary_weights():
    """Check bit arrays and CSV identity; return counts and matrix ranges."""
    rows = []
    csv_cells = 0
    npz_cells = 0
    csv_files = 0
    for folder, filename, directions in (
        ("rram_binary", "mlkem_binary_weights.npz", ("ntt", "intt")),
        ("rram_intt_montgomery_weights", "intt_montgomery_weights.npz", ("intt",)),
    ):
        with np.load(ROOT / folder / filename, allow_pickle=False) as archive:
            binary_names = {f"{direction}_{kind}"
                            for direction in directions
                            for kind in ("physical_planes", "tiles")}
            require(binary_names <= set(archive.files), "Missing binary arrays")
            for name in sorted(binary_names):
                values = archive[name]
                require(values.dtype == np.uint8, name + " must be uint8")
                expected_shape = ((12, 128, 128) if name.endswith("physical_planes")
                                  else (12, 4, 64, 64))
                require(values.shape == expected_shape, name + " shape")
                require(np.all((values == 0) | (values == 1)), name + " binary values")
                npz_cells += values.size
                rows.append({"file": folder + "/" + filename, "array": name,
                             "shape": list(values.shape), "values": [0, 1],
                             "checked_elements": int(values.size)})
            for direction in directions:
                for bit in range(12):
                    csv = ROOT / folder / f"{direction}_bit{bit:02d}.csv"
                    # Parse without integer truncation: a value such as 0.1
                    # must fail the binary check rather than silently become 0.
                    plane = np.loadtxt(csv, delimiter=",", dtype=np.float64)
                    require(plane.shape == (128, 128), csv.name + " shape")
                    require(np.all((plane == 0) | (plane == 1)), csv.name + " binary values")
                    require(np.array_equal(plane, archive[f"{direction}_physical_planes"][bit]),
                            csv.name + " must match its NPZ plane")
                    csv_cells += plane.size
                    csv_files += 1
    matrices = {}
    with np.load(ROOT / "matrices/mlkem_shared_matrices.npz", allow_pickle=False) as archive:
        for name in ("W128", "W128_inv"):
            matrix = archive[name]
            require(matrix.shape == (128, 128) and matrix.dtype == np.int64,
                    name + " mathematical matrix shape/type")
            require(np.all((matrix >= 0) & (matrix < 3329)), name + " residues")
            matrices[name] = {"shape": [128, 128],
                              "min": int(matrix.min()), "max": int(matrix.max()),
                              "representation": "ordinary integer residues modulo 3329"}
    return {
        "status": "passed", "binary_npz_arrays": rows,
        "npz_binary_elements_checked": int(npz_cells),
        "csv_files_checked": csv_files, "csv_binary_elements_checked": int(csv_cells),
        "all_weight_arrays_are_binary": True, "conductance_values_in_weights": False,
        "mathematical_matrices": matrices,
        "metadata_scope": "Offsets, modulus, radix and digital reconstruction constants are not physical weights.",
    }


if __name__ == "__main__":
    report = verify_binary_weights()
    output = ROOT / "transform_only_results"
    output.mkdir(exist_ok=True)
    (output / "binary_weight_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
