"""Ideal binary RRAM mapping and genuine bit-serial ML-KEM NTT/INTT.

Stored physical weights are only uint8 0/1 values. Physical row = input,
physical column = output. Each 128x128 plane consists of four 64x64 tiles.
The compute path uses packed binary AND/popcount, merges the two input-row
tiles, then digitally shifts and adds all 12x12 weight/input bit pairs.
It never reconstructs an integer weight matrix for the transform calculation.
These exact partial counts are a logical model, not measured device behavior.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path
from typing import Callable, Iterable
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parent
RRAM_DIR = ROOT / "rram_binary"
SOURCE_PATH = ROOT / "matrices/mlkem_shared_matrices.npz"
SOURCE_SHA256 = "b96b6c339da9c9599319094c8171603cba2fb6a338a823790be762f40c8de233"
WEIGHTS_NAME = "mlkem_binary_weights.npz"
Q, BITS, LANE_SIZE, TILE_SIZE = 3329, 12, 128, 64
TILE_INPUT_OFFSETS = np.array([0, 0, 64, 64], dtype=np.int64)
TILE_OUTPUT_OFFSETS = np.array([0, 64, 0, 64], dtype=np.int64)
BIT_WEIGHTS = 1 << np.arange(BITS, dtype=np.int64)
_BYTE_POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_arrays() -> dict[str, np.ndarray]:
    if _sha256(SOURCE_PATH.read_bytes()) != SOURCE_SHA256:
        raise ValueError("source matrix SHA-256 does not match the recorded source")
    with np.load(SOURCE_PATH, allow_pickle=False) as data:
        return {"ntt": data["W128"].copy(), "intt": data["W128_inv"].copy()}


def _binary_arrays(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays = {
        "bit_weights": BIT_WEIGHTS,
        "tile_input_offsets": TILE_INPUT_OFFSETS,
        "tile_output_offsets": TILE_OUTPUT_OFFSETS,
        "modulus": np.array(Q, dtype=np.int64),
    }
    for direction, matrix in source.items():
        if matrix.shape != (LANE_SIZE, LANE_SIZE) or matrix.dtype != np.dtype("int64"):
            raise ValueError("source matrices must be 128x128 int64 arrays")
        if np.any(matrix < 0) or np.any(matrix >= Q):
            raise ValueError("source matrix residues must be canonical")
        # Mathematical W[out,in] becomes physical P[in,out] before bit slicing.
        planes = ((matrix.T[None, :, :] >> np.arange(BITS)[:, None, None]) & 1).astype(np.uint8)
        tiles = np.stack([planes[:, row:row + TILE_SIZE, col:col + TILE_SIZE]
                          for row, col in zip(TILE_INPUT_OFFSETS, TILE_OUTPUT_OFFSETS)], axis=1)
        arrays[f"{direction}_physical_planes"] = planes
        arrays[f"{direction}_tiles"] = tiles
    return arrays


def _deterministic_npz(arrays: dict[str, np.ndarray]) -> bytes:
    """Use fixed ZIP metadata and NPY v1 headers for reproducible artifact bytes."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            output.writestr(info, payload.getvalue())
    return archive.getvalue()


def _csv_bytes(plane: np.ndarray) -> bytes:
    return ("\n".join(",".join(map(str, row.tolist())) for row in plane) + "\n").encode("ascii")


def export_weights(output_dir: str | Path = RRAM_DIR) -> dict:
    """Create missing artifacts; reject existing weight/CSV bytes that differ.

    Weight files are reproducible and never silently repaired or overwritten.
    The provenance metadata can be refreshed to record current code hashes.
    This function is the build entry point used by generate_weights.py.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_arrays()
    arrays = _binary_arrays(source)
    files = {WEIGHTS_NAME: _deterministic_npz(arrays)}
    artifacts = {WEIGHTS_NAME: {"role": "binary physical planes, tiles and layout metadata",
                               "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                          for key, value in arrays.items()}}}
    for direction in ("ntt", "intt"):
        for bit in range(BITS):
            name = f"{direction}_bit{bit:02d}.csv"
            files[name] = _csv_bytes(arrays[f"{direction}_physical_planes"][bit])
            artifacts[name] = {"shape": [LANE_SIZE, LANE_SIZE], "dtype": "uint8",
                               "values": [0, 1], "direction": direction, "weight_bit": bit,
                               "role": "physical plane: input rows, output columns; no header"}
    # Check all existing data before writing even a missing file.
    for name, content in files.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"existing binary RRAM artifact differs from expected bytes: {path}")
        artifacts[name]["sha256"] = _sha256(content)
        artifacts[name]["bytes"] = len(content)
    for name, content in files.items():
        if not (output_dir / name).exists():
            (output_dir / name).write_bytes(content)
    code_paths = [ROOT / "mlkem.py", Path(__file__).resolve(), RRAM_DIR / "generate_weights.py"]
    manifest = {
        "format_version": 1,
        "kind": "ideal binary bit-plane RRAM mapping; no analog or measured-hardware model",
        "source": {"path": "../matrices/mlkem_shared_matrices.npz", "sha256": SOURCE_SHA256},
        "code": {str(path.relative_to(ROOT)): {"sha256": _sha256(path.read_bytes())}
                 for path in code_paths if path.exists()},
        "shared_by": ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"],
        "modulus": Q, "weight_bits": BITS, "input_bits": BITS,
        "bit_order": "LSB first; weight bit b and input slot t range from 0 through 11",
        "orientation": "W_math[output,input]; physical_plane[b,input,output] = bit_b(W_math[output,input])",
        "tile_order": [{"index": i, "input_row_offset": int(row), "output_column_offset": int(col)}
                       for i, (row, col) in enumerate(zip(TILE_INPUT_OFFSETS, TILE_OUTPUT_OFFSETS))],
        "inverse_scale": {"value": 3303, "meaning": "128^-1 mod 3329 already fused into W128_inv; not applied again"},
        "reuse": "Even and odd 128-coefficient lanes use the same stored weights in two logical evaluations.",
        "compute": [
            "Canonicalize 256 integer coefficients modulo 3329; split even and odd lanes.",
            "Encode each lane as twelve uint8 input bit planes, retaining zero-valued slots.",
            "For each lane, input bit, weight bit and tile, count binary AND matches over its 64 input rows.",
            "Merge tile 0+2 and tile 1+3 along input rows, yielding 128 output counts in 0..128.",
            "Cast counts to int64, shift by b+t, and digitally sum all 144 bit-pair terms.",
            "Reduce modulo 3329 and re-interleave the even/odd lanes."
        ],
        "array_counts": {"directions": 2, "planes_per_direction": 12, "tiles_per_plane": 4,
                         "total_64x64_binary_tiles": 96, "total_stored_binary_cells": 2 * BITS * LANE_SIZE**2},
        "bounds": {"raw_tile_count_max": 64, "merged_count_max": 128,
                   "unreduced_coefficient_accumulator_max": LANE_SIZE * (Q - 1)**2},
        "timing_scope": "Input bit slots and lane evaluations are algorithmic organization; no hardware cycle count or scheduling claim.",
        "artifacts": artifacts,
    }
    (output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@lru_cache(maxsize=1)
def load_weights() -> dict[str, np.ndarray]:
    """Load and integrity-check generated binary artifacts, never integer weights."""
    manifest_path = RRAM_DIR / "provenance.json"
    if not manifest_path.exists() or not (RRAM_DIR / WEIGHTS_NAME).exists():
        export_weights()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source", {}).get("sha256") != SOURCE_SHA256:
        raise ValueError("binary RRAM provenance refers to an unexpected source matrix")
    for name, record in manifest["artifacts"].items():
        path = RRAM_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"missing RRAM artifact {path}; run rram_binary/generate_weights.py")
        if _sha256(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"binary RRAM artifact SHA-256 mismatch: {path}")
    with np.load(RRAM_DIR / WEIGHTS_NAME, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if not np.array_equal(arrays["tile_input_offsets"], TILE_INPUT_OFFSETS):
        raise ValueError("incorrect binary RRAM input-row tile order")
    if not np.array_equal(arrays["tile_output_offsets"], TILE_OUTPUT_OFFSETS):
        raise ValueError("incorrect binary RRAM output-column tile order")
    if not np.array_equal(arrays["bit_weights"], BIT_WEIGHTS) or int(arrays["modulus"]) != Q:
        raise ValueError("incorrect binary RRAM bit significance or modulus")
    for direction in ("ntt", "intt"):
        planes, tiles = arrays[f"{direction}_physical_planes"], arrays[f"{direction}_tiles"]
        for array, shape in ((planes, (BITS, LANE_SIZE, LANE_SIZE)), (tiles, (BITS, 4, TILE_SIZE, TILE_SIZE))):
            if array.dtype != np.uint8 or array.shape != shape or np.any(array > 1):
                raise ValueError("binary RRAM arrays must have specified shape and uint8 0/1 values")
        for i, (row, col) in enumerate(zip(TILE_INPUT_OFFSETS, TILE_OUTPUT_OFFSETS)):
            if not np.array_equal(tiles[:, i], planes[:, row:row + TILE_SIZE, col:col + TILE_SIZE]):
                raise ValueError("binary RRAM tile does not match its physical plane")
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


def popcount_u64(words: np.ndarray, force_fallback: bool = False) -> np.ndarray:
    """Elementwise population counts; NumPy >=1.24 fallback uses a byte lookup."""
    words = np.ascontiguousarray(words, dtype=np.uint64)
    if hasattr(np, "bitwise_count") and not force_fallback:
        return np.bitwise_count(words)
    bytes_view = words.view(np.uint8).reshape(words.shape + (8,))
    return _BYTE_POPCOUNT[bytes_view].sum(axis=-1, dtype=np.uint16).astype(np.uint8)


def _pack_rows(bits: np.ndarray) -> np.ndarray:
    """Pack the final 64 binary-row axis into uint64, row zero in bit zero."""
    packed = np.packbits(bits, axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").reshape(bits.shape[:-1]).astype(np.uint64)


class BinaryRRAM:
    """Reused binary weights and bit-serial, exact-integer transform evaluation."""

    def __init__(self):
        arrays = load_weights()
        self.physical_planes = {name: arrays[f"{name}_physical_planes"] for name in ("ntt", "intt")}
        self.tiles = {name: arrays[f"{name}_tiles"] for name in ("ntt", "intt")}
        # [weight_bit,tile,output_col] words each encode its 64 input-row cells.
        self.packed_tiles = {name: _pack_rows(tiles.transpose(0, 1, 3, 2))
                             for name, tiles in self.tiles.items()}

    def transform(self, source: Iterable[int], inverse: bool = False,
                  trace: Callable[[str, dict], None] | None = None,
                  stage: str = "rram_binary", force_fallback: bool = False) -> list[int]:
        values = list(source)
        if len(values) != 256 or any(not isinstance(x, Integral) for x in values):
            raise ValueError("binary RRAM transform needs 256 integer coefficients")
        values = [int(x) % Q for x in values]
        direction = "intt" if inverse else "ntt"
        lanes = np.array(values, dtype=np.int64).reshape(LANE_SIZE, 2).T
        # Input slots are explicit even when a whole slot is zero.
        input_bits = ((lanes[:, None, :] >> np.arange(BITS)[None, :, None]) & 1).astype(np.uint8)
        input_words = _pack_rows(input_bits.reshape(2, BITS, 2, TILE_SIZE))
        tile_input_words = input_words[:, :, [0, 0, 1, 1]]
        # Genuine binary multiplication: AND over packed physical input rows,
        # followed by popcount. No dense or reconstructed integer W is used.
        intersections = tile_input_words[:, :, None, :, None] & self.packed_tiles[direction][None, None, :, :, :]
        tile_counts = popcount_u64(intersections, force_fallback=force_fallback)
        # Axes: [lane,input_bit,weight_bit,tile,output_column_in_tile].
        counts16 = tile_counts.astype(np.uint16)
        merged_counts = np.concatenate((counts16[..., 0, :] + counts16[..., 2, :],
                                        counts16[..., 1, :] + counts16[..., 3, :]), axis=-1)
        shifts = np.arange(BITS)[:, None] + np.arange(BITS)[None, :]
        weighted_terms = merged_counts.astype(np.int64) << shifts[None, :, :, None]
        per_input_slot_accumulators = weighted_terms.sum(axis=2, dtype=np.int64)
        accumulators = per_input_slot_accumulators.sum(axis=1, dtype=np.int64)
        output_lanes = accumulators % Q
        output = output_lanes.T.reshape(256).tolist()
        if trace is not None:
            trace(stage, {
                "operation": "rram_binary_vmm", "transform": direction,
                "weight_artifact": "rram_binary/" + WEIGHTS_NAME,
                "physical_orientation": "input_rows_output_columns",
                "lane_order": ["even", "odd"], "input_bit_slots": BITS,
                "weight_bit_planes": BITS, "bit_order": "LSB-first",
                "input_lanes": lanes, "input_bitplanes": input_bits,
                "tile_input_offsets": TILE_INPUT_OFFSETS.copy(),
                "tile_output_offsets": TILE_OUTPUT_OFFSETS.copy(),
                "tile_counts_axes": ["lane", "input_bit", "weight_bit", "tile", "output_column_in_tile"],
                "tile_counts": tile_counts, "merged_counts": merged_counts,
                "shift_amounts": shifts, "weighted_terms": weighted_terms,
                "per_input_slot_accumulators": per_input_slot_accumulators,
                "cumulative_input_slot_accumulators": np.cumsum(per_input_slot_accumulators, axis=1),
                "weighted_accumulators": accumulators, "output_lanes": output_lanes,
                "output": output, "inverse_normalization_in_weights": inverse,
                "raw_tile_count_max": int(tile_counts.max()),
                "merged_count_max": int(merged_counts.max()),
                "count_model": "ideal integer partial count; no analog noise, ADC or cycle model",
            })
        return output
