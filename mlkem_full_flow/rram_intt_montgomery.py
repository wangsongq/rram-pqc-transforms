"""INTT-only binary RRAM weights with fused normalization and REDC32 scaling.

Let V be the existing normalized inverse matrix. Stored K = R*V mod q, where
R=2**32 and V=3303*U mod q. Twelve unsigned 0/1 weight planes multiply ordinary
unsigned twelve-bit input coefficients through AND/popcount and shift/add.
One REDC32 removes R and returns ordinary canonical output. There is no second
normalization, no change to NTT/SampleNTT, and no dense runtime matrix product.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from binary_rram import (
    BITS, Q, SOURCE_PATH, SOURCE_SHA256, TILE_INPUT_OFFSETS, TILE_OUTPUT_OFFSETS,
    BIT_WEIGHTS, _deterministic_npz, _csv_bytes, _pack_rows, popcount_u64,
)


ROOT = Path(__file__).resolve().parent
RRAM_DIR = ROOT / "rram_intt_montgomery_weights"
WEIGHTS_NAME = "intt_montgomery_weights.npz"
RADIX_BITS = 32
R = 1 << RADIX_BITS
R_MOD_Q = R % Q  # 1353
QPRIME = (-pow(Q, -1, R)) % R  # 2488732927
MASK32 = R - 1
DOT_BOUND = 128 * (Q - 1)**2  # 1,417,674,752 < R < qR


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _make_arrays():
    if _sha(SOURCE_PATH.read_bytes()) != SOURCE_SHA256:
        raise ValueError("source inverse matrix artifact SHA-256 mismatch")
    with np.load(SOURCE_PATH, allow_pickle=False) as archive:
        inverse = archive["W128_inv"].copy()
    if inverse.shape != (128, 128) or inverse.dtype != np.int64 or np.any(inverse < 0) or np.any(inverse >= Q):
        raise ValueError("source inverse matrix must be canonical 128x128 int64")
    # V already includes 3303. Multiply only by R mod q, not by 3303 again.
    weight = (inverse * R_MOD_Q) % Q
    planes = ((weight.T[None] >> np.arange(BITS)[:, None, None]) & 1).astype(np.uint8)
    tiles = np.stack([planes[:, row:row + 64, col:col + 64]
                      for row, col in zip(TILE_INPUT_OFFSETS, TILE_OUTPUT_OFFSETS)], axis=1)
    return {
        "intt_physical_planes": planes, "intt_tiles": tiles,
        "bit_weights": BIT_WEIGHTS, "tile_input_offsets": TILE_INPUT_OFFSETS,
        "tile_output_offsets": TILE_OUTPUT_OFFSETS, "modulus": np.array(Q, dtype=np.int64),
        "weight_bits": np.array(BITS, dtype=np.int64), "input_bits": np.array(BITS, dtype=np.int64),
        "radix_bits": np.array(RADIX_BITS, dtype=np.int64), "radix": np.array(R, dtype=np.uint64),
        "radix_mod_q": np.array(R_MOD_Q, dtype=np.int64), "qprime": np.array(QPRIME, dtype=np.uint64),
        "inverse_scale": np.array(3303, dtype=np.int64),
    }


def export_weights(output_dir=RRAM_DIR):
    """Create missing deterministic artifacts; refuse changed existing weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _make_arrays()
    files = {WEIGHTS_NAME: _deterministic_npz(arrays)}
    artifacts = {WEIGHTS_NAME: {"role": "INTT-only fused binary planes, tiles and metadata",
                               "arrays": {name: {"shape": list(a.shape), "dtype": str(a.dtype)}
                                          for name, a in arrays.items()}}}
    for bit in range(BITS):
        name = f"intt_bit{bit:02d}.csv"
        files[name] = _csv_bytes(arrays["intt_physical_planes"][bit])
        artifacts[name] = {"shape": [128, 128], "dtype": "uint8", "values": [0, 1],
                           "weight_bit": bit, "role": "physical input rows, output columns; no header"}
    for name, data in files.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"existing INTT artifact differs from expected bytes: {path}")
        artifacts[name].update(sha256=_sha(data), bytes=len(data))
    for name, data in files.items():
        if not (output_dir / name).exists():
            (output_dir / name).write_bytes(data)
    code_paths = [ROOT / "rram_intt_montgomery.py", ROOT / "rram_transform_only.py",
                  ROOT / "binary_rram.py", RRAM_DIR / "generate_weights.py"]
    manifest = {
        "format_version": 1,
        "kind": "INTT-only ideal binary RRAM weights with fused REDC32 compensation",
        "source": {"path": "../matrices/mlkem_shared_matrices.npz", "key": "W128_inv", "sha256": SOURCE_SHA256},
        "code": {str(path.relative_to(ROOT)): {"sha256": _sha(path.read_bytes())}
                 for path in code_paths if path.exists()},
        "formula": "K = (2^32 * V) mod 3329, V = 3303*U mod 3329; physical_plane[b,in,out]=bit_b(K[out,in])",
        "normalization": "V already contains 128^-1 mod3329=3303. Exactly one normalization is present; no extra scale after REDC32.",
        "domain_contract": {"input": "ordinary canonical residues 0..3328", "stored_weight": "R*V mod q, canonical unsigned12",
                            "dot": "unreduced nonnegative integer sum of K*x", "output": "ordinary canonical residues REDC32(dot)"},
        "modulus": Q, "radix_bits": RADIX_BITS, "radix": R, "radix_mod_q": R_MOD_Q, "qprime": QPRIME,
        "weight_bits": BITS, "input_bits": BITS, "pulse_levels": [0, 1], "bit_order": "LSB first",
        "tile_order": [{"index": i, "input_row_offset": int(row), "output_column_offset": int(col)}
                       for i, (row, col) in enumerate(zip(TILE_INPUT_OFFSETS, TILE_OUTPUT_OFFSETS))],
        "shared_by": ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"],
        "unchanged": "All forward NTT weights and paths, SampleNTT, quadratic products, noise/compression and FO remain ordinary-domain baseline algorithms.",
        "redc32": ["Require 0 <= T < q*R.", "m = (((T & (R-1))*qprime) & (R-1)) using uint64.",
                   "u = (T + m*q) >> 32.", "Return u-q if u>=q, otherwise u."],
        "bounds": {"full_dot_max": DOT_BOUND, "q_times_R": Q * R,
                   "lowword_times_qprime_max": MASK32 * QPRIME,
                   "redc_numerator_strict_upper": 2 * Q * R,
                   "raw_tile_count_max": 64, "merged_count_max": 128},
        "array_counts": {"planes": 12, "tiles_per_plane": 4, "total_64x64_tiles": 48, "stored_binary_cells": 12 * 128 * 128},
        "compute": "Binary AND/popcount per 64-row tile, merge input-row tiles, signed-safe int64 shift/add, REDC32, interleave lanes. No dense VMM shortcut.",
        "scope": "Exact logical model; radix factor stored in weights does not establish a hardware speedup or validation claim.",
        "artifacts": artifacts,
    }
    (output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@lru_cache(maxsize=1)
def load_weights():
    manifest_path = RRAM_DIR / "provenance.json"
    if not manifest_path.exists() or not (RRAM_DIR / WEIGHTS_NAME).exists():
        export_weights()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source", {}).get("sha256") != SOURCE_SHA256:
        raise ValueError("unexpected INTT source provenance")
    expected_files = {WEIGHTS_NAME} | {f"intt_bit{b:02d}.csv" for b in range(BITS)}
    if set(manifest["artifacts"]) != expected_files:
        raise ValueError("INTT artifact manifest must cover the NPZ and twelve CSV planes")
    for name, record in manifest["artifacts"].items():
        if _sha((RRAM_DIR / name).read_bytes()) != record["sha256"]:
            raise ValueError(f"INTT artifact hash mismatch: {name}")
    with np.load(RRAM_DIR / WEIGHTS_NAME, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, expected in (("modulus", Q), ("radix_bits", RADIX_BITS), ("radix", R),
                           ("radix_mod_q", R_MOD_Q), ("qprime", QPRIME),
                           ("weight_bits", BITS), ("input_bits", BITS), ("inverse_scale", 3303)):
        if int(arrays[name]) != expected:
            raise ValueError(f"incorrect INTT metadata: {name}")
    for name, expected in (("bit_weights", BIT_WEIGHTS), ("tile_input_offsets", TILE_INPUT_OFFSETS),
                           ("tile_output_offsets", TILE_OUTPUT_OFFSETS)):
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"incorrect INTT layout: {name}")
    planes, tiles = arrays["intt_physical_planes"], arrays["intt_tiles"]
    for array, shape in ((planes, (12, 128, 128)), (tiles, (12, 4, 64, 64))):
        if array.dtype != np.uint8 or array.shape != shape or np.any(array > 1):
            raise ValueError("INTT weights must be uint8 0/1 arrays with the declared shape")
    for index, (row, col) in enumerate(zip(TILE_INPUT_OFFSETS, TILE_OUTPUT_OFFSETS)):
        if not np.array_equal(tiles[:, index], planes[:, row:row + 64, col:col + 64]):
            raise ValueError("INTT tile differs from its physical plane")
    for array in arrays.values():
        array.flags.writeable = False
    return arrays


def _redc32_core(values):
    original = np.asarray(values)
    if original.dtype.kind not in "iu" or np.any(original < 0) or np.any(original >= Q * R):
        raise ValueError("REDC32 requires integer inputs 0 <= T < q*R")
    t = original.astype(np.uint64, copy=False)
    # Mask BEFORE multiplying: T*qprime can exceed uint64 over the allowed
    # T<qR domain. low32*qprime <2**64, even when it exceeds signed int64.
    low32 = t & np.uint64(MASK32)
    m = (low32 * np.uint64(QPRIME)) & np.uint64(MASK32)
    numerator = t + m * np.uint64(Q)  # <2*q*R <2**45, so no uint64 overflow.
    quotient = (numerator >> np.uint64(RADIX_BITS)).astype(np.int64)
    if np.any(numerator & np.uint64(MASK32)) or np.any(quotient >= 2 * Q):
        raise ArithmeticError("REDC32 invariant failure")
    subtract_q = quotient >= Q
    output = quotient - np.where(subtract_q, Q, 0)
    details = dict(operation="montgomery_reduce", reduction="REDC32", radix_bits=RADIX_BITS,
                   radix=R, qprime=QPRIME, input=t, low32=low32, correction_m=m,
                   numerator=numerator, quotient=quotient, subtract_q=subtract_q,
                   output=output, output_domain="ordinary_canonical", removes_factor="R",
                   extra_inverse_normalization=False)
    return output, details


def redc32(values, trace=None, stage="redc32"):
    """Return canonical T*R^-1 modq for nonnegative T<qR; no division by q."""
    output, details = _redc32_core(values)
    if trace is not None:
        trace(stage, details)
    return output


class MontgomeryINTT:
    """Ordinary-input/ordinary-output inverse transform using fused binary K."""

    def __init__(self):
        arrays = load_weights()
        self.physical_planes = arrays["intt_physical_planes"]
        self.tiles = arrays["intt_tiles"]
        self.packed_tiles = _pack_rows(self.tiles.transpose(0, 1, 3, 2))

    def transform(self, source, trace=None, stage="intt.rram_binary"):
        source = list(source)
        if len(source) != 256 or any(not isinstance(x, Integral) or not 0 <= int(x) < Q for x in source):
            raise ValueError("Montgomery INTT input must be 256 ordinary canonical integers")
        lanes = np.array(source, dtype=np.int64).reshape(128, 2).T
        input_bits = ((lanes[:, None] >> np.arange(BITS)[None, :, None]) & 1).astype(np.uint8)
        words = _pack_rows(input_bits.reshape(2, BITS, 2, 64))
        tile_words = words[:, :, [0, 0, 1, 1]]
        intersections = tile_words[:, :, None, :, None] & self.packed_tiles[None, None]
        tile_counts = popcount_u64(intersections)
        counts = tile_counts.astype(np.int64)
        merged = np.concatenate((counts[..., 0, :] + counts[..., 2, :],
                                 counts[..., 1, :] + counts[..., 3, :]), axis=-1)
        shifts = np.arange(BITS)[:, None] + np.arange(BITS)[None, :]
        weighted_terms = merged << shifts[None, :, :, None]
        per_slot = weighted_terms.sum(axis=2, dtype=np.int64)
        accumulators = per_slot.sum(axis=1, dtype=np.int64)
        if np.any(accumulators < 0) or np.any(accumulators > DOT_BOUND):
            raise ArithmeticError("binary INTT accumulator exceeds the full-dot bound")
        output_lanes, reduction = _redc32_core(accumulators)
        output = output_lanes.T.reshape(256).tolist()
        if trace is not None:
            trace(stage, dict(
                operation="rram_binary_vmm", transform="intt", reduction="REDC32",
                weight_artifact="rram_intt_montgomery_weights/" + WEIGHTS_NAME,
                physical_orientation="input_rows_output_columns", lane_order=["even", "odd"],
                pulse_levels=[0, 1], weight_encoding="unsigned12", input_representation="canonical_unsigned12",
                input_domain="ordinary", output_domain="ordinary", weight_domain="R_times_normalized_inverse",
                input_bit_slots=12, weight_bit_planes=12, slot_weights=BIT_WEIGHTS.copy(),
                input_lanes=lanes, input_bitplanes=input_bits,
                tile_input_offsets=TILE_INPUT_OFFSETS.copy(), tile_output_offsets=TILE_OUTPUT_OFFSETS.copy(),
                tile_counts_axes=["lane", "input_slot", "weight_bit", "tile", "output_column_in_tile"],
                tile_counts=tile_counts, merged_counts=merged, shift_amounts=shifts,
                weighted_terms=weighted_terms, per_input_slot_accumulators=per_slot,
                weighted_accumulators=accumulators, output_lanes=output_lanes, output=output,
                inverse_normalization_in_weights=True, normalization_count=1,
                radix_bits=RADIX_BITS, radix_compensation_in_weights=True,
                montgomery_correction=reduction["correction_m"], redc_quotient=reduction["quotient"],
                redc_subtract_q=reduction["subtract_q"], raw_tile_count_max=int(tile_counts.max()),
                merged_count_max=int(merged.max()), binary_tile_read_vectors=2 * 12 * 12 * 4,
                output_representation="canonical", output_is_raw=False,
                count_model="ideal binary counts plus exact REDC32; no analog or cycle model"))
            trace(stage + ".redc32", reduction)
        return output
