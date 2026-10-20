"""Self-contained exact Saber auxiliary NTT and binary RRAM polynomial engine.

The forward weights W are ordinary residues. Only inverse weights are fused:
K = (2**32 V) mod p, where V already includes inverse(64,p). Integer binary
counts are reconstructed before REDC32 returns ordinary inverse outputs.
Four coefficient lanes share 64x64 physical input-row/output-column planes.
This is a functional integer model; logical reads do not specify hardware time.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
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
MODULI = (3329, 7681)
N, LANES, SIZE = 256, 4, 64
R32, LOW32 = 1 << 32, (1 << 32) - 1
CRT_MODULUS = 3329 * 7681
SOURCE_PATH = ROOT / "matrices/source_weights.npz"
SOURCE_SHA256 = "c8100dc064e6defb914e361044afc62fdb95834fc0d6902cd65a6184a0fb9bfe"
MATRIX_PATH = ROOT / "matrices/saber_auxiliary_matrices.npz"
WEIGHT_PATH = ROOT / "rram_binary/saber_binary_weights.npz"
PROVENANCE_PATH = ROOT / "rram_binary/provenance.json"
_BYTE_COUNTS = np.array([x.bit_count() for x in range(256)], dtype=np.uint8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _bitrev6(value):
    return int(f"{value:06b}"[::-1], 2)


def _deterministic_npz(arrays):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            body = io.BytesIO()
            np.lib.format.write_array(body, np.asarray(arrays[name]), version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, body.getvalue())
    return output.getvalue()


def _source_matrices():
    # Rebuild the mathematical source locally if a source-only checkout omits
    # generated matrices. The deterministic file hash below still binds data.
    if not SOURCE_PATH.exists():
        from matrices.source_build_weights import main as build_source
        build_source()
    require(_sha(SOURCE_PATH.read_bytes()) == SOURCE_SHA256, "Original Saber matrix source hash mismatch")
    matrix_arrays = {"moduli": np.array(MODULI, dtype=np.int64),
                     "R32": np.array(R32, dtype=np.uint64),
                     "root_exponents": np.array([2 * _bitrev6(i) + 1 for i in range(SIZE)], dtype=np.int64)}
    weight_arrays = {"moduli": matrix_arrays["moduli"], "R32": matrix_arrays["R32"]}
    branches = []
    with np.load(SOURCE_PATH, allow_pickle=False) as source:
        for index, p in enumerate(MODULI):
            psi = int(source["psi"][index])
            gamma = np.array([pow(psi, 2 * _bitrev6(i) + 1, p) for i in range(SIZE)], dtype=np.int64)
            inverse64 = pow(SIZE, -1, p)
            W = source[f"W_{p}"].copy()
            V = source[f"Winv_{p}"].copy()
            expected_w = np.array([[pow(int(g), j, p) for j in range(SIZE)] for g in gamma], dtype=np.int64)
            expected_v = np.array([[inverse64 * pow(int(g), -j, p) % p for g in gamma]
                                   for j in range(SIZE)], dtype=np.int64)
            require(np.array_equal(W, expected_w) and np.array_equal(V, expected_v), "Source roots/order disagree with W/V")
            require(np.array_equal((V @ W) % p, np.eye(SIZE, dtype=np.int64)), "V W is not identity")
            require(pow(psi, SIZE, p) == p - 1 and pow(psi, 2 * SIZE, p) == 1, "Root is not exact order 128")
            K = (V * (R32 % p)) % p
            bits = (p - 1).bit_length()
            p_dash = (-pow(p, -1, R32)) & LOW32
            for name, values in (("W", W), ("V", V), ("K", K), ("gamma", gamma)):
                matrix_arrays[f"{name}_{p}"] = values
            matrix_arrays[f"psi_{p}"] = np.array(psi, dtype=np.int64)
            matrix_arrays[f"inverse64_{p}"] = np.array(inverse64, dtype=np.int64)
            matrix_arrays[f"p_dash_{p}"] = np.array(p_dash, dtype=np.uint64)
            for direction, matrix in (("ntt", W), ("intt", K)):
                physical = matrix.T
                planes = ((physical[None, :, :] >> np.arange(bits, dtype=np.int64)[:, None, None]) & 1).astype(np.uint8)
                rebuilt = (planes.astype(np.int64) << np.arange(bits, dtype=np.int64)[:, None, None]).sum(axis=0)
                require(np.array_equal(rebuilt, physical), "Binary plane reconstruction failed")
                weight_arrays[f"{direction}_{p}_physical_planes"] = planes
            branches.append({"modulus": p, "bits": bits, "psi": psi, "inverse64": inverse64,
                             "p_dash": p_dash, "R32_mod_p": R32 % p,
                             "max_transform_accumulator": SIZE * (p - 1) ** 2,
                             "accumulator_below_R32": SIZE * (p - 1) ** 2 < R32,
                             "accumulator_below_pR32": SIZE * (p - 1) ** 2 < p * R32})
    return matrix_arrays, weight_arrays, branches


def export_weights():
    """Create deterministic self-contained artifacts; reject differing data files."""
    matrix_arrays, weight_arrays, branches = _source_matrices()
    files = {MATRIX_PATH: _deterministic_npz(matrix_arrays),
             WEIGHT_PATH: _deterministic_npz(weight_arrays)}
    for p in MODULI:
        for direction in ("ntt", "intt"):
            for bit, plane in enumerate(weight_arrays[f"{direction}_{p}_physical_planes"]):
                payload = "\n".join(",".join(map(str, row.tolist())) for row in plane) + "\n"
                files[ROOT / "rram_binary" / f"{direction}_{p}_bit{bit:02d}.csv"] = payload.encode("ascii")
    for path, data in files.items():
        require(not path.exists() or path.read_bytes() == data, "Existing Saber weight artifact differs: " + str(path))
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    manifest = {
        "format_version": 1, "kind": "Exact binary RRAM Saber auxiliary NTT/INTT; INTT-only Montgomery fusion",
        "source": {"path": "matrices/source_weights.npz", "sha256": SOURCE_SHA256,
                   "source_provenance": "matrices/source_provenance.json",
                   "source_generator": "matrices/source_build_weights.py"},
        "source_supporting_files": {str(p.relative_to(ROOT)): _sha(p.read_bytes())
                                    for p in (ROOT / "matrices/source_provenance.json", ROOT / "matrices/source_build_weights.py")},
        "code": {str(path.relative_to(ROOT)): _sha(path.read_bytes()) for path in
                 (Path(__file__), ROOT / "generate_auxiliary_weights.py") if path.exists()},
        "branches": branches, "lanes": 4, "lane_length": 64,
        "lane_layout": "polynomial[4*j+r] -> lane r position j; r=0..3",
        "gamma": "gamma_i = psi^(2*bitrev6(i)+1) mod p",
        "forward": "W[i,j]=gamma_i^j mod p; ordinary input and ordinary weights; sum then mod p",
        "inverse": "V[j,i]=inverse64*gamma_i^(-j) mod p; K=(2^32*V) mod p; sum then REDC32",
        "inverse_normalization_count": 1,
        "normalization": "inverse64 is already in V and hence K; never multiply inverse64 after REDC32",
        "redc32": "m=((T & (2^32-1))*p_dash)&(2^32-1); u=(T+m*p)>>32; result=u-p if u>=p else u",
        "redc32_arithmetic": "Public reducer accepts integer arrays with 0<=T<p*2^32. Low-word multiplication uses uint64 after masking T. The numerator is <2*p*2^32<2^46. Actual transform accumulators additionally satisfy T<2^32.",
        "physical_orientation": "plane[weight_bit,input_row,output_column] = bit(W.T) or bit(K.T)",
        "stored_values": [0, 1], "weight_planes": {"3329": {"ntt": 12, "intt": 12}, "7681": {"ntt": 13, "intt": 13}},
        "total_csv_planes": 50, "plane_shape": [64, 64], "lanes_reuse_same_weights": True,
        "input_encoding": "Canonical auxiliary residues in fixed 12 or 13 unsigned input-bit slots; zero slots are retained",
        "binary_compute": "Pack64 rows -> binary AND/popcount -> int64 shift/add -> mod p for NTT or uint64 REDC32 for INTT",
        "quartic_product": "64 products modulo X^4-gamma_i; sum all rank terms before each branch INTT",
        "crt": {"modulus": CRT_MODULUS, "canonical_formula": "r0+3329*(((r1-r0)*inverse(3329,7681)) mod7681)",
                "center_before_target_reduction": True, "target_moduli": [8192, 1024],
                "bound": "B=256*max(abs(center_target(left)))*sum_rows(max(abs(center8192(right)))); require 2B<3329*7681",
                "right_support_required": [-5, 5]},
        "artifacts": {str(path.relative_to(ROOT)): {"bytes": len(data), "sha256": _sha(data)} for path, data in files.items()},
        "scope": "Mathematical binary counts and exact arithmetic; no physical measurements or timing inference",
    }
    PROVENANCE_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@lru_cache(maxsize=1)
def load_artifacts():
    if not PROVENANCE_PATH.exists() or not MATRIX_PATH.exists() or not WEIGHT_PATH.exists():
        export_weights()
    manifest = json.loads(PROVENANCE_PATH.read_text())
    require(manifest["source"]["sha256"] == SOURCE_SHA256, "Unexpected source in Saber weight provenance")
    require(_sha(SOURCE_PATH.read_bytes()) == SOURCE_SHA256, "Source matrix hash mismatch")
    for relative, record in manifest["artifacts"].items():
        data = (ROOT / relative).read_bytes()
        require(len(data) == record["bytes"] and _sha(data) == record["sha256"], "Artifact integrity failure: " + relative)
    with np.load(MATRIX_PATH, allow_pickle=False) as archive:
        matrices = {k: archive[k].copy() for k in archive.files}
    with np.load(WEIGHT_PATH, allow_pickle=False) as archive:
        weights = {k: archive[k].copy() for k in archive.files}
    for p in MODULI:
        bits = (p - 1).bit_length()
        for direction, name in (("ntt", "W"), ("intt", "K")):
            planes = weights[f"{direction}_{p}_physical_planes"]
            require(planes.shape == (bits, SIZE, SIZE) and planes.dtype == np.uint8 and np.all(planes <= 1),
                    "Physical weights must be correctly shaped binary uint8 planes")
            # Artifact validation only; binary execution below never evaluates this matrix.
            rebuilt = (planes.astype(np.int64) << np.arange(bits, dtype=np.int64)[:, None, None]).sum(axis=0)
            require(np.array_equal(rebuilt, matrices[f"{name}_{p}"].T), "Physical plane orientation/value mismatch")
    for array in list(matrices.values()) + list(weights.values()):
        array.flags.writeable = False
    return matrices, weights


def _pack_rows(bits):
    require(bits.shape[-1] == SIZE, "Exactly64 rows are packed into one uint64 word")
    packed = np.packbits(bits, axis=-1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").reshape(bits.shape[:-1]).astype(np.uint64)


def popcount_u64(words):
    words = np.ascontiguousarray(words, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(words)
    raw = words.view(np.uint8).reshape(words.shape + (8,))
    return _BYTE_COUNTS[raw].sum(axis=-1, dtype=np.uint16).astype(np.uint8)


def redc32(values, p):
    """Ordinary output from INTT fused K; explicitly unsigned low-word arithmetic."""
    require(isinstance(p, Integral) and not isinstance(p, (bool, np.bool_)) and int(p) in MODULI,
            "Unsupported REDC32 modulus")
    p = int(p)
    source = np.asarray(values)
    require(source.dtype.kind in ("i", "u"), "REDC32 input must have an integer dtype; floats and booleans are invalid")
    require(np.all(source >= 0) and np.all(source < p * R32), "REDC32 requires0<=T<p*2^32")
    # Validate before conversion; uint64 supports the complete standard REDC
    # interval. Mask BEFORE multiplication so its low-word product is <2^64.
    t = source.astype(np.uint64, copy=False)
    p_dash = np.uint64((-pow(p, -1, R32)) & LOW32)
    m = ((t & np.uint64(LOW32)) * p_dash) & np.uint64(LOW32)
    u = (t + m * np.uint64(p)) >> np.uint64(32)
    # Selection occurs while the minuend is unsigned but never underflows:
    # subtract p only where u>=p, by subtracting either p or zero.
    output = u - (u >= np.uint64(p)).astype(np.uint64) * np.uint64(p)
    require(np.all(output < p), "REDC32 output is not canonical")
    return output.astype(np.int64), {"T": t, "m": m, "u": u, "p_dash": int(p_dash)}


def _integer_vector(poly):
    values = list(poly)
    require(len(values) == N and all(isinstance(x, Integral) for x in values), "Expected256 integer coefficients")
    return [int(x) for x in values]


def _center_rows(rows, modulus):
    # Python integer modulo before int64 conversion prevents conversion overflow.
    canonical = np.array([[x % modulus for x in _integer_vector(row)] for row in rows], dtype=np.int64)
    return np.where(canonical >= (modulus + 1) // 2, canonical - modulus, canonical)


class SaberPolynomialEngine:
    """Exact bounded polynomial dot products in Z_8192 or Z_1024[X]/(X^256+1)."""

    def __init__(self, backend="matrix", trace: Callable[[str, dict], None] | None = None):
        require(backend in ("matrix", "rram_binary"), "Backend must be matrix or rram_binary")
        require(trace is None or callable(trace), "trace must be callable or None")
        self.backend, self.trace = backend, trace
        self.matrices, self.weights = load_artifacts()
        self.packed_tiles = {(p, direction): _pack_rows(self.weights[f"{direction}_{p}_physical_planes"].transpose(0, 2, 1))
                             for p in MODULI for direction in ("ntt", "intt")}
        self.reset_counts()

    def reset_counts(self):
        self.counts = Counter({"ntt": 0, "intt": 0, "polynomial_dots": 0,
                               "binary_tile_read_vectors": 0, "binary_column_popcounts": 0,
                               "redc32_coefficients": 0})

    def _emit(self, stage, **values):
        if self.trace is not None:
            self.trace(stage, deepcopy(values))

    def _transform(self, source, p, inverse, stage):
        require(p in MODULI, "Unsupported auxiliary modulus")
        source = [value % p for value in _integer_vector(source)]
        direction, bits = ("intt" if inverse else "ntt"), (p - 1).bit_length()
        lanes = np.array(source, dtype=np.int64).reshape(SIZE, LANES).T
        binary_details = None
        self.counts[direction] += 1
        self.counts[f"{direction}_{p}"] += 1
        if self.backend == "matrix":
            weight_name = "K" if inverse else "W"
            accumulators = (self.matrices[f"{weight_name}_{p}"] @ lanes.T).T
            self._emit(stage + ".matrix", operation="matrix_vmm", transform=direction,
                       auxiliary_modulus=p, modulus=p, input_lanes=lanes, accumulator_lanes=accumulators,
                       weight=weight_name, physical_orientation="input_rows_output_columns")
        else:
            input_bits = ((lanes[:, None, :] >> np.arange(bits, dtype=np.int64)[None, :, None]) & 1).astype(np.uint8)
            input_words = _pack_rows(input_bits)
            intersections = input_words[:, :, None, None] & self.packed_tiles[(p, direction)][None, None, :, :]
            tile_counts = popcount_u64(intersections)
            require(tile_counts.shape == (LANES, bits, bits, SIZE) and int(tile_counts.max()) <= SIZE,
                    "Binary tile count shape/range is invalid")
            shifts = np.arange(bits, dtype=np.int64)[:, None] + np.arange(bits, dtype=np.int64)[None, :]
            weighted = tile_counts.astype(np.int64) << shifts[None, :, :, None]
            per_input = weighted.sum(axis=2, dtype=np.int64)
            accumulators = per_input.sum(axis=1, dtype=np.int64)
            self.counts["binary_tile_read_vectors"] += LANES * bits * bits
            self.counts["binary_column_popcounts"] += LANES * bits * bits * SIZE
            if self.trace is not None:
                binary_details = dict(operation="rram_binary_vmm", transform=direction, direction=direction,
                    auxiliary_modulus=p, modulus=p, input_lanes=lanes,
                    input_bitplanes=input_bits, input_bit_slots=bits, weight_bit_planes=bits,
                    tile_counts=tile_counts, tile_counts_axes=["lane", "input_bit", "weight_bit", "output_column"],
                    per_input_slot_accumulators=per_input, accumulator_lanes=accumulators,
                    weight="K" if inverse else "W", lane_order=[0, 1, 2, 3],
                    physical_orientation="input_rows_output_columns", raw_tile_count_max=int(tile_counts.max()))
        bound = SIZE * (p - 1) ** 2
        require(bound < R32 and bound < p * R32, "Actual VMM accumulator must be below R32 and p*R32")
        require(np.all(accumulators >= 0) and np.all(accumulators <= bound), "Transform accumulator bound exceeded")
        if inverse:
            output_lanes, redc = redc32(accumulators, p)
            self.counts["redc32_coefficients"] += N
            self._emit(stage + ".redc32", operation="redc32", auxiliary_modulus=p, modulus=p,
                       inverse64_already_in_weights=True, normalization_count=1,
                       output_lanes=output_lanes, **redc)
        else:
            output_lanes = accumulators % p
        output = output_lanes.T.reshape(N).tolist()
        if binary_details is not None:
            self._emit(stage + ".rram_binary", input=source, output=output,
                       output_representation="ordinary_canonical", **binary_details)
        self._emit(stage, operation=direction, transform=direction, backend=self.backend,
                   auxiliary_modulus=p, modulus=p, direction=direction, input=source, output=output,
                   input_representation="ordinary_canonical", output_representation="ordinary_canonical",
                   inverse_normalization_count=1 if inverse else 0,
                   inverse_weight_fusion="R32*V then REDC32" if inverse else None)
        return output

    def ntt(self, poly, modulus, stage="ntt"):
        """Transform all 256 coefficients in four interleaved 64-element lanes.

        ``modulus`` is 3329 or 7681. Values are returned as canonical ordinary
        residues, with coefficient r of component i at output index 4*i+r.
        """
        return self._transform(poly, modulus, False, stage)

    def intt(self, poly, modulus, stage="intt"):
        """Invert one auxiliary transform; normalization is applied exactly once.

        The stored inverse is K = (2**32 * V) mod p, with 64**-1 already in V.
        One REDC32 after integer accumulation cancels the Montgomery factor.
        """
        return self._transform(poly, modulus, True, stage)

    def _quartic_dot(self, left, right, p, stage):
        a, b = np.array(left, dtype=np.int64).reshape(-1, SIZE, LANES), np.array(right, dtype=np.int64).reshape(-1, SIZE, LANES)
        require(a.shape == b.shape, "Transformed dot operands have different shapes")
        require(4 * len(a) * (p - 1) ** 3 < np.iinfo(np.int64).max, "Quartic accumulator int64 guard failed")
        gamma = self.matrices[f"gamma_{p}"]
        accumulated = np.zeros((SIZE, LANES), dtype=np.int64)
        for i in range(LANES):
            for j in range(LANES):
                term = (a[:, :, i] * b[:, :, j]).sum(axis=0, dtype=np.int64)
                if i + j >= LANES:
                    term = term * gamma
                accumulated[:, (i + j) % LANES] += term
        output = (accumulated % p).reshape(N).tolist()
        self.counts["quartic_base_products"] += len(a) * SIZE
        self._emit(stage, operation="quartic_ntt_dot", auxiliary_modulus=p,
                   left_ntt=a, right_ntt=b, gamma=gamma, integer_accumulated=accumulated,
                   output=output, component_ring="F_p[X]/(X^4-gamma_i)")
        return output

    def dot(self, left, right, modulus, stage="poly_dot"):
        """Return a negacyclic polynomial dot product modulo 8192 or 1024.

        Both arguments are equal-length sequences of 256-coefficient
        polynomials. The right operand must encode a small CBD secret modulo
        8192. Left values are centered in the requested target ring; right
        values are always centered modulo 8192. Two prime-domain quartic
        products recover an integer sum by centered CRT before target reduction.
        The uniqueness bound 2*B < 3329*7681 is checked before computing.
        """
        require(modulus in (8192, 1024), "Saber target modulus must be8192 or1024")
        left, right = list(left), list(right)
        require(len(left) == len(right) and len(left) > 0, "Dot operands need equal positive rank")
        left_centered = _center_rows(left, modulus)
        right_centered = _center_rows(right, 8192)
        right_maxima = np.max(np.abs(right_centered), axis=1)
        require(np.all(right_maxima <= 5), "Right operand must have centered mod8192 CBD support within[-5,5]")
        max_left = int(np.max(np.abs(left_centered)))
        bound = N * max_left * sum(int(v) for v in right_maxima)
        require(2 * bound < CRT_MODULUS, "Centered CRT uniqueness bound failed: require2B<3329*7681")
        self.counts["polynomial_dots"] += 1
        self._emit(stage + ".inputs", operation="polynomial_dot_inputs", target_modulus=modulus,
                   left_centered=left_centered, right_centered=right_centered,
                   right_center_modulus=8192, integer_bound=bound, crt_modulus=CRT_MODULUS)
        residues = []
        for p in MODULI:
            ahat = [self.ntt(poly, p, f"{stage}.mod{p}.left_ntt[{i}]") for i, poly in enumerate(left_centered)]
            bhat = [self.ntt(poly, p, f"{stage}.mod{p}.right_ntt[{i}]") for i, poly in enumerate(right_centered)]
            product = self._quartic_dot(ahat, bhat, p, f"{stage}.mod{p}.quartic_dot")
            residues.append(np.array(self.intt(product, p, f"{stage}.mod{p}.intt"), dtype=np.int64))
        r0, r1 = residues
        # Center the CRT integer before reducing to a power of two: the CRT
        # product is odd, so canonical CRT followed directly by mod 8192/1024
        # would give the wrong answer for negative integer coefficients.
        canonical = r0 + MODULI[0] * (((r1 - r0) * pow(MODULI[0], -1, MODULI[1])) % MODULI[1])
        centered = np.where(canonical >= (CRT_MODULUS + 1) // 2, canonical - CRT_MODULUS, canonical)
        require(np.all(np.abs(centered) <= bound), "CRT reconstruction exceeded the proven signed integer bound")
        output = (centered % modulus).tolist()
        self._emit(stage + ".crt", operation="crt_centered", target_modulus=modulus,
                   residues=np.stack(residues), auxiliary_moduli=list(MODULI), canonical_crt=canonical,
                   centered_crt=centered, integer_bound=bound, output=output,
                   center_before_target_reduction=True)
        self._emit(stage, operation="polynomial_dot", backend=self.backend, target_modulus=modulus,
                   rank=len(left), integer_bound=bound, output=output)
        return output
