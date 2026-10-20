#!/usr/bin/env python3
"""Generate and independently check logical Saber auxiliary-NTT weights.

Writes source_weights.npz and source_provenance.json beside this script.  The matrices
are mathematical, ordinary-domain residues, NOT measured RRAM conductances.
The checks establish arithmetic identities, NOT complete Saber KEM validation.
Requires Python 3.9+ and NumPy; run with ``python3 matrices/source_build_weights.py`` from the module directory.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
import json
from math import isqrt
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
import numpy as np


HERE = Path(__file__).resolve().parent
MODULI = (3329, 7681)
TARGET_Q = 8192
TARGET_P = 1024
N = 256
LANES = 4
M = N // LANES
ROOT_ORDER = 2 * M
SEED = 20261005
FIELD_RANDOM_CASES = 64
MATRIX_RANDOM_CASES = 16
FAMILIES = (("LightSaber", 2, 10, 3), ("Saber", 3, 8, 4), ("FireSaber", 4, 6, 6))


def bitrev6(value: int) -> int:
    return int(f"{value:06b}"[::-1], 2)


def require_prime(p: int) -> None:
    assert p > 2 and p % 2
    assert all(p % d for d in range(3, isqrt(p) + 1, 2)), f"Not prime: {p}"


def choose_root(p: int) -> tuple[int, int]:
    """First base >=2 yielding exact order 128 after the stated exponent."""
    assert (p - 1) % ROOT_ORDER == 0
    for base in range(2, p):
        psi = pow(base, (p - 1) // ROOT_ORDER, p)
        if pow(psi, M, p) == p - 1 and pow(psi, ROOT_ORDER, p) == 1:
            return base, psi
    raise AssertionError("No root found")


def same(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(actual, expected):
        index = tuple(np.argwhere(actual != expected)[0])
        raise AssertionError(f"{label}: mismatch at {index}")


def centered(values: np.ndarray, modulus: int) -> np.ndarray:
    """Even modulus: [-m/2,m/2-1]; odd modulus: [-(m-1)/2,(m-1)/2]."""
    values = np.asarray(values, dtype=np.int64) % modulus
    return np.where(values >= (modulus + 1) // 2, values - modulus, values)


def matrices(p: int) -> dict:
    require_prime(p)
    base, psi = choose_root(p)
    gamma = np.array([pow(psi, 2 * bitrev6(i) + 1, p) for i in range(M)], dtype=np.int64)
    inverse_m = pow(M, -1, p)
    w = np.array([[pow(int(g), j, p) for j in range(M)] for g in gamma], dtype=np.int64)
    winv = np.array([[inverse_m * pow(int(g), -j, p) % p for g in gamma]
                     for j in range(M)], dtype=np.int64)
    return {"p": p, "base": base, "psi": psi, "gamma": gamma,
            "inverse64": inverse_m, "W": w, "Winv": winv}


def apply(matrix: np.ndarray, poly: np.ndarray, p: int) -> np.ndarray:
    """Each of four residue-class lanes shares this one 64x64 matrix."""
    poly = np.asarray(poly, dtype=np.int64) % p
    return ((matrix @ poly.reshape(M, LANES)) % p).reshape(N)


def butterfly(poly: np.ndarray, p: int, psi: int, inverse: bool = False) -> np.ndarray:
    """Independent six-stage arithmetic implementation, stopping at length 4."""
    values = [int(x) % p for x in poly]
    lengths = (4, 8, 16, 32, 64, 128) if inverse else (128, 64, 32, 16, 8, 4)
    inverse_two = pow(2, -1, p)
    for length in lengths:
        groups = N // (2 * length)
        for group in range(groups):
            zeta = pow(psi, bitrev6(groups + group), p)
            izeta = pow(zeta, -1, p) if inverse else None
            for j in range(group * 2 * length, group * 2 * length + length):
                a, b = values[j], values[j + length]
                if inverse:
                    values[j] = (a + b) * inverse_two % p
                    values[j + length] = (a - b) * inverse_two * izeta % p
                else:
                    t = zeta * b % p
                    values[j] = (a + t) % p
                    values[j + length] = (a - t) % p
    return np.asarray(values, dtype=np.int64)


def quartic_products(a: np.ndarray, b: np.ndarray, gamma: np.ndarray, p: int) -> np.ndarray:
    """64 independent products in F_p[X]/(X^4-gamma_i)."""
    out = np.empty((M, LANES), dtype=np.int64)
    for i, (aa, bb) in enumerate(zip(a.reshape(M, LANES), b.reshape(M, LANES))):
        convolution = np.convolve(aa, bb) % p
        out[i] = convolution[:LANES]
        out[i, :LANES - 1] += gamma[i] * convolution[LANES:]
    return (out % p).reshape(N)


def integer_negacyclic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Direct integer convolution, independent of all auxiliary-modulus code."""
    convolution = np.convolve(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    out = convolution[:N].copy()
    out[:N - 1] -= convolution[N:]
    return out


def crt_recover(residues: list[np.ndarray]) -> np.ndarray:
    p0, p1 = MODULI
    r0, r1 = residues
    canonical = r0 + p0 * (((r1 - r0) * pow(p0, -1, p1)) % p1)
    return centered(canonical, p0 * p1)


def multiply_sum(left: np.ndarray, right: np.ndarray, branches: list[dict]) -> np.ndarray:
    """Sum all polynomial products BEFORE each branch's inverse transform."""
    residues = []
    for branch in branches:
        p, w, wi, gamma = (branch[k] for k in ("p", "W", "Winv", "gamma"))
        accumulated = np.zeros(N, dtype=np.int64)
        for a, b in zip(left, right):
            ahat, bhat = apply(w, a, p), apply(w, b, p)
            accumulated = (accumulated + quartic_products(ahat, bhat, gamma, p)) % p
        residues.append(apply(wi, accumulated, p))
    return crt_recover(residues)


def bitplanes(matrix: np.ndarray, bits: int) -> np.ndarray:
    return ((matrix[None, :, :] >> np.arange(bits, dtype=np.int64)[:, None, None]) & 1).astype(np.uint8)


def test_field(branch: dict, rng: np.random.Generator) -> dict:
    p, psi, gamma, w, wi = (branch[k] for k in ("p", "psi", "gamma", "W", "Winv"))
    assert len(set(gamma.tolist())) == M
    assert all(pow(int(g), M, p) == p - 1 for g in gamma)
    identity = np.eye(M, dtype=np.int64)
    same(wi @ w % p, identity, f"{p}: Winv W")
    same(w @ wi % p, identity, f"{p}: W Winv")
    assert N * (p - 1) ** 2 < np.iinfo(np.int64).max
    for index in range(N):
        unit = np.zeros(N, dtype=np.int64)
        unit[index] = 1
        same(apply(w, unit, p), butterfly(unit, p, psi), f"{p}: forward basis {index}")
        same(apply(wi, unit, p), butterfly(unit, p, psi, True), f"{p}: inverse basis {index}")
    zero = np.zeros(N, dtype=np.int64)
    one = np.ones(N, dtype=np.int64)
    first, last = zero.copy(), zero.copy()
    first[0], last[-1] = 1, 1
    alternating = np.arange(N, dtype=np.int64) % 2 * (p - 1)
    pairs = [(zero, one), (one, one), (first, last), (last, last),
             (one * (p - 1), one * (p - 1)), (alternating, one * (p - 1))]
    pairs += [(rng.integers(0, p, N), rng.integers(0, p, N))
              for _ in range(FIELD_RANDOM_CASES)]
    for index, (a, b) in enumerate(pairs):
        ahat, bhat = apply(w, a, p), apply(w, b, p)
        same(ahat, butterfly(a, p, psi), f"{p}: forward {index}")
        same(apply(wi, ahat, p), a, f"{p}: roundtrip {index}")
        product = quartic_products(ahat, bhat, gamma, p)
        recovered = apply(wi, product, p)
        same(recovered, integer_negacyclic(a, b) % p, f"{p}: quartic product {index}")
        same(recovered, butterfly(product, p, psi, True), f"{p}: inverse {index}")
    for name, matrix in (("W", w), ("Winv", wi)):
        bits = (p - 1).bit_length()
        planes = bitplanes(matrix, bits)
        reconstructed = np.sum(planes.astype(np.int64) << np.arange(bits)[:, None, None], axis=0)
        same(reconstructed, matrix, f"{p}: {name} bitplanes")
    return {"result": "PASS", "both_matrix_inverse_identities": True,
            "forward_unit_polynomials": N, "inverse_unit_polynomials": N,
            "random_field_product_pairs": FIELD_RANDOM_CASES,
            "structured_field_product_pairs": 6,
            "roundtrip_quartic_product_and_independent_butterflies": len(pairs),
            "binary_weight_reconstruction": "PASS"}


def cbd(rng: np.random.Generator, shape: tuple[int, ...], mu: int) -> np.ndarray:
    bits = rng.integers(0, 2, shape + (mu,), dtype=np.int64)
    return bits[..., :mu // 2].sum(axis=-1) - bits[..., mu // 2:].sum(axis=-1)


def test_family(name: str, rank: int, mu: int, et: int,
                branches: list[dict], rng: np.random.Generator) -> dict:
    small = mu // 2
    bound = N * (TARGET_Q // 2) * rank * small
    assert MODULI[0] * MODULI[1] > 2 * bound
    shape = (rank, rank, N)
    cases = []
    for sign in (1, -1):
        cases.append((f"attain_{'negative' if sign == 1 else 'positive'}_bound",
                      np.full(shape, -TARGET_Q // 2, dtype=np.int64),
                      np.full((rank, N), sign * small, dtype=np.int64)))
    cases.append(("zero_secret", np.full(shape, -TARGET_Q // 2, dtype=np.int64),
                  np.zeros((rank, N), dtype=np.int64)))
    alternating = np.where(np.arange(N) % 2, TARGET_Q // 2 - 1, -TARGET_Q // 2)
    cases.append(("alternating_extremes", np.broadcast_to(alternating, shape).copy(),
                  np.broadcast_to(np.where(np.arange(N) % 2, small, -small), (rank, N)).copy()))
    impulse_a, impulse_s = np.zeros(shape, dtype=np.int64), np.zeros((rank, N), dtype=np.int64)
    impulse_a[:, :, N - 1], impulse_s[:, 1] = -TARGET_Q // 2, small
    cases.append(("negacyclic_wrap_X256", impulse_a, impulse_s))
    for i in range(MATRIX_RANDOM_CASES):
        a = centered(rng.integers(0, TARGET_Q, shape), TARGET_Q)
        s = cbd(rng, (rank, N), mu)
        cases.append((f"uniform_matrix_CBD_secret_{i}", a, s))
    cases.append(("random_supported_extremes",
                  rng.choice(np.array([-TARGET_Q // 2, TARGET_Q // 2 - 1]), shape),
                  rng.choice(np.array([-small, small]), (rank, N))))
    maximum = 0
    matrix_outputs = 0
    for label, a, s in cases:
        for transpose in (False, True):
            operator = np.swapaxes(a, 0, 1) if transpose else a
            for row in range(rank):
                exact = sum((integer_negacyclic(operator[row, j], s[j]) for j in range(rank)),
                            np.zeros(N, dtype=np.int64))
                assert np.max(np.abs(exact)) <= bound
                recovered = multiply_sum(operator[row], s, branches)
                same(recovered, exact, f"{name}: integer {label}, transpose={transpose}, row={row}")
                same(recovered % TARGET_Q, exact % TARGET_Q, f"{name}: original q")
                maximum = max(maximum, int(np.max(np.abs(exact))))
                if label == "attain_negative_bound":
                    assert int(recovered[-1]) == -bound
                if label == "attain_positive_bound":
                    assert int(recovered[-1]) == bound
                matrix_outputs += 1
    assert maximum == bound
    # Saber also uses an inner product after rounding to p=1024.
    p_bound = N * (TARGET_P // 2) * rank * small
    inner_cases = [(np.full((rank, N), -TARGET_P // 2, dtype=np.int64),
                    np.full((rank, N), sign * small, dtype=np.int64)) for sign in (1, -1)]
    inner_cases += [(centered(rng.integers(0, TARGET_P, (rank, N)), TARGET_P),
                     cbd(rng, (rank, N), mu)) for _ in range(MATRIX_RANDOM_CASES)]
    for index, (a, s) in enumerate(inner_cases):
        exact = sum((integer_negacyclic(a[j], s[j]) for j in range(rank)), np.zeros(N, dtype=np.int64))
        assert np.max(np.abs(exact)) <= p_bound
        recovered = multiply_sum(a, s, branches)
        same(recovered, exact, f"{name}: inner product integer {index}")
        same(recovered % TARGET_P, exact % TARGET_P, f"{name}: original p")
    return {"result": "PASS", "rank": rank, "mu": mu, "epsilon_T": et,
            "secret_support": [-small, small], "large_coefficients": [-4096, 4095],
            "whole_matrix_vector_integer_bound": bound,
            "twice_bound": 2 * bound, "crt_modulus": MODULI[0] * MODULI[1],
            "maximum_observed_absolute_coefficient": maximum,
            "positive_and_negative_worst_case_attained": True,
            "matrix_vector_test_cases": len(cases), "both_matrix_orientations_tested": True,
            "verified_output_polynomials_mod8192": matrix_outputs,
            "verified_output_polynomials_mod1024": len(inner_cases),
            "random_matrix_and_inner_product_cases_each": MATRIX_RANDOM_CASES,
            "test_kind": "exact integer and modular polynomial arithmetic; not full KEM"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_npz(arrays: dict) -> bytes:
    """Encode named arrays without timestamps so regeneration is byte-exact."""
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


def main() -> None:
    rng = np.random.default_rng(SEED)
    branches = [matrices(p) for p in MODULI]
    field_results = {str(b["p"]): test_field(b, rng) for b in branches}
    family_results = {name: test_family(name, rank, mu, et, branches, rng)
                      for name, rank, mu, et in FAMILIES}
    arrays = {"moduli": np.array(MODULI, dtype=np.int64),
              "W": np.stack([b["W"] for b in branches]),
              "Winv": np.stack([b["Winv"] for b in branches]),
              "gamma": np.stack([b["gamma"] for b in branches]),
              "psi": np.array([b["psi"] for b in branches], dtype=np.int64),
              "inverse64": np.array([b["inverse64"] for b in branches], dtype=np.int64),
              "root_exponents": np.array([2 * bitrev6(i) + 1 for i in range(M)], dtype=np.int64)}
    for b in branches:
        p, bits = b["p"], (b["p"] - 1).bit_length()
        for name in ("W", "Winv"):
            arrays[f"{name}_{p}"] = b[name]
            arrays[f"physical_{name}_{p}"] = b[name].T
            arrays[f"bitplanes_{name}_{p}"] = bitplanes(b[name], bits)
    artifact = HERE / "source_weights.npz"
    artifact.write_bytes(deterministic_npz(arrays))
    metadata = {
        "created_for": "Self-contained Saber auxiliary-transform reference",
        "status": "algorithm-generated logical matrices; arithmetic checks passed",
        "data_kind": "mathematical residues and binary weights, not measurements",
        "not_claimed": ["RRAM conductance measurements, write success, or read-noise statistics",
                        "Experimentally demonstrated latency, energy, or hardware correctness",
                        "Full Saber KEM correctness, security verification, or standard KAT conformance",
                        "Exact root/ordering match to a particular third-party implementation",
                        "All weights fit on the existing four-array chip"],
        "sources": [
            {"title": "Saber Round 3 specification", "url": "https://www.esat.kuleuven.be/cosic/pqcrypto/saber/files/saberspecround3.pdf",
             "use": "n=256, original q=8192 and p=1024, family rank and secret support"},
            {"title": "Multi-moduli NTTs for Saber on Cortex-M3 and Cortex-M4", "url": "https://vincentvbh.github.io/papers/2021-995.pdf",
             "use": "Auxiliary moduli 3329 and 7681; six-layer four-component NTT and CRT strategy"}],
        "polynomial_ring": "Z_q[X]/(X^256+1), q=8192 or p=1024 as required by the arithmetic stage",
        "auxiliary_moduli": list(MODULI), "crt_modulus": MODULI[0] * MODULI[1],
        "matrix_shape_per_modulus_per_direction": [M, M],
        "lane_count": LANES, "lane_rule": "input[4*j+r] is position j of lane r, r=0,1,2,3",
        "output_layout": "output[4*i+r] is coefficient r of the component modulo X^4-gamma_i",
        "root_choice": "First integer base >=2 whose ((p-1)/128)-th power has exact order 128",
        "branch_parameters": [{"p": b["p"], "bits": (b["p"] - 1).bit_length(),
                                "base": b["base"], "psi": b["psi"], "psi_order": ROOT_ORDER,
                                "inverse64": b["inverse64"]} for b in branches],
        "gamma_formula": "gamma_i = psi^(2*bitrev6(i)+1) mod p",
        "forward_formula": "W[i,j] = gamma_i^j mod p",
        "inverse_formula": "Winv[j,i] = inverse(64,p) * gamma_i^(-j) mod p",
        "normalization": "Already included in Winv; do not multiply inverse64 again",
        "representation": "Canonical residues 0..p-1, ordinary domain, not Montgomery",
        "physical_layout": "For input rows/output columns, physical weight matrix is mathematical W transpose",
        "bitplane_layout": "bitplanes_[name]_[p][i,:,:], i=0 is LSB; each stored entry is exactly 0 or 1",
        "heatmap_instruction": "Plot logical weights, or normalize by p-1 and label normalized logical weight; never label conductance",
        "crt_reconstruction": "r = r0 + 3329*((r1-r0)*inverse(3329,7681) mod7681); center modulo25570049, then reduce to8192 or1024",
        "capacity": {"binary_cells_per_64x64_plane": M * M,
                     "planes_for_both_moduli_one_direction": 25,
                     "binary_cells_one_direction": 25 * M * M,
                     "planes_for_separate_forward_and_inverse_weights": 50,
                     "binary_cells_forward_and_inverse": 50 * M * M,
                     "existing_chip_64x64_arrays": 4,
                     "existing_chip_binary_cells": 4 * M * M,
                     "all_weights_resident_on_existing_chip": False,
                     "note": "Capacity above is logical mapping demand, not an implemented scheduling plan; even one direction exceeds current chip capacity."},
        "validation": {"result": "PASS", "rng_seed": SEED,
                       "auxiliary_fields": field_results, "original_modulus_arithmetic": family_results,
                       "int64_overflow_bounds_checked": True},
        "limitations": ["Uniform and CBD test samples use a seeded NumPy generator, not Saber SHAKE/GenSecret byte streams",
                        "No message encoding, rounding offsets, serialization, FO transform, KEM KATs, or hardware faults are tested",
                        "Logical forward and inverse matrices share a mathematical operator across four lanes; this does not imply four physical reads",
                        "Keep the center-lift bound valid before reducing to the original modulus; directly chaining auxiliary and original reductions is incorrect"],
        "python": platform.python_version(), "numpy": np.__version__,
        "artifact": artifact.name, "artifact_sha256": digest(artifact),
        "generator": Path(__file__).name, "generator_sha256": digest(Path(__file__)),
        "npz_keys": sorted(arrays),
    }
    (HERE / "source_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": "PASS", "directory": "matrices",
                      "branch_parameters": metadata["branch_parameters"],
                      "family_results": family_results,
                      "artifact_sha256": metadata["artifact_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
