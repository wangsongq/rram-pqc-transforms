#!/usr/bin/env python3
"""Independent mathematical and boundary audit of Saber auxiliary arithmetic.

Uses direct modular powers, Python arbitrary-precision REDC and negacyclic
convolution, unpacked binary counts, and one-cell causal perturbations. It
does not rerun the official KAT suite or modify any implementation/weights.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np

from auxiliary_ntt import SaberPolynomialEngine, load_artifacts, redc32
from saber import PARAMS, pack_poly, unpack_poly

ROOT = Path(__file__).resolve().parent
MODULI = (3329, 7681)
R = 1 << 32
PRODUCT = 3329 * 7681


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, label):
    require(np.array_equal(actual, expected), label)


def rejected(function, label):
    try:
        function()
    except (ValueError, TypeError, OverflowError):
        return
    raise AssertionError("Input should have been rejected: " + label)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mathematical_matrices(p):
    # Independently rediscover the first base with a primitive 128th root.
    for base in range(2, p):
        psi = pow(base, (p - 1) // 128, p)
        if pow(psi, 64, p) == p - 1 and pow(psi, 128, p) == 1:
            break
    exponents = []
    for i in range(64):
        reversed_i = sum(((i >> bit) & 1) << (5 - bit) for bit in range(6))
        exponents.append(2 * reversed_i + 1)
    gamma = [pow(psi, exponent, p) for exponent in exponents]
    forward = [[pow(gamma[i], j, p) for j in range(64)] for i in range(64)]
    inverse = [[pow(64, -1, p) * pow(gamma[i], -j, p) % p for i in range(64)] for j in range(64)]
    return base, psi, gamma, np.array(forward, dtype=np.int64), np.array(inverse, dtype=np.int64)


def transform_oracle(poly, matrix, p):
    # Python sums and products, independent of implementation dtype and kernel.
    out = [0] * 256
    for lane in range(4):
        for row in range(64):
            out[4 * row + lane] = sum(int(matrix[row, col]) * (int(poly[4 * col + lane]) % p) for col in range(64)) % p
    return out


def negacyclic_oracle(left, right, target):
    total = [0] * 256
    for a_raw, b_raw in zip(left, right):
        a = [(int(x) + target // 2) % target - target // 2 for x in a_raw]
        b = [(int(x) + 4096) % 8192 - 4096 for x in b_raw]
        for i, a_i in enumerate(a):
            for j, b_j in enumerate(b):
                degree = i + j
                total[degree & 255] += a_i * b_j * (1 if degree < 256 else -1)
    return total


def verify_matrices_and_basis(report, oracle):
    matrices, weights = load_artifacts()
    engine = SaberPolynomialEngine("rram_binary")
    branch_reports = []
    for p in MODULI:
        base, psi, gamma, w, v = mathematical_matrices(p)
        oracle[p] = (w, v)
        k = np.array([[int(value) * R % p for value in row] for row in v], dtype=np.int64)
        for name, expected in (("W", w), ("V", v), ("K", k), ("gamma", gamma)):
            equal(matrices[f"{name}_{p}"], expected, f"Independent {name} matrix/root p={p}")
        equal(v @ w % p, np.eye(64, dtype=np.int64), f"V W identity p={p}")
        equal(w @ v % p, np.eye(64, dtype=np.int64), f"W V identity p={p}")
        require(pow(64, -1, p) == int(matrices[f"inverse64_{p}"]), "Inverse normalization constant")
        bits = (p - 1).bit_length()
        for direction, actual_weight, output_matrix in (("ntt", w, w), ("intt", k, v)):
            planes = weights[f"{direction}_{p}_physical_planes"]
            require(planes.shape == (bits, 64, 64) and planes.dtype == np.uint8, "Binary plane dimensions/type")
            require(np.all((planes == 0) | (planes == 1)), "Only binary cell values")
            reconstructed = sum(planes[bit].astype(np.int64) * (1 << bit) for bit in range(bits))
            equal(reconstructed, actual_weight.T, f"LSB unsigned reconstruction and transpose p={p}/{direction}")
            # Independently unpack the uint64 word: bit index is input row.
            packed = engine.packed_tiles[(p, direction)]
            for bit in range(bits):
                unpacked = np.array([[(int(packed[bit, column]) >> row) & 1 for column in range(64)] for row in range(64)], dtype=np.uint8)
                equal(unpacked, planes[bit], f"Packed row direction p={p}/{direction}/{bit}")
            for index in range(256):
                poly = [0] * 256
                poly[index] = 1
                expected = [0] * 256
                for row in range(64):
                    expected[4 * row + index % 4] = int(output_matrix[row, index // 4])
                actual = getattr(engine, direction)(poly, p, stage="audit.basis")
                equal(actual, expected, f"Basis p={p}/{direction}/index={index}")
        branch_reports.append({"modulus": p, "root_base": base, "psi": psi, "inverse64": pow(64, -1, p),
                               "weight_bits": bits, "basis_cases_each_direction": 256,
                               "forward_stored": "W", "inverse_stored": "R32*V mod p", "input_output_domain": "ordinary canonical"})
    expected_reads = sum(2 * 256 * 4 * (p - 1).bit_length() ** 2 for p in MODULI)
    require(engine.counts["binary_tile_read_vectors"] == expected_reads, "Full-basis logical read count")
    require(engine.counts["binary_column_popcounts"] == expected_reads * 64, "Full-basis column popcount count")
    require(engine.counts["redc32_coefficients"] == 512 * 256, "Exactly one REDC per inverse coefficient")
    report["matrices_and_all_basis"] = {"status": "passed", "basis_transforms": 1024, "output_coefficients_compared": 1024 * 256,
                                          "branches": branch_reports, "counts": dict(engine.counts),
                                          "logical_reads_are_not_hardware_cycles": True}


def verify_redc(report, rng):
    branches, tested, rejected_count = [], 0, 0
    for p in MODULI:
        maximal_u_input = (p - 1) * R + p
        values = sorted(set([0, 1, p - 1, p, p + 1, R - 1, R, R + 1,
                             64 * (p - 1) ** 2, p * R - 1, maximal_u_input,
                             *[rng.randrange(p * R) for _ in range(256)]]))
        output, trace = redc32(values, p)
        expected = [value * pow(R, -1, p) % p for value in values]
        equal(output, expected, f"Arbitrary-precision REDC oracle p={p}")
        p_dash = (-pow(p, -1, R)) % R
        corrections = [((t % R) * p_dash) % R for t in values]
        numerators = [t + m * p for t, m in zip(values, corrections)]
        quotients = [n // R for n in numerators]
        equal(trace["T"], values, "REDC trace T")
        equal(trace["m"], corrections, "REDC trace correction")
        equal(trace["u"], quotients, "REDC trace quotient")
        require(trace["p_dash"] == p_dash, "REDC trace p_dash")
        require(all(n % R == 0 for n in numerators), "REDC numerator exact divisibility")
        require(max(quotients) == 2 * p - 1, "REDC maximal quotient attained")
        require((R - 1) * p_dash < 2**64 and 2 * p * R < 2**46, "uint64 intermediate bound")
        for invalid in ([-1], [p * R], [p * R + 1], [1 << 65], [1.5], [-0.5], [True], [float("nan")]):
            rejected(lambda invalid=invalid: redc32(invalid, p), f"REDC {p}/{invalid}")
            rejected_count += 1
        branches.append({"modulus": p, "cases": len(values), "legal_public_input": f"0 <= T < {p * R}",
                         "p_dash": p_dash, "max_kernel_dot_bound": 64 * (p - 1) ** 2,
                         "kernel_dot_below_R32": 64 * (p - 1) ** 2 < R,
                         "max_quotient_attained": max(quotients), "max_tested_numerator": max(numerators),
                         "strict_numerator_bound": 2 * p * R, "strict_numerator_bound_below_2pow46": True})
        tested += len(values)
    rejected(lambda: redc32([1], 8192), "unsupported REDC modulus")
    report["redc_boundaries"] = {"status": "passed", "integer_inputs_compared": tested, "invalid_inputs_rejected": rejected_count + 1, "branches": branches}


def verify_binary_counts_and_causality(report, oracle, rng):
    comparisons, trace_events, mutation_cases, boundary_cases = 0, 0, 0, 0
    for p in MODULI:
        bits = (p - 1).bit_length()
        for direction in ("ntt", "intt"):
            events = []
            engine = SaberPolynomialEngine("rram_binary", trace=lambda stage, values: events.append((stage, values)))
            samples = [[p - 1] * 256, [rng.randrange(p) for _ in range(256)],
                       [(-1 if i % 2 else 1) * (2**80 + i) for i in range(256)]]
            for poly in samples:
                events.clear()
                actual = getattr(engine, direction)(poly, p, stage="audit.counts")
                expected = transform_oracle(poly, oracle[p][direction == "intt"], p)
                equal(actual, expected, "Structured/random/unbounded integer canonicalization")
                event = next(data for _, data in events if data.get("operation") == "rram_binary_vmm")
                raw = event["tile_counts"]
                planes = engine.weights[f"{direction}_{p}_physical_planes"]
                lanes = np.array([int(x) % p for x in poly], dtype=np.int64).reshape(64, 4).T
                independent = np.empty((4, bits, bits, 64), dtype=np.int64)
                for lane in range(4):
                    for input_bit in range(bits):
                        pulse = (lanes[lane] // (1 << input_bit)) % 2
                        for weight_bit in range(bits):
                            # Explicit boolean-cell sums; no packed AND/popcount.
                            independent[lane, input_bit, weight_bit] = (planes[weight_bit].astype(np.int64) * pulse[:, None]).sum(axis=0)
                equal(raw, independent, "Raw binary column counts independent recomputation")
                require(raw.dtype == np.uint8 and int(raw.max()) <= 64, "Raw popcount uint8 bound")
                rebuilt = np.zeros((4, 64), dtype=np.int64)
                for input_bit in range(bits):
                    for weight_bit in range(bits):
                        rebuilt += independent[:, input_bit, weight_bit] * (1 << (input_bit + weight_bit))
                equal(rebuilt, event["accumulator_lanes"], "Integer shift-add reconstruction")
                require(event["weight"] == ("K" if direction == "intt" else "W"), "Only inverse uses Montgomery fused weight")
                final = events[-1][1]
                require(final["input_representation"] == final["output_representation"] == "ordinary_canonical", "No Montgomery-domain interface leak")
                require(final["inverse_normalization_count"] == int(direction == "intt"), "Inverse normalization exactly once")
                redc_events = [data for _, data in events if data.get("operation") == "redc32"]
                require(len(redc_events) == int(direction == "intt"), "Only inverse performs REDC")
                if redc_events:
                    require(redc_events[0]["normalization_count"] == 1 and redc_events[0]["inverse64_already_in_weights"], "Inverse64 already fused")
                    require(max(int(v) for v in rebuilt.flat) < R, "Kernel-specific REDC input bound")
                comparisons += raw.size
                trace_events += len(events)
                boundary_cases += 1
            require(engine.counts["binary_tile_read_vectors"] == 3 * 4 * bits * bits, "Structured test logical reads")
            # Prove binary transforms do not depend on the dense matrix values.
            engine.trace = None
            engine.matrices = {key: np.zeros_like(value) for key, value in engine.matrices.items()}
            poly = [rng.randrange(p) for _ in range(256)]
            equal(getattr(engine, direction)(poly, p), transform_oracle(poly, oracle[p][direction == "intt"], p), "Binary execution independent of dense matrices")
            # Flip one stored cell (highest weight plane, last physical row).
            before = getattr(engine, direction)(poly, p)
            weight_bit, input_row, output_column = bits - 1, 63, 17
            words = engine.packed_tiles[(p, direction)].copy()
            old = (int(words[weight_bit, output_column]) >> input_row) & 1
            words[weight_bit, output_column] ^= np.uint64(1) << np.uint64(input_row)
            engine.packed_tiles[(p, direction)] = words
            after = getattr(engine, direction)(poly, p)
            expected_after = list(before)
            factor = (1 - 2 * old) * (1 << weight_bit)
            if direction == "intt":
                factor = factor * pow(R, -1, p) % p
            for lane in range(4):
                index = 4 * output_column + lane
                expected_after[index] = (expected_after[index] + poly[4 * input_row + lane] * factor) % p
            equal(after, expected_after, "Single cell bit causes only predicted weighted column changes")
            require(before != after, "Single cell causal test must make an observable change")
            mutation_cases += 1
    report["binary_counts_trace_and_causality"] = {"status": "passed", "boundary_transforms": boundary_cases,
        "individual_raw_counts_recomputed": comparisons, "trace_events_checked": trace_events,
        "single_cell_causal_perturbations": mutation_cases, "dense_matrix_independence_cases": 4,
        "storage": "Unsigned 12-bit / 13-bit weights in binary cells; fixed unsigned input slots",
        "physical_orientation": "row=input, column=output; four lanes reuse the same planes"}


def verify_crt_and_dot(report, rng):
    cases, output_comparisons, attained, invalid_count = 0, 0, [], 0
    for name, parameters in PARAMS.items():
        rank, eta = parameters.k, parameters.mu // 2
        for target in (8192, 1024):
            maximal = 256 * (target // 2) * rank * eta
            require(2 * maximal < PRODUCT, "Family worst-case CRT uniqueness")
            test_vectors = []
            for sign in (-1, 1):
                left = [[-(target // 2)] * 256 for _ in range(rank)]
                one_secret = [-sign * eta] + [sign * eta] * 255
                # Canonical mod8192 serialized secret must recover its sign,
                # including when the left operand is centered modulo1024.
                right = [[coefficient % 8192 for coefficient in one_secret] for _ in range(rank)]
                test_vectors.append((f"attained_{sign:+d}", left, right, sign))
            for trial in range(2):
                left = [[rng.randrange(target) for _ in range(256)] for _ in range(rank)]
                right = [[rng.randint(-eta, eta) % 8192 for _ in range(256)] for _ in range(rank)]
                test_vectors.append((f"random_{trial}", left, right, None))
            for label, left, right, sign in test_vectors:
                integer = negacyclic_oracle(left, right, target)
                expected = [x % target for x in integer]
                if sign is not None:
                    require(integer[0] == sign * maximal and max(map(abs, integer)) == maximal, "Signed worst-case bound attained exactly")
                    attained.append({"parameter_set": name, "rank": rank, "mu": parameters.mu, "target_modulus": target,
                                     "sign": sign, "attained_integer_bound": maximal, "twice_bound": 2 * maximal, "crt_modulus": PRODUCT})
                for backend in ("matrix", "rram_binary"):
                    crt_records, input_records = [], []
                    def sink(stage, data):
                        if data.get("operation") == "crt_centered":
                            crt_records.append(data)
                        elif data.get("operation") == "polynomial_dot_inputs":
                            input_records.append(data)
                    engine = SaberPolynomialEngine(backend, trace=sink)
                    actual = engine.dot(left, right, target, stage="audit.dot")
                    equal(actual, expected, f"Independent negacyclic {name}/{target}/{label}/{backend}")
                    require(len(crt_records) == 1, "One complete two-prime CRT reconstruction")
                    equal(crt_records[0]["centered_crt"], integer, "CRT recovers exact integers before original modulus")
                    require(crt_records[0]["center_before_target_reduction"], "Center CRT before target modulo")
                    require(input_records[0]["right_center_modulus"] == 8192, "Secret centers mod8192 even for target1024")
                    counts = engine.counts
                    require(counts["ntt"] == 4 * rank and counts["intt"] == 2, "Dot transform count")
                    require(counts["quartic_base_products"] == 2 * rank * 64, "Dot quartic product count")
                    require(counts["redc32_coefficients"] == 512, "Dot REDC count")
                    expected_reads = (2 * rank + 1) * 4 * (12**2 + 13**2) if backend == "rram_binary" else 0
                    require(counts["binary_tile_read_vectors"] == expected_reads, "Dot logical read count")
                    require(counts["binary_column_popcounts"] == 64 * expected_reads, "Dot column count")
                    output_comparisons += 256
                    cases += 1
    for backend in ("matrix", "rram_binary"):
        engine = SaberPolynomialEngine(backend)
        zero = [0] * 256
        invalid = [
            (lambda: engine.dot([], [], 8192), "empty rank"),
            (lambda: engine.dot([zero], [], 8192), "mismatched rank"),
            (lambda: engine.dot([zero], [[6] * 256], 8192), "outside CBD support"),
            (lambda: engine.dot([[4096] * 256] * 3, [[5] * 256] * 3, 8192), "CRT uniqueness exceeded"),
            (lambda: engine.dot([[0] * 255], [zero], 8192), "wrong degree count"),
            (lambda: engine.dot([[0.5] * 256], [zero], 8192), "noninteger dot"),
            (lambda: engine.dot([zero], [zero], 3329), "unsupported target"),
            (lambda: engine.ntt([0] * 255, 3329), "short transform"),
            (lambda: engine.intt([0.5] * 256, 3329), "noninteger inverse"),
            (lambda: engine.ntt(zero, 8192), "unsupported auxiliary modulus"),
        ]
        for function, label in invalid:
            rejected(function, backend + "/" + label)
            invalid_count += 1
    report["crt_negacyclic_and_guards"] = {"status": "passed", "dot_executions": cases,
        "output_coefficients_compared": output_comparisons, "signed_attained_bound_cases": attained,
        "invalid_inputs_rejected": invalid_count,
        "right_secret_centering": "always modulo8192, including ciphertext inner products modulo1024"}


def verify_packing_and_rounding(report, rng):
    packing_cases, rounding_cases = 0, 0
    for bits in range(1, 14):
        mask = (1 << bits) - 1
        for poly in ([mask] * 256, [-1 if i % 2 else 1 << bits for i in range(256)], [rng.randint(-8192, 8192) for _ in range(256)]):
            independent = bytearray(32 * bits)
            for coefficient, value in enumerate(poly):
                for bit in range(bits):
                    absolute = coefficient * bits + bit
                    independent[absolute // 8] |= ((value >> bit) & 1) << (absolute % 8)
            equal(list(pack_poly(poly, bits)), list(independent), "Packing independent bit scatter")
            equal(unpack_poly(bytes(independent), bits), [x & mask for x in poly], "Unpacking mask/sign boundary")
            packing_cases += 1
    # Early reduction from a 16-bit reference accumulator must preserve every
    # stored low bit after Saber rounding, including subtraction wraparound.
    values = np.arange(65536, dtype=np.int64)
    equal(((values + 4) >> 3) & 1023, ((((values % 8192) + 4) >> 3) & 1023), "KeyGen q rounding invariant")
    rounding_cases += len(values)
    for parameters in PARAMS.values():
        for message_bit in (0, 1):
            original = ((values - message_bit * 512 + parameters.h1) >> (10 - parameters.et)) & ((1 << parameters.et) - 1)
            reduced = (((values % 1024) - message_bit * 512 + parameters.h1) >> (10 - parameters.et)) & ((1 << parameters.et) - 1)
            equal(original, reduced, "Encaps p rounding invariant")
            rounding_cases += len(values)
        for compressed in range(1 << parameters.et):
            original = ((values + parameters.h2 - (compressed << (10 - parameters.et))) >> 9) & 1
            reduced = (((values % 1024) + parameters.h2 - (compressed << (10 - parameters.et))) >> 9) & 1
            equal(original, reduced, "Decaps p rounding invariant")
            rounding_cases += len(values)
    report["saber_packing_and_rounding_boundaries"] = {"status": "passed", "packing_cases": packing_cases,
        "integer_rounding_comparisons": rounding_cases,
        "meaning": "C uint16 accumulator representatives and centered/reduced Python arithmetic retain identical encoded low bits; all family ET widths checked"}


def run_verification() -> dict:
    started = time.perf_counter()
    tracked = [ROOT / "auxiliary_ntt.py", ROOT / "saber.py", ROOT / "check_auxiliary.py",
               ROOT / "matrices/source_weights.npz", ROOT / "matrices/saber_auxiliary_matrices.npz",
               ROOT / "rram_binary/saber_binary_weights.npz"]
    initial_hashes = {str(path.relative_to(ROOT)): digest(path) for path in tracked}
    report = {"status": "running", "source_sha256": initial_hashes,
              "official_kat_repeated": False,
              "reference_methods": ["direct modular powers", "Python arbitrary-precision REDC", "Python signed negacyclic convolution", "unpacked binary column sums", "single-cell causal perturbations"]}
    try:
        rng = random.Random(20261005)
        oracle = {}
        verify_matrices_and_basis(report, oracle)
        verify_redc(report, rng)
        verify_binary_counts_and_causality(report, oracle, rng)
        verify_crt_and_dot(report, rng)
        verify_packing_and_rounding(report, rng)
        require(initial_hashes == {str(path.relative_to(ROOT)): digest(path) for path in tracked}, "Implementation or weight changed during audit")
        report["source_integrity_unchanged"] = True
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    report["elapsed_seconds"] = time.perf_counter() - started
    output = ROOT / "generated/results/auxiliary_review.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run_verification()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
