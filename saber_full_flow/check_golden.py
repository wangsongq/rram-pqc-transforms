#!/usr/bin/env python3
"""Check the binary Saber implementation against official Round-3 KAT outputs.

This checker never synthesizes expected keys, ciphertexts, or shared secrets.
Those bytes are read directly from the unmodified upstream .rsp files. The
derived fixture contains only raw randombytes inputs captured in official C.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

from golden_vectors.kat_io import FAMILIES, ROOT, load_inputs, load_kat, sha256, verify_sources

BACKENDS = ("schoolbook", "matrix", "rram_binary")
CHECKS = ("pk", "sk", "ct", "ss_encaps", "ss_decaps")


def run_checks(
    backend: str = "rram_binary",
    output_dir: str | Path | None = None,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run all 300 cases on one backend (or an explicitly marked smoke subset).

    Return a JSON-compatible report and write golden_checks.csv and
    golden_summary.json. Every expected value comes from official .rsp data;
    KeyGen, Encaps, and Decaps receive their own official inputs so failures in
    one operation cannot silently replace another operation's reference input.
    Failed comparisons and exceptions remain visible in the CSV and report.
    """
    from saber import Saber

    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    selected = (backend,)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100):
        raise ValueError("limit must be None or an integer in 1..100")
    count_per_family = 100 if limit is None else limit
    out = ROOT / "generated/results" if output_dir is None else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    provenance = verify_sources()
    inputs = load_inputs()
    cases = {name: load_kat(name)[:count_per_family] for name in FAMILIES}
    for name, records in cases.items():
        for record in records:
            if inputs[(name, record["count"])]["seed"] != record["seed"]:
                raise ValueError(f"RNG fixture seed differs from official KAT: {name}/{record['count']}")
    source_hashes = {
        path.name: sha256(path) for path in (
            ROOT / "saber.py", ROOT / "auxiliary_ntt.py", ROOT / "check_golden.py",
            ROOT / "golden_vectors/kat_io.py",
        ) if path.exists()
    }
    rows, failures = [], []
    for backend in selected:
        for name, records in cases.items():
            family_started = time.perf_counter()
            for record in records:
                count = record["count"]
                raw = inputs[(name, count)]
                case_started = time.perf_counter()
                row = {"parameter_set": name, "backend": backend, "count": count, **{key: "not_run" for key in CHECKS}}
                errors = []

                def compare(field: str, actual, expected: bytes):
                    equal = isinstance(actual, bytes) and actual == expected
                    row[field] = "passed" if equal else "failed"
                    if not equal:
                        actual_len = len(actual) if hasattr(actual, "__len__") else None
                        mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None) if isinstance(actual, bytes) else None
                        errors.append(f"{field}: expected {len(expected)} bytes; actual type={type(actual).__name__}, length={actual_len}, first_mismatch={mismatch}")

                # Fresh instances also prevent persistent state from hiding an
                # API dependency absent from the official deterministic inputs.
                for operation in ("keygen", "encaps", "decaps"):
                    op_started = time.perf_counter()
                    try:
                        implementation = Saber(name, backend=backend)
                        if operation == "keygen":
                            pk, sk = implementation.keygen(raw["seed_a"], raw["seed_s"], raw["z"])
                            compare("pk", pk, record["pk"])
                            compare("sk", sk, record["sk"])
                        elif operation == "encaps":
                            shared_secret, ciphertext = implementation.encaps(record["pk"], raw["m"])
                            compare("ct", ciphertext, record["ct"])
                            compare("ss_encaps", shared_secret, record["ss"])
                        else:
                            recovered = implementation.decaps(record["sk"], record["ct"])
                            compare("ss_decaps", recovered, record["ss"])
                    except Exception as exc:
                        affected = {"keygen": ("pk", "sk"), "encaps": ("ct", "ss_encaps"), "decaps": ("ss_decaps",)}[operation]
                        for field in affected:
                            row[field] = "error"
                        errors.append(f"{operation}: {type(exc).__name__}: {exc}")
                    row[f"{operation}_seconds"] = time.perf_counter() - op_started
                row["status"] = "passed" if all(row[field] == "passed" for field in CHECKS) else "failed"
                row["errors"] = " | ".join(errors)
                row["elapsed_seconds"] = time.perf_counter() - case_started
                rows.append(row)
                if row["status"] != "passed":
                    failures.append({"parameter_set": name, "backend": backend, "count": count, "errors": errors})
                if progress is not None and ((count + 1) % 10 == 0 or count + 1 == count_per_family):
                    progress(f"Official KAT {backend}/{name}: {count + 1}/{count_per_family}, {time.perf_counter() - family_started:.2f} s")
    verify_sources()
    if source_hashes != {name: sha256(ROOT / ("golden_vectors/" if name == "kat_io.py" else "") / name) for name in source_hashes}:
        raise RuntimeError("Implementation or checker source changed during the golden run")
    expected_keys = {(name, backend, count) for name in FAMILIES for backend in selected for count in range(count_per_family)}
    actual_keys = [(r["parameter_set"], r["backend"], r["count"]) for r in rows]
    if set(actual_keys) != expected_keys or len(actual_keys) != len(expected_keys):
        raise RuntimeError("Golden suite completeness/uniqueness check failed")
    statuses = Counter(row["status"] for row in rows)
    comparisons_passed = sum(row[field] == "passed" for row in rows for field in CHECKS)
    report = {
        "status": "passed" if not failures else "failed",
        "official_repository": provenance["repository"], "official_commit": provenance["commit"],
        "expected_source": "Pinned official .rsp files, verified by SHA256; never generated by these Python implementations",
        "input_source": "Four 32-byte randombytes outputs captured from official C AES256 CTR DRBG and validated against all 300 official C KAT executions",
        "backends": list(selected), "parameter_sets": list(FAMILIES),
        "full_official_case_set": count_per_family == 100,
        "official_unique_cases": 3 * count_per_family,
        "case_backend_executions": len(rows), "case_executions_passed": statuses["passed"], "case_executions_failed": statuses["failed"],
        "byte_comparisons": len(rows) * len(CHECKS), "byte_comparisons_passed": comparisons_passed,
        "comparison_fields": list(CHECKS),
        "by_parameter_and_backend": [
            {"parameter_set": name, "backend": backend, "cases": count_per_family,
             "passed": sum(r["status"] == "passed" for r in rows if r["parameter_set"] == name and r["backend"] == backend)}
            for backend in selected for name in FAMILIES
        ],
        "failures": failures, "source_sha256": source_hashes,
        "source_integrity_verified_before_and_after": True,
        "elapsed_seconds": time.perf_counter() - started,
        "csv": str(out / "golden_checks.csv"),
        "counting_note": "One case execution comprises KeyGen pk/sk, Encaps ct/ss, and Decaps ss comparisons. Invalid-ciphertext checks are separate and are not counted as official KAT cases.",
    }
    with (out / "golden_checks.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "golden_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="rram_binary",
                        help="Implementation to verify; schoolbook and matrix are independent arithmetic oracles")
    parser.add_argument("--limit", type=int, default=None, help="Explicit smoke subset per parameter set (default: all 100)")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run_checks(args.backend, args.output_dir, args.limit, progress=print)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
