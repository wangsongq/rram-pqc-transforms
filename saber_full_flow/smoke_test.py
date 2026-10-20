#!/usr/bin/env python3
"""Quick binary-weight, official-case and implicit-rejection checks.

This runs only official case 0 of each parameter set. Use check_golden.py for
all 300 official cases and check_auxiliary.py for independent arithmetic checks.
"""
from __future__ import annotations

import json
from pathlib import Path

from auxiliary_ntt import export_weights
from check_binary_weights import verify_weights
from check_golden import run_checks
from check_rejection import run_checks as check_rejection

ROOT = Path(__file__).resolve().parent


def main() -> None:
    export_weights()
    weights = verify_weights()
    results = ROOT / "generated/results"
    results.mkdir(parents=True, exist_ok=True)
    golden = run_checks(limit=1, output_dir=results / "smoke",
                        progress=lambda line: print(line, flush=True))
    rejection = check_rejection()
    if golden["status"] != "passed":
        raise AssertionError("Official Saber smoke checks failed")
    report = {
        "status": "passed",
        "scope": "Official case 0 per family on rram_binary; not the full KAT corpus",
        "binary_csv_files": weights["csv_files"],
        "binary_npz_plane_stacks": weights["npz_plane_stacks"],
        "official_unique_cases": golden["official_unique_cases"],
        "case_backend_executions": golden["case_backend_executions"],
        "byte_comparisons": golden["byte_comparisons"],
        "implicit_rejection_checks": rejection["checks"],
    }
    (results / "smoke_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
