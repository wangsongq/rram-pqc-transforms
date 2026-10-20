#!/usr/bin/env python3
"""Check the binary Saber KEM's implicit rejection for altered ciphertexts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from golden_vectors.kat_io import load_inputs, load_kat, verify_sources
from saber import Saber

ROOT = Path(__file__).resolve().parent


def run_checks() -> dict:
    """Check three ciphertext alterations for each official parameter set.

    Expected keys are computed independently from the stored fallback secret:
    SHA3-256(z || SHA3-256(altered ciphertext)). These are negative tests, not
    additional official KAT cases.
    """
    verify_sources()
    inputs, rows = load_inputs(), []
    for family in ("LightSaber", "Saber", "FireSaber"):
        kat = load_kat(family)[0]
        raw = inputs[(family, 0)]
        for index in (0, len(kat["ct"]) // 2, len(kat["ct"]) - 1):
            altered = bytearray(kat["ct"])
            altered[index] ^= 1
            altered = bytes(altered)
            expected = hashlib.sha3_256(
                raw["z"] + hashlib.sha3_256(altered).digest()
            ).digest()
            decisions = []

            def sink(stage, values):
                if values.get("operation") == "implicit_rejection":
                    decisions.append(values["accepted"])

            model = Saber(family, backend="rram_binary", trace=sink)
            actual = model.decaps(kat["sk"], altered)
            if actual != expected or decisions != [False]:
                raise AssertionError(f"Implicit rejection mismatch: {family}/{index}")
            rows.append({
                "parameter_set": family,
                "flipped_ciphertext_byte": index,
                "status": "passed",
            })
    report = {
        "status": "passed",
        "backend": "rram_binary",
        "checks": len(rows),
        "expected_KDF": "SHA3-256(z || SHA3-256(altered ciphertext))",
        "results": rows,
    }
    destination = ROOT / "generated/results"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "implicit_rejection_checks.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
