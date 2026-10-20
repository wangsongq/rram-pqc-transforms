"""Shared helpers for the offline, official ML-KEM validation suite.

The fixed NIST JSON inputs and expected results are joined by (tgId, tcId).
Expected values are loaded from these independently supplied files; they are
never recomputed with the implementation being checked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

VECTORS = Path(__file__).resolve().parent / "golden_vectors"


def read_json(path: str | Path):
    """Read a bundled JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, detail):
    """Enforce a scientific check, including under ``python -O``."""
    if not condition:
        raise ValueError("Verification failed: " + str(detail))


def test_lookup(obj):
    """Index official cases by their group and case identifiers."""
    return {(group["tgId"], case["tcId"]): case
            for group in obj["testGroups"] for case in group["tests"]}


def verify_sources():
    """Check every official fixture against its fixed size and SHA-256."""
    provenance = read_json(VECTORS / "provenance.json")
    for record in provenance["files"]:
        data = (VECTORS / record["path"]).read_bytes()
        if (len(data) != record["bytes"]
                or hashlib.sha256(data).hexdigest() != record["sha256"]):
            raise ValueError("Golden source integrity failure: " + record["path"])
    return provenance


def execute_case(kem, function, case):
    """Execute one public ACVP case, returning values in the official schema.

    Seed-form decapsulation cases first expand their d,z seeds into dk.
    A key-check case whose expected testPassed is false is a negative test,
    not a failure of the local validation.
    """
    def b(key):
        return bytes.fromhex(case[key])

    if function == "keyGen":
        ek, dk = kem.keygen(b("d"), b("z"))
        return {"ek": ek, "dk": dk}
    if function == "encapsulation":
        key, ciphertext = kem.encaps(b("ek"), b("m"))
        return {"c": ciphertext, "k": key}
    if function == "decapsulation":
        dk = b("dk") if "dk" in case else kem.keygen(b("d"), b("z"))[1]
        return {"k": kem.decaps(dk, b("c"))}
    if function == "encapsulationKeyCheck":
        return {"testPassed": kem.check_ek(b("ek"))}
    if function == "decapsulationKeyCheck":
        return {"testPassed": kem.check_dk(b("dk"))}
    raise ValueError("Unsupported official test function: " + function)
