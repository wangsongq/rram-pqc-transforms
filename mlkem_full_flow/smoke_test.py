#!/usr/bin/env python3
"""Fast offline checks of all three recommended ML-KEM parameter sets."""
import hashlib
import json
from pathlib import Path

from mlkem import MLKEM, PARAMS
from rram_transform_only import TransformOnlyMLKEM
from validation import require, verify_sources


def main():
    verify_sources()
    rows = []
    invalid_checks = 0
    for name in PARAMS:
        d, z, m = (hashlib.sha256((name + ":" + label).encode()).digest()
                   for label in ("d", "z", "m"))
        reference = MLKEM(name, backend="butterfly")
        expected_ek, expected_dk = reference.keygen(d, z)
        expected_key, expected_ct = reference.encaps(expected_ek, m)
        kem = TransformOnlyMLKEM(name, signed_cbd_inputs=False)
        ek, dk = kem.keygen(d, z)
        key, ciphertext = kem.encaps(ek, m)
        require((ek, dk, key, ciphertext) ==
                (expected_ek, expected_dk, expected_key, expected_ct),
                name + " complete-flow equality with butterfly")
        require(kem.decaps(dk, ciphertext) == key, name + " roundtrip")
        modified = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]
        fallback = hashlib.shake_256(z + modified).digest(32)
        require(kem.decaps(dk, modified) == fallback, name + " implicit rejection")
        for values in ([0] * 256, [3328] * 256, list(range(256))):
            require(kem.intt(kem.ntt(values)) == values, name + " transform roundtrip")
        for action in (lambda: kem.ntt([0] * 255),
                       lambda: kem.intt([0.5] * 256),
                       lambda: kem.encaps(ek[:-1], m),
                       lambda: kem.decaps(dk, ciphertext[:-1])):
            try:
                action()
            except ValueError:
                invalid_checks += 1
            else:
                raise ValueError("Malformed input was accepted")
        rows.append({"parameter_set": name, "passed": True})
    report = {"status": "passed", "parameter_sets": rows,
              "invalid_inputs_rejected": invalid_checks,
              "official_fixture_hashes_verified": True}
    output = Path(__file__).resolve().parent / "transform_only_results"
    output.mkdir(exist_ok=True)
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
