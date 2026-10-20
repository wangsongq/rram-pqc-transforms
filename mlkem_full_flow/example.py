#!/usr/bin/env python3
"""Run one deterministic public example through the recommended backend."""
import argparse
import hashlib
import json

from mlkem import PARAMS
from rram_transform_only import TransformOnlyMLKEM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-set", choices=PARAMS, default="ML-KEM-768")
    parser.add_argument("--signed-cbd", action="store_true",
                        help="Use the optional three-slot encoding at known CBD stages")
    args = parser.parse_args()
    # Public, deterministic inputs are for reproducibility, not deployed keys.
    d, z, m = (hashlib.sha256(("public-mlkem-example:" + label).encode()).digest()
               for label in ("d", "z", "m"))
    kem = TransformOnlyMLKEM(args.parameter_set, signed_cbd_inputs=args.signed_cbd)
    ek, dk = kem.keygen(d, z)
    shared, ciphertext = kem.encaps(ek, m)
    if kem.decaps(dk, ciphertext) != shared:
        raise ArithmeticError("Shared keys differ")
    print(json.dumps({
        "parameter_set": args.parameter_set, "profile": kem.profile,
        "roundtrip_passed": True,
        "encapsulation_key_bytes": len(ek), "decapsulation_key_bytes": len(dk),
        "ciphertext_bytes": len(ciphertext),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        "transform_calls": kem.counts, "logical_reads": kem.logical_read_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
