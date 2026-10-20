#!/usr/bin/env python3
"""Run one Saber key-generation, encapsulation and decapsulation example.

No keys, secret values or trace arrays are written to disk.
"""
from saber import Saber


def main() -> None:
    kem = Saber("Saber", backend="rram_binary")
    public_key, secret_key = kem.keygen()
    shared_key, ciphertext = kem.encaps(public_key)
    if kem.decaps(secret_key, ciphertext) != shared_key:
        raise AssertionError("Shared-key agreement failed")
    print("Saber rram_binary: KeyGen, Encaps and Decaps passed.")
    print(f"Public key: {len(public_key)} bytes; ciphertext: {len(ciphertext)} bytes.")


if __name__ == "__main__":
    main()
