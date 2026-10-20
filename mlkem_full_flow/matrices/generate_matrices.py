#!/usr/bin/env python3
"""Generate both 128-by-128 ML-KEM transform matrices from their closed form.

Rows index outputs and columns index inputs. The two parity lanes of a
256-coefficient polynomial share W. Its inverse already includes 128**-1.
No original workspace, downloaded program, or existing matrix is required.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

Q, SIZE, ROOT = 3329, 128, 17
OUTPUT = Path(__file__).resolve().parent
NAME = "mlkem_shared_matrices.npz"
EXPECTED_SHA256 = "b96b6c339da9c9599319094c8171603cba2fb6a338a823790be762f40c8de233"


def generate_arrays() -> dict[str, np.ndarray]:
    """Return canonical int64 arrays W[out,in] and W_inv[out,in].

    gamma[i] = 17**(2*bit_reverse_7(i)+1) mod q.
    W[i,j] = gamma[i]**j; W_inv[j,i] = 128**-1 * gamma[i]**(-j).
    This is the incomplete NTT used by ML-KEM, not a 256th-root full NTT.
    """
    reverse = lambda i: int(f"{i:07b}"[::-1], 2)
    forward = np.array(
        [[pow(ROOT, (2 * reverse(i) + 1) * j, Q) for j in range(SIZE)]
         for i in range(SIZE)], dtype=np.int64)
    scale = pow(SIZE, -1, Q)
    inverse = np.array(
        [[scale * pow(ROOT, -(2 * reverse(j) + 1) * i, Q) % Q
          for j in range(SIZE)] for i in range(SIZE)], dtype=np.int64)
    identity = np.eye(SIZE, dtype=np.int64)
    if not (np.array_equal((inverse @ forward) % Q, identity)
            and np.array_equal((forward @ inverse) % Q, identity)):
        raise ArithmeticError("Closed-form transform matrices are not inverses")
    return {"W128": forward, "W128_inv": inverse}


def export_matrices(output_dir: str | Path = OUTPUT) -> dict:
    """Generate and verify the pinned NPZ, rejecting differing existing data."""
    arrays = generate_arrays()
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays)
    data = payload.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("NPZ encoding differs from the pinned artifact; "
                         "check NumPy/zlib versions before updating the release")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / NAME
    if path.exists() and path.read_bytes() != data:
        raise ValueError("Existing matrix artifact differs from its closed form")
    if not path.exists():
        path.write_bytes(data)
    manifest = {
        "kind": "closed-form algorithmic matrices; not conductance measurements",
        "modulus": Q, "root": ROOT, "shape": [SIZE, SIZE],
        "orientation": "mathematical output rows, input columns",
        "inverse_scale": pow(SIZE, -1, Q),
        "formula": "W[i,j]=17^((2*BitRev7(i)+1)*j) mod 3329",
        "inverse_formula": "W_inv[j,i]=128^-1*17^(-(2*BitRev7(i)+1)*j) mod 3329",
        "checks": {"both_matrix_products_equal_identity_mod_q": True,
                   "pinned_artifact_hash_matches": True},
        "artifact": {"path": NAME, "sha256": digest, "bytes": len(data)},
        "generator": {"path": Path(__file__).name,
                      "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    (output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export_matrices(args.out_dir), indent=2))
