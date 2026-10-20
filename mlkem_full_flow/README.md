# ML-KEM transforms and complete-flow validation

This directory provides the complete readable ML-KEM-512, ML-KEM-768 and
ML-KEM-1024 software flows, binary transform-weight generation, and independent
official test fixtures. `TransformOnlyMLKEM` is the recommended array model:
only NTT/INTT are replaced; sampling, quadratic base multiplication, hashing,
compression, key checks and implicit rejection retain the reference code.

The array model uses exact binary intersections and population counts, followed
by integer reconstruction. It does not model a physical RRAM array, ADC errors,
latency, power, or a fabricated chip. This research code is not constant-time
and is not a production cryptographic library.

## Quick start

Use Python 3.10 or newer. NumPy is the only dependency for arithmetic and tests.
The bundled fixtures permit validation without network access.

```sh
cd mlkem_full_flow
python -m pip install -r requirements.txt
python matrices/generate_matrices.py
python rram_binary/generate_weights.py
python rram_intt_montgomery_weights/generate_weights.py
python example.py --parameter-set ML-KEM-768
python check_binary_weights.py
python check_transform_only.py
python check_intt_montgomery.py
```

The generators verify existing files rather than silently replacing changed
weights. They can also regenerate all weights from the mathematical definition:
the matrix generator needs no pre-existing NPZ or external workspace.

`check_transform_only.py` verifies 435 distinct NIST ACVP cases against the
official expected results for each of two input profiles, for **870 executions**.
The cases comprise 75 key-generation, 165 encapsulation/decapsulation/key-check,
and 195 revised encapsulation/decapsulation/key-check cases. Negative key-check
cases pass when the implementation returns the expected `False` result.
It additionally compares linked flows, tampered ciphertext rejection and every
observed ordinary-domain transform boundary with the butterfly reference.

`check_intt_montgomery.py` checks the fused weights, REDC32 bounds and edge cases,
matrix/bit-count equivalence, trace reconstruction, invalid inputs and the
unchanged digital-method boundary. Generated reports go to
`transform_only_results/`; these are reproducible outputs rather than fixtures.

## Recommended API

Run from this directory, or add this directory to the Python import path.

```python
from rram_transform_only import TransformOnlyMLKEM

kem = TransformOnlyMLKEM("ML-KEM-768", signed_cbd_inputs=False)
# Public fixed seeds for a reproducible example only.
ek, dk = kem.keygen(d=bytes(range(32)), z=bytes(range(32, 64)))
shared_key, ciphertext = kem.encaps(ek, m=bytes([7]) * 32)
assert kem.decaps(dk, ciphertext) == shared_key

x = [i % 3329 for i in range(256)]
y = kem.ntt(x)
assert kem.intt(y) == x
```

`keygen` returns `(ek, dk)`, `encaps` returns `(shared_key, ciphertext)`, and
`decaps` returns a 32-byte shared key. Public `ntt` and `intt` accept exactly
256 integer coefficients, canonicalize them modulo 3329 and return a list of
256 ordinary canonical residues. Incorrect sizes/types raise `ValueError`.
`counts` records complete 256-coefficient transforms; `reset_counts()` resets
these and the logical input/read counters.

An optional `trace(stage, values)` callback receives intermediate values and
binary partial counts. Traces include seeds, private keys and shared secrets.
Use the public test fixtures when exporting traces.

## Mapping and arithmetic

For each 256-coefficient input, the even and odd 128-coefficient lanes use the
same matrix. With `i` indexing outputs and `j` inputs,

```text
q = 3329, zeta = 17
W[i,j] = zeta^((2*BitRev7(i)+1)*j) mod q
V[j,i] = 128^-1 * zeta^(-(2*BitRev7(i)+1)*j) mod q
128^-1 mod q = 3303
```

Physical input rows and output columns store binary bit slices of the transpose
of the mathematical matrix. Every physical weight is strictly **0 or 1**; no
high/low RRAM conductance value or conductance unit is stored. The ordinary
matrices contain integer residues modulo 3329 and serve as mathematical
references, not conductance maps. Each 128-by-128 matrix has twelve least-significant-bit-first binary
planes; each plane occupies four 64-by-64 logical tiles. The two parity lanes
reuse these weights. Tile outputs are nonnegative counts in `0..64`, and the
two input-row tiles merge to counts in `0..128`. Digital shifts and additions
reconstruct all input-bit/weight-bit contributions. Counts are promoted to
64-bit integers before shifting.

The recommended forward transform uses ordinary `W`. For the inverse,
`K = (2^32 * V) mod q` is programmed instead of `V`. An exact integer dot
product with `K` is followed by one Montgomery reduction, REDC32, which removes
the radix and returns ordinary residues. `V` already contains the inverse
normalization: applying 3303 again would be incorrect. The maximum dot product
is `128*(q-1)^2 = 1,417,674,752 < 2^32`, inside REDC32's input range.

| Artifact | Shape / interpretation |
|---|---|
| `matrices/mlkem_shared_matrices.npz` | `W128`, `W128_inv`, each `[128,128]`, mathematical `[output,input]` |
| `rram_binary/mlkem_binary_weights.npz` | Ordinary forward/inverse planes `[12,128,128]`, physical `[bit,input,output]` |
| `rram_intt_montgomery_weights/intt_montgomery_weights.npz` | Compensated inverse planes `[12,128,128]` plus layout/radix metadata |
| `*_bitNN.csv` | Physical `[input,output]` binary planes, without headers |
| Runtime `tile_counts` | `[lane=2,input_bit=12,weight_bit=12,tile=4,output=64]` for the unsigned profile |

One unsigned transform evaluates `2*12*12*4 = 1152` logical tile-read vectors.
These are model operations, not hardware cycles. Ordinary forward plus
compensated inverse need 96 logical 64-by-64 tiles when both directions are
resident. The release also supplies ordinary inverse weights for comparison;
those are an alternative inverse representation, not an additional required
active bank. `check_binary_weights.py` verifies every physical NPZ plane/tile
and every CSV entry is 0 or 1, and that each CSV matches its NPZ plane. NPZ
metadata such as the modulus, tile offsets and radix is checked separately
from the physical weight arrays.

`signed_cbd_inputs=False` uses twelve input slots everywhere. The optional
`True` profile (the class default) uses three binary pulse slots with digital
weights `[1,2,-4]` at known CBD forward-transform stages. It does not infer a
schedule from secret magnitudes and does not apply negative array pulses.
Unknown forward stages and every inverse transform retain twelve slots.
The profile changes logical input counts, not the mathematical transform.

## Validation references

`mlkem.py` retains the conventional butterfly transform and dense integer
matrix transform needed for independent arithmetic validation. The runtime
binary-array path uses binary intersections and population counts, without a
dense matrix-product shortcut. `validation.py` reads the pinned official
fixtures, checks their hashes and executes the public ACVP case schema.

`smoke_test.py` is a quick check of all three parameter sets, including exact
agreement with the butterfly reference, shared-key roundtrips, modified
ciphertext rejection, transform boundaries and malformed-input rejection.

## Provenance

`golden_vectors/provenance.json` pins the public NIST ACVP-Server repository
revision, upstream URLs, case counts and byte-level SHA-256 digests. The supplied
JSON files and their fixed digests are unchanged. Tests verify their integrity
before comparing outputs; expected values are never generated by this model.
Passing these examples is not NIST CAVP/CMVP certification.

`matrices/provenance.json` records the closed-form generation and inverse
checks. Each weight directory records the generated artifact hashes and
current source hashes. `core_provenance.json` describes the release-local
implementation and mathematical contract without depending on private paths.
See the repository-level third-party notices for fixture licensing.
