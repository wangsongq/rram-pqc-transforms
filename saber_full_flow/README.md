# Saber auxiliary-prime transforms and complete KEM validation

This module implements the Round-3 SHAKE/SHA3, centered-binomial variants of
LightSaber, Saber and FireSaber. It contains complete KeyGen, Encaps and Decaps,
two auxiliary-prime NTT/INTT branches, reproducible binary weight generation,
and byte-for-byte checks against the official Saber known-answer tests.

The `rram_binary` backend is an exact software model: weights and input slices
are binary, and array sums are computed using AND/popcount followed by digital
shift/add reconstruction. It does not model measured conductance, analog noise,
physical timing or energy. This readable research implementation is not a
constant-time cryptographic library; optional traces contain secret values.

## Run locally

Use Python 3.10 or later and NumPy. No network access or C compiler is required
for the ordinary Python checks; official KAT outputs and derived input draws
are included.

From this module directory:

```bash
python3 example.py
python3 smoke_test.py
```

The example performs one complete KeyGen/Encaps/Decaps roundtrip without
exporting keys or traces. The smoke test checks all binary weights, executes
official case 0 for each parameter set on the `rram_binary` backend (3 cases,
15 output comparisons), and runs nine altered-ciphertext rejection checks.

For complete validation:

```bash
python3 check_binary_weights.py
python3 check_golden.py
python3 check_auxiliary.py
python3 check_rejection.py
```

`check_golden.py` defaults to the binary backend and executes **300 distinct
official cases with 1,500 serialized-output comparisons**. `check_auxiliary.py`
checks transform identities, all basis vectors, integer arithmetic, CRT bounds,
binary counts and packing/rounding boundaries against independent oracles.
`check_rejection.py` checks three ciphertext alterations for each parameter set.
Failures produce a nonzero process exit status.

Reports are written to `generated/results/`, which can be removed after
inspection. It is not needed to run the implementation. There are no plotting,
large trace-export or backend-comparison experiment entry points.

## Public API

```python
from saber import Saber

kem = Saber("Saber", backend="rram_binary")
public_key, secret_key = kem.keygen()
shared_key, ciphertext = kem.encaps(public_key)
assert kem.decaps(secret_key, ciphertext) == shared_key
```

`keygen(seed_a, seed_s, z)` accepts three raw 32-byte random draws, and
`encaps(public_key, m)` accepts one raw 32-byte draw. The seed processing and
message hashing occur inside the API, matching the official C call order.
Omitted randomness is obtained from `secrets.token_bytes(32)`. Decapsulation
re-encrypts the recovered message and selects the stored fallback `z` when the
ciphertext does not match, then applies the official KDF.

| Parameter set | Module rank | CBD parameter `mu` | Ciphertext scalar bits | Public key | Secret key | Ciphertext |
|---|---:|---:|---:|---:|---:|---:|
| LightSaber | 2 | 10 | 3 | 672 bytes | 1,568 bytes | 736 bytes |
| Saber | 3 | 8 | 4 | 992 bytes | 2,304 bytes | 1,088 bytes |
| FireSaber | 4 | 6 | 6 | 1,312 bytes | 3,040 bytes | 1,472 bytes |

All use 256 coefficients, native modulus 8192, rounded modulus 1024 and a
32-byte shared key.

The binary implementation and two independent arithmetic oracles share the
same sampling, packing, hashing, rounding and KEM logic:

* `schoolbook`: independent exact negacyclic-convolution oracle, without NTT or FFT.
* `matrix`: integer matrix-product oracle under both auxiliary primes.
* `rram_binary`: binary weight/input planes, packed AND/popcount and exact
  digital reconstruction. It does not replace the binary kernel with a dense
  matrix product.

## Four lanes, inverse fusion and CRT

For each auxiliary prime `p` in `{3329, 7681}`, the 256 coefficients are
interleaved into four 64-element lanes:

```text
lane[r][j] = polynomial[4*j + r], r = 0..3, j = 0..63
gamma[i]  = psi**(2*bitrev6(i) + 1) mod p
W[i,j]    = gamma[i]**j mod p
V[j,i]    = inverse(64,p) * gamma[i]**(-j) mod p
K         = (2**32 * V) mod p
```

The primitive 128th roots are `psi = 1915` for 3329 and `psi = 5258` for 7681.
The forward transform stores ordinary-domain `W` and reduces the reconstructed
sum modulo `p`. The inverse stores `K`, which already includes the `64^-1`
normalization. A single REDC32 after reconstruction removes the `2^32` factor.
There is no second normalization or REDC operation.

Each branch forms 64 quartic products in `F_p[X]/(X^4-gamma[i])`. Dot products
accumulate the rank terms before the inverse transform. CRT then reconstructs
the signed integer coefficient, **centers it modulo 3329*7681 = 25,570,049**,
and only then reduces it modulo 8192 or 1024. Reducing a non-centered CRT value
directly to a power of two gives incorrect negative coefficients.

The right operand is a small secret centered modulo 8192 even when the target
ring is modulo 1024. The implementation checks its support and the sufficient
uniqueness condition `2*B < 25,570,049` before each dot product.

## Weight files and regeneration

The 3329 branch uses 12 weight planes and the 7681 branch uses 13. Separate
forward and inverse operators require **50 logical 64x64 binary planes** in
total; all four lanes reuse each operator. Every exported physical weight is
strictly **0 or 1**; no RRAM conductance values are assigned. Dense `W`, `V` and
`K` in `matrices/` contain modular integers for generation and arithmetic
checking, not analog or multilevel conductance weights. This is a logical storage count,
not a claim that all weights are resident on one experimental chip.

Physical CSV orientation is
`plane[weight_bit, input_row, output_column] = bit(W.T)` for NTT and `bit(K.T)`
for INTT. Bit 0 is the least significant bit. The input encoding uses 12 or
13 unsigned bit slices, including zero slices.

```bash
python3 matrices/source_build_weights.py
python3 generate_auxiliary_weights.py
```

The first command reconstructs the ordinary mathematical source independently
and checks it against six-stage butterfly arithmetic and direct negacyclic
convolution. The second checks the source checksum, adds inverse fusion, and
writes/verifies all CSV/NPZ artifacts. Both are self-contained. NPZ encoding
uses fixed ZIP timestamps so the binary arrays can be regenerated byte-exactly.
`check_binary_weights.py` validates every CSV token before conversion, checks
all four binary NPZ stacks, and compares each stack with its CSV planes. A token
such as `0.1` fails validation rather than being truncated to `0`.
Existing generated weight files with differing contents are rejected rather
than silently overwritten. If omitted from a source-only checkout, the source
NPZ and final weight files are generated automatically.

| File | Purpose |
|---|---|
| `saber.py` | Complete KEM and independent schoolbook backend |
| `auxiliary_ntt.py` | Four-lane transforms, binary engine, quartic products and centered CRT |
| `matrices/source_build_weights.py` | Independent source-matrix generator and arithmetic checks |
| `matrices/source_weights.npz` | Ordinary mathematical `W` and inverse matrices |
| `matrices/saber_auxiliary_matrices.npz` | `W`, `V`, fused `K`, roots and reducer parameters |
| `rram_binary/*_bit*.csv` | Fifty physical binary weight planes |
| `rram_binary/saber_binary_weights.npz` | The same planes in one NumPy archive |
| `rram_binary/provenance.json` | Equations, orientation, code hashes and artifact checksums |
| `golden_vectors/KAT/` | Pinned upstream `.req` and `.rsp` files |
| `golden_vectors/official_inputs.jsonl` | Reproduced raw RNG draws; no locally generated expected outputs |
| `official_reference/` | Unmodified official C sources/notices and local audit tools |

## Official source and optional C reproduction

The upstream source is
[KULeuven-COSIC/SABER](https://github.com/KULeuven-COSIC/SABER), pinned to commit
`f7f39e4db2f3e22a21e1dd635e0601caae2b4510`.
`golden_vectors/provenance.json` records the source URLs, sizes and SHA256
checksums of all upstream files. Expected public keys, secret keys, ciphertexts
and shared keys always come from those official `.rsp` files.

To independently rebuild the official C reference and reproduce the input
draws, install GCC and the OpenSSL development headers/libraries (for example,
`gcc` and `libssl-dev` on Ubuntu), then run:

```bash
python3 official_reference/validate_reference.py
```

The audit recreates `generated/reference_build/` from the pinned C sources and
compiles all three parameter sets from scratch. Only the copied parameter
selector and RNG function symbol are changed for instrumentation. A wrapper
captures the four official 32-byte draws for each case without changing them.
All 300 official cases and 1,500 output comparisons must pass; the reproduced
input JSONL must match the fixed supplied checksum. The command writes its
report and reproduced inputs under `generated/reference/` and does not update
official fixtures or their expected hashes. `CC`, `CPPFLAGS` and `LDFLAGS` may
select a non-system compiler or OpenSSL installation.

The upstream Saber license is retained in `official_reference/LICENSE`.
Per-file notices, including the NIST RNG and KAT-generator notices, are retained
unchanged. The repository-level license applies to the original project code.
