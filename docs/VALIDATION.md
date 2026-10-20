# Release validation

The local preparation was validated on 2026-10-20 with Python 3.12.3 and
NumPy 2.4.2. These are software functional results, not device measurements.

| Check | Result |
| --- | --- |
| ML-KEM recommended transform-only path | 435 distinct official cases, two input profiles; 870/870 executions passed |
| ML-KEM inverse Montgomery audit | 21,009 scalar comparisons, 512 basis vectors, 12 dense/edge/random vectors and nine invalid inputs passed |
| ML-KEM artifact regeneration | 39 matrix/weight files reproduced byte-for-byte from source |
| ML-KEM binary weights | All six NPZ weight tensors and 36 CSV planes contain only `0/1` |
| Saber complete Python verification | Binary backend: 300/300 distinct official cases and 1,500/1,500 output comparisons passed |
| Saber altered-ciphertext checks | 9/9 implicit-rejection checks passed |
| Saber official C reference | 300/300 cases and 1,500/1,500 output comparisons passed after a clean rebuild |
| Saber official RNG-input reproduction | 1,200 draws reproduced; derived input file matched the fixed SHA-256 |
| Saber artifact regeneration | 55 matrix/weight files reproduced byte-for-byte from source |
| Saber binary weights | All four NPZ weight tensors and 50 CSV planes contain only `0/1` |

Additional checks cover independent arithmetic identities, linked KEM flows
and input boundaries.

Both families also passed the common smoke runner after the clean repository
was copied to a different directory containing spaces and invoked from outside
that checkout. This checks that the packaged scripts resolve their own inputs.

Machine-readable evidence:

- [ML-KEM validation](../mlkem_full_flow/release_validation.json)
- [Saber validation](../saber_full_flow/release_validation.json)

The optional C reference was built with GCC 13.3.0 and Ubuntu OpenSSL 3.0.13
development files. Development headers/libraries were supplied through
`CPPFLAGS` and `LDFLAGS`; no prebuilt reference binary or ABI shim was used.
The ordinary Python suite does not need these build dependencies.

To reproduce the main results, run `python scripts/verify.py --suite full`.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the additional commands and
the distinction between independent cases and repeated backend executions.
