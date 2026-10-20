# Reproducing the checks

## Environment

Use Python 3.10+ and install `requirements.txt`. The release preparation was
tested with Python 3.12 and NumPy 2.4.2. Record your actual interpreter and
dependency versions when reporting a run:

```sh
python --version
python -m pip freeze
```

The root runner always uses its own Python interpreter for child scripts.
It resolves the checkout relative to the script rather than the caller's
working directory. The two families remain independently runnable.

## Official fixtures

ML-KEM fixtures come from NIST ACVP-Server commit
`975de31eb83d87039ec88934fdc47d8c312b892d`, using the final FIPS 203 suites:

| Fixture group | Distinct cases |
| --- | ---: |
| Key generation | 75 |
| Encapsulation | 150 |
| Decapsulation | 90 |
| Encapsulation-key checks | 60 |
| Decapsulation-key checks | 60 |
| Total | 435 |

The `tr1` suite includes seeded decapsulation inputs. Official negative cases
are successful tests when the implementation rejects the input as specified;
their expected `testPassed=false` is not itself a failed verification.

Saber fixtures come from KULeuven-COSIC/SABER commit
`f7f39e4db2f3e22a21e1dd635e0601caae2b4510`: 100 cases per parameter set.
The supplied `official_inputs.jsonl` contains random inputs captured from the
official deterministic KAT run. It is not a file of expected KEM outputs.
Expected keys, ciphertexts and shared secrets are read independently from the
official `.rsp` files.

Each module retains fixture provenance and SHA-256 checks. Fixture digests are
fixed independently of generated reports. Source-code hashes in local reports
identify the implementation being checked; changing comments can legitimately
change a source hash without changing an official fixture hash.

## Run the current verification

```sh
python scripts/verify.py --suite smoke
python scripts/verify.py --suite full
```

The full ML-KEM check verifies that binary weight tensors and CSV planes contain
only `0/1`, covers both transform-only profiles, and runs the inverse-Montgomery
audit. The full Saber check covers binary weights, all official cases on the
binary backend, independent auxiliary arithmetic, and modified-ciphertext
rejection. The command does not silently reuse previous pass reports.

## Rebuild weights

Both modules include their source-matrix generator and binary-plane exporters.
Follow the commands in the corresponding module README:

- [ML-KEM generators](../mlkem_full_flow/README.md)
- [Saber generators](../saber_full_flow/README.md)

Reconstruction tests compare generated matrices and planes to the transform
definitions and independent arithmetic. Existing invalid artifacts must cause
a verification failure rather than becoming new expected results.

## Rebuild the official Saber reference

The optional native reference requires GCC or another supported C compiler,
plus OpenSSL development headers and libraries. It is built from the vendored,
pinned source; compiled binaries are not shipped.

```sh
python scripts/verify.py --family saber --suite reference
```

The native build and RNG-capture instrumentation are explained in the Saber
README and the official-reference directory. Standard Python verification
does not require a native build.

## Outputs and comparison boundaries

The root report is `generated/verification/latest.json`. Module reports contain
individual case results, transform checks and source/fixture provenance.
Inspect the status and exit code, not just the presence of a report file.

The ML-KEM suite executes 435 distinct cases under two input profiles, giving
870 executions. The Saber suite executes its 300 distinct cases on the binary
backend. Repetition across profiles does not increase the number of distinct cases.
All supplied fixture keys are public test data. Trace exports intentionally
contain sensitive intermediate values and should be used with test inputs.
