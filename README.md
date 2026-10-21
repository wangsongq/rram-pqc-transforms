# Binary NTT/INTT for ML-KEM and Saber

Complete Python implementations of ML-KEM and Saber with binary NTT/INTT
weights, integer reconstruction, inverse-normalization fusion and official
test-vector validation. Both modules include KeyGen, Encaps and Decaps and
can be run independently.

| Family | Parameter sets | Transform mapping |
| --- | --- | --- |
| ML-KEM | ML-KEM-512, ML-KEM-768, ML-KEM-1024 | Modulo 3329; two 128-coefficient lanes; 12 binary weight planes per operator |
| Saber | LightSaber, Saber, FireSaber | Auxiliary primes 3329 and 7681; four 64-coefficient lanes per prime; 12 and 13 binary weight planes |

The binary backend explicitly evaluates binary weights and input bits, then
reconstructs integer sums. All binary weight planes contain only `0` and `1`;
there are no RRAM conductance values. Mathematical source matrices contain
modular integer coefficients before bit decomposition.

## Quick start

Python 3.10 or later and NumPy are required.

```sh
git clone https://github.com/wangsongq/rram-pqc-transforms.git
cd rram-pqc-transforms
python3 -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/verify.py --suite smoke
python scripts/verify.py --suite full
```

Run a complete KEM example for either family:

```sh
python mlkem_full_flow/example.py --parameter-set ML-KEM-768
python saber_full_flow/example.py
```

The official test fixtures are included. After installing dependencies, the
Python checks run offline and do not fetch expected outputs.

| Command | Scope |
| --- | --- |
| `python scripts/verify.py --suite smoke` | Fast checks for both families |
| `python scripts/verify.py --suite full` | Binary-weight checks, official cases and arithmetic checks for both families |
| `python scripts/verify.py --family mlkem --suite full` | ML-KEM only |
| `python scripts/verify.py --family saber --suite full` | Saber only |
| `python scripts/verify.py --family saber --suite reference` | Rebuild and check the pinned official Saber C reference |
| `python scripts/verify.py --suite full --list` | Show the underlying commands without running them |

Full checks take longer than smoke checks. A failed check returns a nonzero
exit status. The runner writes logs and a machine-readable report to
`generated/verification/`; each family also writes its own detailed reports.
Generated outputs are excluded from version control.

## Contents

| Directory | Contents |
| --- | --- |
| [`mlkem_full_flow/`](mlkem_full_flow/README.md) | Complete FIPS 203 KEM, butterfly/matrix/binary transforms, weight generators and NIST ACVP fixtures |
| [`saber_full_flow/`](saber_full_flow/README.md) | Complete Round-3 Saber KEM, auxiliary-prime transforms, CRT, weight generators and official KAT fixtures |
| `scripts/verify.py` | Common verification entry point |
| [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) | Matrix orientation, normalization, CRT and computation boundaries |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Test coverage, fixture provenance and reproducibility instructions |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | Results from the local release checks |

## Which implementation should I use?

For ML-KEM, the recommended binary implementation is `TransformOnlyMLKEM` in
`rram_transform_only.py`, with `signed_cbd_inputs=False`. It replaces NTT/INTT
while retaining the digital KEM operations. Its inverse weights combine
normalization with Montgomery scaling; the inverse output is converted back to
ordinary residues. The optional signed-CBD input profile changes input encoding
at known sampling stages while preserving the same binary weights.

For Saber, use `Saber(..., backend="rram_binary")` in `saber.py`. Each polynomial
product uses auxiliary moduli 3329 and 7681, four 64-coefficient lanes per
modulus, quartic products, and centered CRT reconstruction. Schoolbook and
matrix arithmetic support independent correctness checks.

See each module's README for API examples and weight-generation commands.

## Regenerate the weights

The source matrices and binary weights are included. To reproduce them from
their mathematical definitions, run these commands from the repository root:

```sh
python mlkem_full_flow/matrices/generate_matrices.py
python mlkem_full_flow/rram_binary/generate_weights.py
python mlkem_full_flow/rram_intt_montgomery_weights/generate_weights.py

python saber_full_flow/matrices/source_build_weights.py
python saber_full_flow/generate_auxiliary_weights.py
```

CSV files contain binary planes with input rows and output columns. NPZ files
store the same planes and layout metadata; source-matrix NPZ files store
modular integers. Bit 0 is the least significant bit. Existing inconsistent
weight files cause an error rather than silently becoming new expected data.

## Verification scope

- **ML-KEM:** 435 distinct official NIST ACVP cases. The current transform-only
  checker runs both unsigned and signed-CBD profiles, giving 870 case executions.
- **Saber:** 300 distinct official Round-3 KAT cases run on the binary backend,
  with byte-for-byte public-key, secret-key, ciphertext and shared-secret checks.
- Additional checks exercise transform identities, representation boundaries,
  CRT bounds and rejection of modified ciphertexts.

The prepared code passed all official cases, including 870 ML-KEM executions
across two profiles and 300 Saber executions with 1,500 output comparisons.
See [validation results](docs/VALIDATION.md) for the environment, arithmetic
checks and optional official C-reference reproduction.

Repeated executions across profiles/backends are not additional independent
test cases. Passing these fixtures is software verification, not NIST
certification or evidence of physical-array accuracy. These inspectable Python
implementations are not constant-time production cryptography. Fixtures and
example seeds are public test material; tracing records keys and randomness.

## Optional dependencies

The Saber C-reference rebuild requires a C compiler and OpenSSL development
headers/libraries. On Debian/Ubuntu, install `build-essential` and `libssl-dev`.
This is optional for the Python verification suite.

## License

Original project code and documentation are provided under the [MIT license](LICENSE).
Bundled official fixtures and reference code retain their upstream notices and
terms; see the [NIST ACVP notice](licenses/NIST-ACVP-NOTICE.txt) and
[official Saber license](saber_full_flow/official_reference/LICENSE).
