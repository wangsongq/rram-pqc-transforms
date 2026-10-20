# Third-party material and attribution

The root [MIT license](LICENSE) covers the original project code and
documentation. It does not replace the following upstream notices.

## NIST ACVP test vectors

The JSON fixtures in `mlkem_full_flow/golden_vectors/ML-KEM-*` are unmodified
copies of public NIST ACVP-Server sample vectors. The National Institute of
Standards and Technology is the source of this material.

- Source: [usnistgov/ACVP-Server](https://github.com/usnistgov/ACVP-Server)
- Pinned commit: `975de31eb83d87039ec88934fdc47d8c312b892d`
- Original terms: [the pinned README license section](https://github.com/usnistgov/ACVP-Server/blob/975de31eb83d87039ec88934fdc47d8c312b892d/README.md#license)
- Complete notice: [licenses/NIST-ACVP-NOTICE.txt](licenses/NIST-ACVP-NOTICE.txt)
- Fixture URLs and checksums: `mlkem_full_flow/golden_vectors/provenance.json`

Packaging changes dated 2026-10-20 add portable runners, documentation and
provenance checks. The upstream test JSON bytes and their expected outputs
are unchanged. Passing these local checks is not NIST CAVP/CMVP certification
or NIST endorsement.

## Official Saber source and KAT fixtures

The C reference and `.req`/`.rsp` fixtures are copied from the official
[KULeuven-COSIC/SABER repository](https://github.com/KULeuven-COSIC/SABER),
commit `f7f39e4db2f3e22a21e1dd635e0601caae2b4510`.

The upstream public-domain / Unlicense declaration is preserved in
[saber_full_flow/official_reference/LICENSE](saber_full_flow/official_reference/LICENSE).
The original source files and their individual notices remain unchanged:

- `rng.c`, `rng.h` and `test/PQCgenKAT_kem.c` retain the Lawrence E. Bassham
  2017 notices.
- `cbd.c`, `cbd.h`, `verify.c` and `verify.h` retain their CRYSTALS-Kyber
  attribution and author lists.
- `fips202.c` retains its attribution to the public-domain SUPERCOP Keccak
  implementation and TweetFips202.

Original URLs, byte sizes and checksums are recorded in
`saber_full_flow/golden_vectors/provenance.json`. The upstream README files
describe the larger upstream repository; only the reference implementation
and the selected Round-3 KAT fixtures are bundled here.

Local additions include `official_reference/capture_rng.c` and
`official_reference/validate_reference.py`. The verification script selects
the parameter set and renames the RNG symbol in temporary build copies to
capture public test inputs. It does not modify the bundled upstream originals.
The derived `golden_vectors/official_inputs.jsonl` contains these captured
random draws; expected KEM outputs remain in the original `.rsp` files.
Packaging and build-tool changes dated 2026-10-20 make the reproduction use
standard compiler/OpenSSL development interfaces and portable paths.

## External dependencies

NumPy is installed separately for numerical computation.
OpenSSL and a C compiler are optional for rebuilding the
official Saber reference. Their source and compiled libraries are not bundled
or relicensed by this repository.
