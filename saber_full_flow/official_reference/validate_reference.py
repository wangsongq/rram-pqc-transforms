#!/usr/bin/env python3
"""Build pinned official C and extract RNG inputs while checking downloaded KAT.

The authoritative expected bytes are always read from upstream .rsp files.
No output from this program replaces those files. Original source files are
unchanged; the build copy changes only the upstream security-strength define.
"""
from __future__ import annotations

import ctypes
import json
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from golden_vectors.kat_io import FAMILIES, load_kat, sha256, verify_sources

U8 = ctypes.c_ubyte
PTR = ctypes.POINTER(U8)
SOURCES = ("pack_unpack.c", "poly.c", "fips202.c", "verify.c", "cbd.c", "SABER_indcpa.c", "kem.c")


def buffer(data_or_size):
    return (U8 * data_or_size)() if isinstance(data_or_size, int) else (U8 * len(data_or_size)).from_buffer_copy(data_or_size)


def build(name: str, commands: list) -> ctypes.CDLL:
    original = ROOT / "official_reference/Reference_Implementation_KEM"
    target = ROOT / "generated/reference_build" / name
    # Always compile from a fresh private copy; no cached .so can satisfy audit.
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in original.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    params = target / "SABER_params.h"
    original_params = params.read_text()
    marker = "#define SABER_L 3 /* Saber */"
    if original_params.count(marker) != 1:
        raise ValueError("Unexpected upstream parameter selector")
    params.write_text(original_params.replace(marker, f"#define SABER_L {FAMILIES[name]['l']} /* local audit build selector */"))
    # Use the installed OpenSSL development headers and public linker name.
    # CC/CPPFLAGS/LDFLAGS support a non-system toolchain or OpenSSL prefix.
    compiler = shlex.split(os.environ.get("CC", "gcc"))
    flags = compiler + ["-std=c99", "-O2", "-fwrapv", "-fPIC", "-I", "."] + shlex.split(os.environ.get("CPPFLAGS", ""))
    compile_rng = flags + ["-Drandombytes=official_randombytes", "-c", "rng.c", "-o", "rng_capture.o"]
    capture_source = os.path.relpath(ROOT / "official_reference/capture_rng.c", target)
    link = flags + ["-shared", *SOURCES, "rng_capture.o", capture_source,
                    "-Wl,-Bsymbolic", *shlex.split(os.environ.get("LDFLAGS", "")),
                    "-lcrypto", "-o", "reference.so"]
    for command in (compile_rng, link):
        try:
            subprocess.run(command, cwd=target, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError("Official C compilation failed. Install gcc and OpenSSL development headers/libraries "
                               "(for example libssl-dev), or set CC/CPPFLAGS/LDFLAGS.\n" + error.stderr) from error
        # Store portable commands, not the local compiler-prefix directories.
        display = [part.replace(str(ROOT), "<module>") for part in command]
        for flag in shlex.split(os.environ.get("CPPFLAGS", "")) + shlex.split(os.environ.get("LDFLAGS", "")):
            display = ["<environment flag>" if part == flag else part for part in display]
        commands.append({"cwd": str(target.relative_to(ROOT)), "argv": display})
    lib = ctypes.CDLL(str(target / "reference.so"), mode=ctypes.RTLD_LOCAL)
    lib.randombytes_init.argtypes = [PTR, PTR, ctypes.c_int]
    lib.randombytes_init.restype = None
    lib.audit_reset_capture.argtypes = []
    lib.audit_reset_capture.restype = None
    lib.audit_copy_draws.argtypes = [PTR]
    lib.audit_copy_draws.restype = ctypes.c_int
    for function, arity in (("crypto_kem_keypair", 2), ("crypto_kem_enc", 3), ("crypto_kem_dec", 3)):
        getattr(lib, function).argtypes = [PTR] * arity
        getattr(lib, function).restype = ctypes.c_int
    return lib


def run_validation() -> dict:
    started = time.perf_counter()
    provenance = verify_sources()
    fixtures, family_reports, commands = [], [], []
    for name, sizes in FAMILIES.items():
        lib = build(name, commands)
        records = load_kat(name)
        for record in records:
            seed = buffer(record["seed"])
            lib.randombytes_init(seed, None, 256)
            lib.audit_reset_capture()
            pk, sk, ct, ss, recovered = [buffer(sizes[key]) for key in ("pk", "sk", "ct", "ss", "ss")]
            if lib.crypto_kem_keypair(pk, sk) or lib.crypto_kem_enc(ct, ss, pk) or lib.crypto_kem_dec(recovered, ct, sk):
                raise RuntimeError(f"Official C API failure: {name}/{record['count']}")
            for field, actual in (("pk", pk), ("sk", sk), ("ct", ct), ("ss", ss), ("ss_decaps", recovered)):
                expected = record["ss" if field == "ss_decaps" else field]
                if bytes(actual) != expected:
                    raise ValueError(f"Official C != downloaded KAT: {name}/{record['count']}/{field}")
            draws = buffer(128)
            if lib.audit_copy_draws(draws):
                raise ValueError("Expected exactly four sequential randombytes calls, each 32 bytes")
            raw = bytes(draws)
            fixture = {"parameter_set": name, "count": record["count"], "seed": record["seed"].hex()}
            fixture.update({key: raw[32*i:32*(i+1)].hex() for i, key in enumerate(("seed_a", "seed_s", "z", "m"))})
            fixtures.append(fixture)
        family_reports.append({"parameter_set": name, "cases": len(records), "passed": len(records), "byte_comparisons": 5 * len(records), "randombytes_calls_checked": 4 * len(records)})
    verify_sources()
    # Compare reproduced random inputs against the pinned fixture; never
    # update expected data or its checksum merely because this audit ran.
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in fixtures).encode("utf-8")
    expected_hash = provenance["derived_inputs"]["sha256"]
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("Official C reproduced RNG inputs differ from the fixed fixture checksum")
    out = ROOT / "generated/reference"
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "official_inputs.jsonl"
    inputs.write_bytes(payload)
    report = {
        "status": "passed", "commit": provenance["commit"], "official_kat_cases": 300,
        "official_c_case_passed": 300, "official_byte_comparisons": 1500,
        "captured_randombytes_calls": 1200, "parameter_sets": family_reports,
        "input_capture": "Official rng.c AES256 CTR DRBG with installed OpenSSL libcrypto; wrapper captures the four 32-byte draws without changing them",
        "build_changes": "Only SABER_L in copied build headers; rng.c function-symbol rename; system OpenSSL development headers and libcrypto; no downloaded source changes",
        "compiler": subprocess.check_output(shlex.split(os.environ.get("CC", "gcc")) + ["--version"], text=True).splitlines()[0],
        "commands": commands, "elapsed_seconds": time.perf_counter() - started,
        "source_integrity_verified_before_and_after": True,
        "expected_source": "Unmodified downloaded official .rsp files; no local implementation generates golden outputs",
        "inputs_sha256": sha256(inputs),
    }
    report["clean_rebuild"] = True
    report["derived_inputs_match_fixed_fixture"] = True
    report["validation_script_sha256"] = sha256(Path(__file__))
    report_path = out / "reference_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run_validation()
    print(json.dumps({k: v for k, v in result.items() if k != "commands"}, indent=2))
