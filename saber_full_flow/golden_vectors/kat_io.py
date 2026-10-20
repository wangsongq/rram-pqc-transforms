"""Strict parser for the pinned, unmodified official SABER KAT files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "LightSaber": {"l": 2, "pk": 672, "sk": 1568, "ct": 736, "ss": 32},
    "Saber": {"l": 3, "pk": 992, "sk": 2304, "ct": 1088, "ss": 32},
    "FireSaber": {"l": 4, "pk": 1312, "sk": 3040, "ct": 1472, "ss": 32},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources() -> dict:
    provenance = json.loads((ROOT / "golden_vectors/provenance.json").read_text())
    for entry in provenance["files"]:
        path = ROOT / entry["local_path"]
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Official source integrity failure: {path}")
    return provenance


def load_kat(name: str) -> list[dict]:
    sizes = FAMILIES[name]
    path = ROOT / f"golden_vectors/KAT/{name}/PQCkemKAT_{sizes['sk']}.rsp"
    records = []
    current = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " = " not in line:
            raise ValueError(f"Malformed KAT line in {path}")
        field, value = line.split(" = ", 1)
        if field == "count" and current:
            records.append(current)
            current = {}
        if field in current:
            raise ValueError(f"Duplicate KAT field {field}")
        current[field] = int(value) if field == "count" else bytes.fromhex(value)
    if current:
        records.append(current)
    if len(records) != 100 or [r["count"] for r in records] != list(range(100)):
        raise ValueError(f"Expected exactly official cases 0..99 for {name}")
    for record in records:
        if set(record) != {"count", "seed", "pk", "sk", "ct", "ss"}:
            raise ValueError(f"Unexpected KAT schema for {name}")
        for field in ("seed", "pk", "sk", "ct", "ss"):
            size = 48 if field == "seed" else sizes[field]
            if len(record[field]) != size:
                raise ValueError(f"Wrong KAT {field} length for {name}")
    return records


def load_inputs() -> dict[tuple[str, int], dict]:
    path = ROOT / "golden_vectors/official_inputs.jsonl"
    provenance = json.loads((ROOT / "golden_vectors/provenance.json").read_text())
    if sha256(path) != provenance["derived_inputs"]["sha256"]:
        raise ValueError("Derived official RNG inputs integrity failure")
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["parameter_set"], row["count"])
        if key in result:
            raise ValueError(f"Duplicate RNG inputs for {key}")
        for field in ("seed", "seed_a", "seed_s", "z", "m"):
            row[field] = bytes.fromhex(row[field])
            if len(row[field]) != (48 if field == "seed" else 32):
                raise ValueError(f"Wrong RNG input length for {key}: {field}")
        result[key] = row
    expected = {(name, count) for name in FAMILIES for count in range(100)}
    if set(result) != expected:
        raise ValueError("Derived RNG inputs must cover exactly all 300 official cases")
    return result
