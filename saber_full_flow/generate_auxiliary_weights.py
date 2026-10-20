#!/usr/bin/env python3
"""Generate source-verified Saber matrices and 50 physical binary weight CSVs."""
import json
import sys
sys.dont_write_bytecode = True
from auxiliary_ntt import export_weights


def main():
    report = export_weights()
    print(json.dumps({"source_sha256": report["source"]["sha256"],
                      "branches": report["branches"],
                      "csv_planes": report["total_csv_planes"],
                      "artifact_files": len(report["artifacts"])}, indent=2))


if __name__ == "__main__":
    main()
