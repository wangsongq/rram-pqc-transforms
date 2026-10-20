#!/usr/bin/env python3
"""Run either algorithm's checks using this checkout and this Python interpreter.

Each family intentionally retains its independent reference implementation.
Subprocesses avoid module-name collisions between their standalone scripts.
Commands run from the corresponding family directory, so invoking this runner
from another working directory does not introduce an external path dependency.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("mlkem", "saber")
DIRECTORIES = {
    "mlkem": ROOT / "mlkem_full_flow",
    "saber": ROOT / "saber_full_flow",
}


def commands_for(family: str, suite: str) -> list[list[str]]:
    """Select checks without silently replacing full tests with cached reports."""
    if suite == "smoke":
        return [["smoke_test.py"]]
    if suite == "full":
        if family == "mlkem":
            return [
                ["check_binary_weights.py"],
                ["check_transform_only.py"],
                ["check_intt_montgomery.py"],
            ]
        return [
            ["check_binary_weights.py"],
            ["check_golden.py"],
            ["check_auxiliary.py"],
            ["check_rejection.py"],
        ]
    if suite == "reference":
        if family == "saber":
            return [["official_reference/validate_reference.py"]]
        raise ValueError("The C reference rebuild is specific to Saber.")
    raise ValueError(f"Unknown suite: {suite}")


def main() -> int:
    """Stream child output, retain logs, and propagate any verification failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILIES, "both"), default="both")
    parser.add_argument(
        "--suite", choices=("smoke", "full", "reference"),
        default="smoke",
    )
    parser.add_argument("--list", action="store_true", help="Print commands only.")
    args = parser.parse_args()
    families = FAMILIES if args.family == "both" else (args.family,)
    if args.suite == "reference":
        if args.family == "mlkem":
            parser.error("--suite reference requires --family saber or both")
        families = ("saber",)

    plan = [(family, command)
            for family in families
            for command in commands_for(family, args.suite)]
    if args.list:
        for family, command in plan:
            print(f"{DIRECTORIES[family].name}: python " + " ".join(command))
        return 0

    output = ROOT / "generated" / "verification"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "suite": args.suite,
        "scope": "Software functional verification; host seconds are not hardware latency.",
        "status": "running",
        "steps": [],
    }
    report_path = output / "latest.json"

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    for index, (family, command) in enumerate(plan, 1):
        relative_script = (DIRECTORIES[family] / command[0]).relative_to(ROOT)
        log_path = output / f"{index:02d}_{family}_{args.suite}.log"
        print(f"\n[{index}/{len(plan)}] {relative_script}", flush=True)
        started = time.monotonic()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, *command],
                    cwd=DIRECTORIES[family],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
        except OSError as error:
            print(f"Could not start {relative_script}: {error}", file=sys.stderr)
            code = 1
        report["steps"].append({
            "family": family,
            "command": ["python", str(relative_script), *command[1:]],
            "exit_code": code,
            "wall_seconds": round(time.monotonic() - started, 3),
            "log": str(log_path.relative_to(ROOT)),
        })
        if code:
            report["status"] = "failed"
            save()
            return code
        save()

    report["status"] = "passed"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"\nAll requested checks passed. Report: {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
