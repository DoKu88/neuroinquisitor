"""
Run every non-utility example script under this directory.  A single failure
does not stop the run.  Results are written to
  <repo_root>/outputs/status_run/run_<timestamp>.yaml
"""

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml  # PyYAML

EXAMPLES_DIR = Path(__file__).parent
REPO_ROOT = EXAMPLES_DIR.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "status_run"

SKIP_PATTERNS = {"_utils.py", "run_all.py"}


def is_example(path: Path) -> bool:
    return path.suffix == ".py" and not any(
        path.name.endswith(p) for p in SKIP_PATTERNS
    )


def run_script(script: Path) -> dict:
    rel = script.relative_to(EXAMPLES_DIR)
    print(f"━━━ {rel} ━━━", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
    )
    elapsed = round(time.monotonic() - t0, 2)
    status = "passed" if result.returncode == 0 else "failed"
    print(f"  → {status} (exit {result.returncode}, {elapsed}s)\n", flush=True)
    return {
        "script": str(rel),
        "status": status,
        "exit_code": result.returncode,
        "duration_s": elapsed,
    }


def main() -> None:
    scripts = sorted(
        p for p in EXAMPLES_DIR.rglob("*.py") if is_example(p)
    )

    if not scripts:
        print("No example scripts found.")
        return

    records = [run_script(s) for s in scripts]

    passed = [r for r in records if r["status"] == "passed"]
    failed = [r for r in records if r["status"] == "failed"]

    print("━━━ Summary ━━━")
    print(f"  Passed : {len(passed)}")
    print(f"  Failed : {len(failed)}")
    if failed:
        for r in failed:
            print(f"    ✗ {r['script']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = OUTPUT_DIR / f"run_{timestamp}.yaml"

    log = {
        "run_at": timestamp,
        "summary": {
            "total": len(records),
            "passed": len(passed),
            "failed": len(failed),
        },
        "results": records,
    }

    with open(log_path, "w") as f:
        yaml.dump(log, f, default_flow_style=False, sort_keys=False)

    print(f"\nLog written to: {log_path.relative_to(REPO_ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
