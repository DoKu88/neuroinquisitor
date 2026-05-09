"""
Run every non-utility example script under this directory.  A single failure
does not stop the run.  Results are written to
  <repo_root>/outputs/status_run/run_<timestamp>.yaml
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml  # PyYAML

EXAMPLES_DIR = Path(__file__).parent
REPO_ROOT = EXAMPLES_DIR.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "status_run"
CONFIGS_DIR = EXAMPLES_DIR / "configs"

SKIP_PATTERNS = {"_utils.py", "run_all.py"}


def is_example(path: Path) -> bool:
    return path.suffix == ".py" and not any(
        path.name.endswith(p) for p in SKIP_PATTERNS
    )


def _load_config(script: Path) -> dict | None:
    key = str(script.relative_to(EXAMPLES_DIR)).replace("/", "_").removesuffix(".py")
    config_path = CONFIGS_DIR / f"{key}.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def _output_dirs(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_dir()}


def run_script(script: Path) -> dict:
    rel = script.relative_to(EXAMPLES_DIR)
    print(f"━━━ {rel} ━━━", flush=True)
    outputs_root = REPO_ROOT / "outputs"
    before = _output_dirs(outputs_root) if outputs_root.exists() else set()
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
    )
    elapsed = round(time.monotonic() - t0, 2)
    after = _output_dirs(outputs_root) if outputs_root.exists() else set()
    new_dirs = sorted(str(p.relative_to(REPO_ROOT)) for p in after - before)
    status = "passed" if result.returncode == 0 else "failed"
    print(f"  → {status} (exit {result.returncode}, {elapsed}s)\n", flush=True)
    record: dict = {
        "script": str(rel),
        "status": status,
        "exit_code": result.returncode,
        "duration_s": elapsed,
    }
    if new_dirs:
        record["output_dirs"] = new_dirs
    config = _load_config(script)
    if config:
        record["config"] = config
    return record


def write_log(log_path: Path, timestamp: str, records: list[dict], total: int) -> None:
    passed = sum(1 for r in records if r["status"] == "passed")
    failed = sum(1 for r in records if r["status"] == "failed")
    pending = total - len(records)
    log = {
        "run_at": timestamp,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pending": pending,
        },
        "results": records,
    }
    with open(log_path, "w") as f:
        yaml.dump(log, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    scripts = sorted(
        p for p in EXAMPLES_DIR.rglob("*.py") if is_example(p)
    )

    if not scripts:
        print("No example scripts found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = OUTPUT_DIR / f"run_{timestamp}.yaml"

    write_log(log_path, timestamp, [], len(scripts))
    print(f"Log: {log_path.relative_to(REPO_ROOT)}\n")

    records: list[dict] = []
    for script in scripts:
        record = run_script(script)
        records.append(record)
        write_log(log_path, timestamp, records, len(scripts))

    passed = [r for r in records if r["status"] == "passed"]
    failed = [r for r in records if r["status"] == "failed"]

    print("━━━ Summary ━━━")
    print(f"  Passed : {len(passed)}")
    print(f"  Failed : {len(failed)}")
    if failed:
        for r in failed:
            print(f"    ✗ {r['script']}")

    print(f"\nLog written to: {log_path.relative_to(REPO_ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
