"""S3 Backend — end-to-end integration example.

What this demonstrates
----------------------
* `S3Backend` uploading snapshots **asynchronously** to a real S3 bucket
  while a tiny MLP is being trained.
* `SafetensorsFormat` writing weights with bfloat16-native support (works the
  same with `format="hdf5"` if you don't want safetensors).
* `NeuroInquisitor.close()` draining the background upload queue so no
  snapshot is lost when the process exits — the critical guarantee for
  ephemeral environments like Modal containers.
* Round-trip read of the snapshots from S3 via `NeuroInquisitor.load()` and
  bit-for-bit comparison against the in-memory weights captured during
  training.

How to run
----------
1. Edit ``examples/configs/s3_backend_example.yaml`` and set ``s3.bucket``.
2. Export AWS credentials (any of):
     export AWS_ACCESS_KEY_ID=...
     export AWS_SECRET_ACCESS_KEY=...
     # or:  export AWS_PROFILE=my-profile
3. Install extras (one-time):
     pip install "neuroinquisitor[s3,safetensors]"
4. Run:
     python examples/s3_backend_example.py

The example writes nothing to local disk except a small ``tmp/`` staging
directory.  All snapshot data lives in S3 under ``s3://<bucket>/<prefix>/<run>/``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import boto3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from neuroinquisitor import NeuroInquisitor, RunMetadata
from neuroinquisitor.backends.s3 import S3Backend
from neuroinquisitor.formats.safetensors_format import SafetensorsFormat  # noqa: F401

load_dotenv(dotenv_path=Path(__file__).parent / "configs" / ".env")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_cfg_path = Path(__file__).parent / "configs" / "s3_backend_example.yaml"
_cfg = yaml.safe_load(_cfg_path.read_text())

_S3 = _cfg["s3"]
BUCKET = _S3["bucket"]
PREFIX = _S3.get("prefix", "")
REGION = _S3.get("region", "us-east-1")
CLEANUP = bool(_S3.get("cleanup_after_upload", True))

FORMAT = _cfg.get("format", "hdf5")

N_SAMPLES = _cfg["n_samples"]
N_FEATURES = _cfg["n_features"]
N_CLASSES = _cfg["n_classes"]
HIDDEN = _cfg["hidden"]
NUM_EPOCHS = _cfg["num_epochs"]
BATCH_SIZE = _cfg["batch_size"]
LR = _cfg["lr"]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _preflight() -> None:
    if not BUCKET:
        print(
            f"\n[ERROR] s3.bucket is empty in {_cfg_path}.\n"
            "Edit the config file and set the bucket name before running.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    if not (
        os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
    ) and not os.getenv("AWS_PROFILE"):
        print(
            "[WARN] No AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_PROFILE in the env.\n"
            "       boto3 will still try an IAM role; if that fails the upload will error.\n"
        )

    # Sanity: does the bucket exist? (head_bucket is the cheapest probe.)
    client = boto3.client("s3", region_name=REGION)
    try:
        client.head_bucket(Bucket=BUCKET)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "?")
        print(
            f"[ERROR] head_bucket({BUCKET!r}) failed with code {code}.\n"
            "        Check the bucket name, region, and credentials, then re-run.\n",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------


def make_dataset(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, size=N_SAMPLES)
    return torch.from_numpy(x), torch.from_numpy(y).long()


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Train + snapshot
# ---------------------------------------------------------------------------


def run() -> None:
    _preflight()

    run_name = f"s3-example-{int(time.time())}"
    run_prefix = f"{PREFIX}/{run_name}".strip("/")
    print(f"Run name : {run_name}")
    print(f"Bucket   : s3://{BUCKET}/{run_prefix}/")
    print(f"Format   : {FORMAT}")
    print()

    tmp_dir = Path(tempfile.mkdtemp(prefix="ni-s3-example-"))
    backend = S3Backend(
        bucket=BUCKET,
        prefix=run_prefix,
        tmp_dir=tmp_dir,
        cleanup_after_upload=CLEANUP,
    )

    torch.manual_seed(0)
    model = TinyMLP()
    x, y = make_dataset()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    observer = NeuroInquisitor(
        model,
        log_dir=tmp_dir,   # only used as a label; the backend is the real sink
        backend=backend,
        format=FORMAT,
        create_new=True,
        run_metadata=RunMetadata(model_class="TinyMLP"),
    )

    # Capture the in-memory weights at every snapshot so we can verify the
    # S3 round-trip is bit-for-bit correct at the end.
    expected: dict[int, dict[str, np.ndarray]] = {}

    print("Training …")
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        perm = torch.randperm(N_SAMPLES)
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            full_loss = loss_fn(model(x), y).item()

        # Snapshot — returns immediately because the upload is async.
        observer.snapshot(epoch=epoch, metadata={"loss": full_loss})
        expected[epoch] = {
            name: p.detach().cpu().to(torch.float32).numpy().copy()
            for name, p in model.named_parameters()
        }
        print(f"  epoch {epoch}  loss={full_loss:.4f}   (snapshot enqueued)")

    t_train = time.time() - t0
    print(f"\nTraining done in {t_train:.2f}s — closing observer (drains S3 uploads)…")

    t0 = time.time()
    observer.close()
    t_drain = time.time() - t0
    print(f"Drain complete in {t_drain:.2f}s. All snapshots are in S3.\n")

    # -----------------------------------------------------------------
    # Round-trip verification — load everything back through a new
    # S3Backend instance, prove bytes survived the trip.
    # -----------------------------------------------------------------
    print("Verifying round-trip from S3 …")
    verify_tmp = Path(tempfile.mkdtemp(prefix="ni-s3-verify-"))
    verify_backend = S3Backend(bucket=BUCKET, prefix=run_prefix, tmp_dir=verify_tmp)
    col = NeuroInquisitor.load(tmp_dir, backend=verify_backend, format=FORMAT)
    assert sorted(col.epochs) == list(range(NUM_EPOCHS)), col.epochs
    for epoch, exp in expected.items():
        loaded = col.by_epoch(epoch)
        for name, arr in exp.items():
            np.testing.assert_allclose(loaded[name], arr, rtol=1e-5, atol=1e-6)
        print(f"  epoch {epoch}: {len(loaded)} tensors verified ✓")
    verify_backend.close()

    print(
        f"\nDone. {NUM_EPOCHS} snapshots written to s3://{BUCKET}/{run_prefix}/\n"
        "Tip: aws s3 ls s3://{}/{}/".format(BUCKET, run_prefix)
    )


if __name__ == "__main__":
    run()
