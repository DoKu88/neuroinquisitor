"""Tests for S3Backend — all S3 calls mocked with moto."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from moto import mock_aws  # noqa: E402

from neuroinquisitor.backends.s3 import S3Backend  # noqa: E402


BUCKET = "ni-test-bucket"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stable fake credentials so boto3 does not look at the host config."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture()
def s3_bucket():  # noqa: ANN201
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture()
def backend(s3_bucket, tmp_path: Path) -> S3Backend:
    b = S3Backend(bucket=BUCKET, tmp_dir=tmp_path / "ni-tmp")
    yield b
    if not b._closed:
        b.close()


def _get_body(client, key: str) -> bytes:
    return client.get_object(Bucket=BUCKET, Key=key)["Body"].read()


# ---------------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------------


def test_write_data_uploads_in_background(backend: S3Backend, s3_bucket) -> None:
    backend.write("epoch_0000.h5", b"hello")
    backend.flush()
    assert _get_body(s3_bucket, "epoch_0000.h5") == b"hello"


def test_write_from_path_uploads(
    backend: S3Backend, s3_bucket, tmp_path: Path
) -> None:
    src = tmp_path / "snap.h5"
    src.write_bytes(b"abc123")
    backend.write_from_path("epoch_0001.h5", src)
    backend.flush()
    assert _get_body(s3_bucket, "epoch_0001.h5") == b"abc123"


def test_index_json_writes_synchronously(backend: S3Backend, s3_bucket) -> None:
    """JSON keys must skip the background queue."""
    backend.write("index.json", b'{"x": 1}')
    # No flush needed — the object should already be present.
    assert _get_body(s3_bucket, "index.json") == b'{"x": 1}'
    # And it must not be on the upload queue.
    assert len(backend._pending) == 0


def test_exists_and_delete(backend: S3Backend, s3_bucket) -> None:
    backend.write("a.h5", b"x")
    backend.flush()
    assert backend.exists("a.h5") is True
    assert backend.exists("missing.h5") is False
    backend.delete("a.h5")
    assert backend.exists("a.h5") is False


def test_read_path_downloads(backend: S3Backend, s3_bucket, tmp_path: Path) -> None:
    s3_bucket.put_object(Bucket=BUCKET, Key="remote.h5", Body=b"remote-bytes")
    local = backend.read_path("remote.h5")
    assert local.exists()
    assert local.read_bytes() == b"remote-bytes"


def test_read_path_uses_local_cache(
    backend: S3Backend, s3_bucket, tmp_path: Path
) -> None:
    backend.write("cached.h5", b"cached")
    backend.flush()
    # After write the file is already in tmp; read_path returns the cached one.
    local = backend.read_path("cached.h5")
    assert local.read_bytes() == b"cached"


def test_read_path_missing_raises(backend: S3Backend) -> None:
    with pytest.raises(FileNotFoundError):
        backend.read_path("nope.h5")


# ---------------------------------------------------------------------------
# Drain / close
# ---------------------------------------------------------------------------


def test_flush_propagates_upload_exception(
    backend: S3Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self, src, key) -> None:  # noqa: ANN001
        raise RuntimeError("upload boom")

    monkeypatch.setattr(S3Backend, "_upload", boom)
    backend.write("bad.h5", b"x")
    with pytest.raises(RuntimeError, match="upload boom"):
        backend.flush()


def test_close_drains_pending(backend: S3Backend, s3_bucket) -> None:
    for i in range(5):
        backend.write(f"k_{i}.h5", b"x" * 32)
    backend.close()
    for i in range(5):
        assert _get_body(s3_bucket, f"k_{i}.h5") == b"x" * 32


def test_close_is_idempotent(backend: S3Backend) -> None:
    backend.close()
    backend.close()  # must not raise


# ---------------------------------------------------------------------------
# cleanup_after_upload
# ---------------------------------------------------------------------------


def test_cleanup_after_upload_removes_local_file(
    s3_bucket, tmp_path: Path
) -> None:
    b = S3Backend(
        bucket=BUCKET, tmp_dir=tmp_path / "ni-tmp2", cleanup_after_upload=True
    )
    src = tmp_path / "snap.h5"
    src.write_bytes(b"x" * 16)
    b.write_from_path("epoch_0000.h5", src)
    b.close()
    assert not src.exists()


# ---------------------------------------------------------------------------
# Prefix
# ---------------------------------------------------------------------------


def test_prefix_applied_to_keys(s3_bucket, tmp_path: Path) -> None:
    b = S3Backend(bucket=BUCKET, prefix="runs/r1", tmp_dir=tmp_path / "ni-tmp3")
    b.write("index.json", b"{}")
    b.write("epoch_0000.h5", b"abc")
    b.close()
    assert _get_body(s3_bucket, "runs/r1/index.json") == b"{}"
    assert _get_body(s3_bucket, "runs/r1/epoch_0000.h5") == b"abc"


# ---------------------------------------------------------------------------
# ImportError when boto3 absent — simulated by monkeypatching the module-level
# `boto3` reference to None.
# ---------------------------------------------------------------------------


def test_import_error_when_boto3_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import neuroinquisitor.backends.s3 as s3mod

    monkeypatch.setattr(s3mod, "boto3", None)
    with pytest.raises(ImportError, match="neuroinquisitor\\[s3\\]"):
        S3Backend(bucket=BUCKET, tmp_dir=tmp_path)
