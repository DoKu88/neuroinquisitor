"""S3Backend — asynchronous upload backend for cloud storage.

Snapshot files are written to a local tmp file and uploaded to S3 in a background
thread pool so the training loop is not blocked by network latency.  Index files
(``.json``) are uploaded synchronously to preserve catalog consistency.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

from .base import Backend

if TYPE_CHECKING:
    from collections.abc import Iterable


logger = logging.getLogger(__name__)


class S3Backend(Backend):
    """Stores snapshot files in an S3 bucket with asynchronous uploads.

    Snapshot data is written to a local tmp file then uploaded by a
    :class:`~concurrent.futures.ThreadPoolExecutor` worker so the calling
    training loop returns immediately.  Index files (any key ending in
    ``.json``) are uploaded synchronously because catalog consistency
    matters more than throughput.

    Credentials are resolved via the standard boto3 chain
    (``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``, ``AWS_PROFILE``,
    or an IAM role).  This class does not accept credentials directly.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        tmp_dir: str | os.PathLike[str] | None = None,
        max_workers: int = 4,
        cleanup_after_upload: bool = False,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._tmp = Path(tmp_dir) if tmp_dir is not None else Path(tempfile.mkdtemp())
        self._tmp.mkdir(parents=True, exist_ok=True)
        self._cleanup = cleanup_after_upload
        self._client = boto3.client("s3")

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: deque[Future[None]] = deque()
        self._lock = threading.Lock()
        self._failure: BaseException | None = None  # first background error
        self._closed = False

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def _local_path(self, key: str) -> Path:
        # mirror the key under the tmp dir so reads find a cached download
        return self._tmp / key

    # ------------------------------------------------------------------
    # Pending future bookkeeping
    # ------------------------------------------------------------------

    def _prune_completed(self) -> None:
        """Discard completed futures.  Stash the first failure for flush()."""
        with self._lock:
            while self._pending and self._pending[0].done():
                fut = self._pending.popleft()
                exc = fut.exception()
                if exc is not None and self._failure is None:
                    self._failure = exc

    def _upload(self, src: Path, key: str) -> None:
        self._client.upload_file(str(src), self._bucket, self._full_key(key))
        if self._cleanup:
            src.unlink(missing_ok=True)

    def _enqueue_upload(self, src: Path, key: str) -> None:
        future = self._executor.submit(self._upload, src, key)
        with self._lock:
            self._pending.append(future)
        self._prune_completed()

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def write(self, key: str, data: bytes) -> None:
        """Write *data* to *key*.

        ``.json`` keys upload synchronously.  Everything else is staged to a
        local tmp file then uploaded in a background thread.
        """
        if key.endswith(".json"):
            self._client.put_object(
                Bucket=self._bucket, Key=self._full_key(key), Body=data
            )
            return

        local = self._local_path(key)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        self._enqueue_upload(local, key)

    def write_from_path(self, key: str, src: str | os.PathLike[str]) -> None:
        """Enqueue upload of an already-written local file at *src*."""
        self._enqueue_upload(Path(src), key)

    def read_path(self, key: str) -> Path:
        local = self._local_path(key)
        if local.exists():
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self._bucket, self._full_key(key), str(local))
        except ClientError as exc:
            raise FileNotFoundError(
                f"Key {key!r} not found in s3://{self._bucket}/{self._full_key(key)}"
            ) from exc
        return local

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._full_key(key))
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=self._full_key(key))

    # ------------------------------------------------------------------
    # Draining
    # ------------------------------------------------------------------

    def _drain(self, futures: Iterable[Future[None]]) -> None:
        first_exc: BaseException | None = None
        for fut in list(futures):
            exc = fut.exception()
            if exc is not None and first_exc is None:
                first_exc = exc
        if first_exc is not None and self._failure is None:
            self._failure = first_exc

    def flush(self, timeout: float | None = None) -> None:  # noqa: ARG002
        """Block until all pending uploads complete.  Re-raises upload errors."""
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        self._drain(pending)
        if self._failure is not None:
            exc, self._failure = self._failure, None
            raise exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        return (
            f"S3Backend(bucket={self._bucket!r}, prefix={self._prefix!r}, "
            f"tmp_dir={str(self._tmp)!r})"
        )
