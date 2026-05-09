# 001 — Snapshot tensor conversion path (`detach().cpu().numpy()`)

**Status**: `open`
**Area**: `src/neuroinquisitor/core.py` — `NeuroInquisitor.snapshot()`
**First flagged**: 2026-05-08

## What is the issue?

Inside `snapshot()`, every captured tensor is passed through
`detach().cpu().numpy()` before being handed to the active `Format` for
serialization:

```206:218:src/neuroinquisitor/core.py
        params: dict[str, np.ndarray] = {
            name: param.detach().cpu().numpy()
            for name, param in self._model.named_parameters()
        }

        buffers: dict[str, np.ndarray] | None = None
        if self._capture_policy.capture_buffers:
            raw_buffers = {
                name: buf.detach().cpu().numpy()
                for name, buf in self._model.named_buffers()
                if buf is not None
            }
            buffers = raw_buffers if raw_buffers else None
```

This is fine for the current scope (CIFAR‑style examples, small models,
infrequent snapshots, fp32 on CPU or a single GPU), but it bakes in
three assumptions that won't hold for realistic users:

### 1. Numpy supports the model's dtype

`tensor.numpy()` raises `TypeError: Got unsupported ScalarType ...` for
any dtype numpy can't represent. The big one is **`torch.bfloat16`**,
which is the default for most modern transformer / LLM training. Also
affected: `torch.float8_*`, complex32, and a few quantized dtypes.

> Anyone running `NeuroInquisitor(model).snapshot(...)` on a bf16
> model will hit a hard error on the very first snapshot.

### 2. The training thread can afford a synchronous D2H copy

`.cpu()` is a no-op for CPU tensors, but for GPU training it issues a
**synchronous device→host copy** of every captured tensor. The full
`snapshot()` call — copy, HDF5 serialize, backend write — runs inline
on the training thread. For occasional snapshots that's invisible; for
sub‑epoch snapshot frequencies on a real model it becomes the
dominating cost in the loop.

### 3. The CPU tensor is safe to alias

On CPU‑only training, `param.detach().cpu().numpy()` returns a *view*
into live training memory (because `.cpu()` is a no-op and `.numpy()`
is a zero-copy view). With a single‑threaded loop calling `snapshot()`
at a deterministic point, this is fine; the array is fully consumed by
`Format.write` before the loop continues. The moment any part of the
pipeline runs off-thread, this becomes a torn-write hazard.

## Why does it matter?

| Scenario | Impact |
|---|---|
| User trains an LLM in bf16 | Hard crash on first `snapshot()` |
| User trains a large model on GPU and snapshots every N steps | Visible step-time regression; D2H + HDF5 serialize on the hot path |
| Future async writer thread | Silent torn-write on CPU tensors unless we add a `clone()` |
| Format that natively wants `torch.Tensor` (e.g. `safetensors`) | Forced to round-trip through numpy, losing dtype fidelity (bf16 → upcast or reinterpret) |

## What are the alternatives?

### A. Status quo — `detach().cpu().numpy()`
Simple, correct, dtype‑honest, matches the current
`Format.write(params: dict[str, np.ndarray], ...)` signature. Best
default for the current scope.

**Cost**: assumptions 1–3 above.

### B. Add a `.clone()` after `.cpu()`
`buf.detach().cpu().clone().numpy()` (equivalently
`buf.detach().to("cpu", copy=True).numpy()`). Guarantees an independent
host buffer regardless of source device. Buys safety for any future
async path without changing the public API.

**Cost**: an extra host‑side allocation when training on CPU
(roughly 1× model size of additional RAM, transient).

### C. Pinned host staging + `non_blocking=True`
Pre-allocate pinned host tensors matching each parameter once at
attach time. On snapshot, do `param.detach().to(staging, non_blocking=True)`
on a side CUDA stream. Then `numpy()`-view the staging buffers and
hand them to `Format.write` after the stream event completes.

**Cost**: ~1× model size of pinned RAM (2× if double-buffered to
overlap copy with serialization). Foreground stall drops to "issue a
non-blocking D2H + record an event".

### D. Background writer thread / queue
`snapshot()` enqueues `(file_key, params_dict, metadata)`; a worker
thread does the HDF5 serialize and `backend.write`. Combined with C,
the foreground cost collapses to a copy kickoff and a queue append.

**Cost**: bounded queue + backpressure policy (drop / block / warn),
`close()` must drain the queue, and any CPU‑resident tensor handed to
the queue must be cloned (see B) to avoid torn writes.

### E. Skip the numpy detour: torch‑native format
Add a sibling to `HDF5Format`, e.g. `SafeTensorsFormat` or
`TorchFormat`, that consumes `dict[str, torch.Tensor]` directly.
`safetensors` is fast, mmap-friendly, language-portable, and natively
handles bf16 / fp8.

**Cost**: `Format.write` must be generalized over tensor type — either
a new method (`write_tensors`) or a small `Format.wants_tensors`
capability flag that the orchestrator inspects before deciding whether
to call `.cpu().numpy()`. Biggest payoff for modern training; biggest
design change.

### F. GPUDirect / `kvikio` (write straight from device)
Bypass host memory entirely. Realistic only for multi-GB checkpoints
at high frequency.

**Cost**: hard dependency, hard to test, niche audience. Not
recommended for the current project scope.

### G. Lossy: quantize / delta-encode
Orthogonal to the conversion question — about disk savings, not about
the cost of getting tensors out of the model. Only worth considering
when disk / IO is the actual bottleneck.

## Recommendation

In priority order:

1. **Harden the current path against unsupported dtypes.**
   Add an explicit check (or a controlled upcast) before calling
   `.numpy()`, with a clear error message pointing the user at the
   future torch‑native format. This is the highest value / lowest
   cost change and removes the only outright crash mode.

2. **Plan for a torch‑native format (E).**
   The `Format` abstraction is most of the way there; the only
   awkwardness is the numpy-typed signature. Decide on either
   `Format.wants_tensors: bool` or a parallel `write_tensors`
   method, and add `SafeTensorsFormat` once that's settled. This
   gives bf16 users a real path without compromising HDF5's
   portability story.

3. **Defer C and D (pinned staging + background writer)** until
   there is a concrete user training a real model at a snapshot
   frequency that justifies the complexity. When we do add it,
   gate it behind an explicit `CapturePolicy.async_writes` (or
   similar) flag, and pair it with B (`clone()` for CPU tensors)
   to close the aliasing hazard.

## Open questions

- Should the dtype check live in `core.py` (orchestrator) or in
  `HDF5Format.write` (format-specific)? Leaning toward the format,
  since whether a dtype is supported is a property of the format,
  not the orchestrator.
- For an eventual `SafeTensorsFormat`, do we accept the file extension
  divergence (`.safetensors` vs `.h5`) as a per-snapshot file split,
  or do we keep both formats writing single self-contained files
  per snapshot? (Current design assumes the latter.)
- If we add async writes, is `close()` allowed to block arbitrarily
  long while the queue drains, or do we surface a progress hook?
