# Phase 3: LLM-Scale Finetuning & Cloud Storage

**Goal**: Scale NeuroInquisitor from small-to-medium research models to 7B-parameter LLMs running on cloud GPU infrastructure (Modal), enabling researchers to capture, store, and analyze how model internals shift during finetuning — across both LoRA/QLoRA and full finetuning regimes.

**Prerequisite**: Sprints 1–8 complete (core capture, storage, replay, analyzers, format contracts).

---

## Problem Statement

The current architecture was designed and validated against models with tens of millions of parameters (ResNet, small transformers). Scaling to 7B-parameter models reveals three blockers:

1. **BytesIO memory bottleneck**: `HDF5Format.write()` buffers the entire snapshot in RAM before writing. For a full Qwen 7B checkpoint in bfloat16, this requires ~14 GB of transient memory just for serialization.

2. **Synchronous blocking writes**: Every `snapshot()` call blocks the training loop until I/O completes. On Modal with S3 as the target, upload latency (~5–60s per snapshot) makes synchronous writes impractical.

3. **No remote storage backend**: Modal containers are ephemeral. Without an S3 backend, snapshot data is lost when the container exits.

**Additionally**, a latent bug in `core.py` causes a `TypeError` when calling `.numpy()` on bfloat16 tensors — which all modern LLMs use by default. This must be fixed before any LLM-scale work is possible.

---

## Sprints

| Sprint | Title | Focus |
|---|---|---|
| Sprint 10 | LLM-Scale Core Fixes | bfloat16 fix, layer_filter, streaming serialization |
| Sprint 11 | Cloud Storage & Modern Formats | S3Backend (async), SafetensorsFormat |
| Sprint 12 | Qwen 7B Finetuning Showcase | End-to-end example, interpretability analysis |

---

## Storage Size Reference

| Scenario | Raw bfloat16 | HDF5 gzip-4 | safetensors |
|---|---|---|---|
| LoRA adapters (r=16, typical) | ~140 MB | ~90 MB | ~140 MB |
| Selective layers (attn+MLP, 32L) | ~4 GB | ~3 GB | ~4 GB |
| Full Qwen 7B | ~14 GB | ~10 GB | ~14 GB |

**Format guidance:**
- LoRA/QLoRA → `format="hdf5"`, `compress=True` — small enough for current path, compression reduces storage ~35%.
- Full finetuning, selective layers → `format="safetensors"` — direct-to-file, memory-mappable reads, no in-RAM buffer.
- Full finetuning, all params → `format="safetensors"` + `layer_filter` — essential to skip frozen embeddings and `lm_head` (~30% of params, slowest-moving).

---

## Architecture Changes (Phase 3)

```
Before (Sprints 1–8):
  snapshot() → BytesIO → backend.write(bytes) → local disk

After (Sprint 10–11):
  snapshot() → format.write_to_path(tmp_file) → backend.write_from_path(tmp) → [enqueue]
                                                                                  ↓
                                                                    background thread → S3
```

All changes are backward compatible: `LocalBackend` continues to use the existing bytes path with no modification.

---

## Interpretability Research Targets

With Phase 3 in place, the intended research workflow is:

1. **Finetune Qwen 7B** on a target task (instruction following, coding, domain adaptation) with `NeuroInquisitor` capturing selective layer snapshots to S3 every N steps.
2. **Trajectory analysis** (`trajectory_stats`): track L2 drift of attention Q/K/V matrices across training — identify which layers move most, which stabilize early.
3. **Rank dynamics** (`spectrum_rank`): detect rank collapse or rank expansion in weight matrices — known to correlate with overfitting and generalization.
4. **Representational similarity** (`similarity_compare` / CKA): compare activations between base model and finetuned checkpoints to quantify how much each layer's representations shift.
5. **Linear probing** (`probe_linear`): measure whether task-relevant features emerge in earlier or later layers over the course of finetuning.

---

## Definition of Done (Phase 3)

- `NeuroInquisitor` successfully captures snapshots from a bfloat16 LLM without error.
- `layer_filter` reduces captured parameters to only the layers of interest, cutting snapshot size by 50–70% for full finetuning.
- S3Backend uploads snapshots in the background without measurably blocking the training loop.
- `SafetensorsFormat.write_to_path()` writes a 14 GB snapshot without requiring 14 GB of transient RAM.
- A complete LoRA finetuning example runs end-to-end on Modal, persisting data to S3 and producing at least one interpretability plot.
