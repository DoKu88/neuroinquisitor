# Sprint 7-01: Examples Consolidation

**Goal**: Reduce the examples directory from 8 top-level demos + 2 redundant Captum files to exactly 5 cohesive demos that cover all three major architecture families (FC, CNN, Transformer), demonstrate NeuroInquisitor's temporal advantage clearly, and include one integration example per major third-party interpretability library (Captum, TorchLens, TransformerLens).

## Background

The current examples set has significant redundancy: `basic_usage.py`, `mnist_example.py`, and `cifar10_example.py` all demonstrate identical NI API usage on near-identical training loops. Two Captum examples exist when one is sufficient. No single example demonstrates NeuroInquisitor's architecture-agnostic API across FC, CNN, and Transformer models simultaneously — the most compelling sales pitch for the package.

The `specific_actions/` how-to files (8 files) are kept as-is. This sprint covers only the top-level demos and integration examples.

## Files to Delete

| File | Reason |
|---|---|
| `examples/basic_usage.py` | Superseded by `multi_arch_showcase.py` |
| `examples/basic_usage_utils.py` | Same |
| `examples/mnist_example.py` | Redundant with CIFAR-10; no additional NI concepts demonstrated |
| `examples/mnist_example_utils.py` | Same |
| `examples/cifar10_example.py` | NI usage identical to basic_usage; CNN coverage moves to TorchLens integration |
| `examples/cifar10_example_utils.py` | Same |
| `examples/captum_use_examples/cifar10_captum.py` | Strictly weaker than `grokking_captum.py` — same two integration paths, less interesting science |
| `examples/captum_use_examples/cifar10_captum_utils.py` | Same |

## Files to Keep (unchanged or minor edits)

| File | Status | Notes |
|---|---|---|
| `examples/grokking_example.py` + utils | Keep, minor edit | Remove the "every implemented capability" framing from the docstring — that role moves to `multi_arch_showcase.py`. Retain step-based snapshotting and the scientific framing. |
| `examples/captum_use_examples/grokking_captum.py` + utils | Keep, minor edit | Add a short "what NI adds to Captum" paragraph at the top of the docstring: without NI, Captum runs once on the final model; with NI, attribution evolves across every checkpoint. Reposition as the canonical Captum integration rather than a grokking-specific demo. |
| `examples/torchlens_use_examples/torchlens_cifar10.py` + utils | Keep, no changes | The "film strip instead of a single photo" framing is the best articulation of NI's value proposition in the codebase. Do not dilute it. All TorchLens calls already live in the main file; utils is matplotlib only. |
| `examples/transformerlens_use_examples/cifar10_transformerlens.py` + utils | Keep, no changes | Strongest integration example in the repo. Five distinct use cases, all temporal. All TransformerLens calls already live in the main file; utils is matplotlib only. |
| `examples/specific_actions/` (all 8 files) | Keep, no changes | How-to reference files, not demos. |
| `examples/run_all.py` | Keep, update | Remove deleted examples; add `multi_arch_showcase.py`. |

## New File to Create

### `examples/multi_arch_showcase.py` (+ `examples/multi_arch_showcase_utils.py`)

Train a **fully connected network**, a **small CNN**, and a **small Transformer** on the **same synthetic classification task** — no dataset downloads required. All three are instrumented with identical `NeuroInquisitor` setup. The analysis section runs the same `SnapshotCollection` and `ReplaySession` calls on each, proving the API is architecture-agnostic.

**Task**: sequence classification on synthetic data. Each input is a short integer sequence; the label is a binary function of the sequence. This is trivially solvable by the Transformer, solvable by the MLP (with enough width), and borderline for the CNN (without spatial structure). The asymmetry makes the comparison interesting and gives the weight/activation evolution plots something to show.

**NI features demonstrated** (same checklist as the old `basic_usage.py` but across all three models):
- `CapturePolicy` — parameters, buffers, optimizer state
- `RunMetadata` — training provenance
- `NeuroInquisitor.snapshot()` — per-epoch checkpoints with metadata
- `SnapshotCollection` — `by_epoch`, `by_layer`, `select`, `to_state_dict`, `to_numpy`
- `ReplaySession` — activations, gradients, logits via forward/backward hooks
- Visualization — side-by-side weight evolution and activation drift per architecture

**Companion utils file** handles all matplotlib/visualization code so the main file stays readable.

**Requirements**:
- Zero external dataset downloads. All data generated with `torch.randn` / `torch.randint`.
- Runs in under 2 minutes on CPU.
- Each architecture is defined in its own clearly separated section of the file.
- The three `NeuroInquisitor` setups are identical — any difference between them belongs to the model, not the observer.
- Output directory: `outputs/multi_arch_showcase/<run-name>/`.

## Updated Examples Directory Structure (after sprint)

```
examples/
    multi_arch_showcase.py          ← NEW: FC + CNN + Transformer, synthetic data
    multi_arch_showcase_utils.py    ← NEW: visualization helpers
    grokking_example.py             ← KEEP: step-based snapshots, phase transition
    grokking_example_utils.py       ← KEEP
    run_all.py                      ← UPDATE: remove deleted entries, add new demo
    configs/                        ← UPDATE: add config for multi_arch_showcase
    specific_actions/               ← KEEP unchanged (8 how-to files)
    captum_use_examples/
        grokking_captum.py          ← KEEP: canonical Captum integration
        grokking_captum_utils.py    ← KEEP
    torchlens_use_examples/
        torchlens_cifar10.py        ← KEEP
        torchlens_cifar10_utils.py  ← KEEP
    transformerlens_use_examples/
        cifar10_transformerlens.py  ← KEEP
        cifar10_transformerlens_utils.py ← KEEP
```

## Integration example convention

For all three integration examples (`grokking_captum.py`, `torchlens_cifar10.py`, `cifar10_transformerlens.py`), the calls to the third-party library **must live in the main `.py` file**, not in the companion `*_utils.py` file. A developer reading the main file should be able to see exactly how Captum / TorchLens / TransformerLens is invoked alongside NeuroInquisitor — that is the point of the example. The utils file is for matplotlib/visualization code only.

Concretely:
- `grokking_captum.py`: `LayerIntegratedGradients`, `LayerConductance`, and all `.attribute()` calls stay in the main file.
- `torchlens_cifar10.py`: `tl.log_forward_pass`, `model_log.log_backward`, `tlv.show_model_graph`, `model_log.to_pandas()`, and all per-layer stat extraction stay in the main file.
- `cifar10_transformerlens.py`: `model.run_with_cache`, `model.run_with_hooks`, all hook-filter lambdas, and weight SVD computation stay in the main file.

This is already the case for `cifar10_transformerlens.py` and `torchlens_cifar10.py`. Verify `grokking_captum.py` conforms — all Captum calls are currently split between the main file and utils; consolidate any that leaked into utils back into the main file.

## Tasks

- [ ] `NI-EX-001` Delete the 8 files listed in "Files to Delete" above.

- [ ] `NI-EX-002` Write `examples/multi_arch_showcase.py`.
  - Define three model classes: `TinyMLP` (FC), `SmallCNN` (1D conv over patches), `TinyTransformer` (1-layer encoder).
  - All three trained on the same synthetic sequence classification task.
  - Each instrumented with identical `NeuroInquisitor` setup (same `CapturePolicy`, same `RunMetadata` fields, same `snapshot()` call signature).
  - Analysis section: `SnapshotCollection` and `ReplaySession` called identically on all three. Print a summary table comparing activation magnitudes and weight norms per architecture.
  - Call `generate_visualizations()` from utils for all plots.

- [ ] `NI-EX-003` Write `examples/multi_arch_showcase_utils.py`.
  - Side-by-side weight evolution plots (one column per architecture, one row per layer type).
  - Activation drift over epochs (mean activation magnitude per layer, per architecture).
  - Loss curves for all three architectures on one plot.

- [ ] `NI-EX-004` Write `examples/configs/multi_arch_showcase.yaml`.
  - Fields: `num_epochs`, `lr`, `n_samples`, `seq_len`, `n_classes`, `train_frac`, `batch_size`, `replay_modules` per architecture.

- [ ] `NI-EX-005` Edit `examples/grokking_example.py` docstring.
  - Remove "Demonstrates every implemented capability" bullet list — that belongs to `multi_arch_showcase.py` now.
  - Replace with a focused paragraph on what makes grokking a unique use case for NeuroInquisitor: step-based snapshotting, watching the phase transition in weight space, Fourier structure emerging in token embeddings.

- [ ] `NI-EX-006` Edit `examples/captum_use_examples/grokking_captum.py`.
  - Add a "What NeuroInquisitor adds to Captum" paragraph at the top of the docstring: explain that without NI, Captum runs once on the final model; with NI's checkpoint infrastructure, attribution becomes a function of training time.
  - Remove any framing that positions it as a grokking-specific demo — it should read as the canonical Captum integration that happens to use a transformer.
  - Audit the split between main file and utils: all `LayerIntegratedGradients`, `LayerConductance`, and `.attribute()` calls must be in the main file. Move any that leaked into `grokking_captum_utils.py` back to the main file. Utils is for matplotlib only.

- [ ] `NI-EX-007` Update `examples/run_all.py`.
  - Remove references to `basic_usage.py`, `mnist_example.py`, `cifar10_example.py`.
  - Add `multi_arch_showcase.py`.
  - Verify the execution order makes sense (showcase first, grokking second, integrations last).

## Testing

There are no automated test scripts for this sprint. Validation is manual:

1. **`multi_arch_showcase.py`**: run end-to-end and verify (a) all three architectures complete training, (b) `SnapshotCollection` and `ReplaySession` succeed for each, (c) output plots are written to `outputs/multi_arch_showcase/<run-name>/`, (d) total wall time is under 2 minutes on CPU.

2. **`grokking_example.py`**: confirm it still runs correctly after the docstring edit.

3. **`grokking_captum.py`**: confirm it still runs correctly after the docstring edit.

4. **`run_all.py`**: run it and confirm all 5 demos execute without error.

5. **Deletion check**: confirm none of the deleted files are imported or referenced anywhere in `examples/` or `src/`.

## Definition of Done

- Exactly 5 runnable demo files exist outside `specific_actions/`: `multi_arch_showcase.py`, `grokking_example.py`, `captum_use_examples/grokking_captum.py`, `torchlens_use_examples/torchlens_cifar10.py`, `transformerlens_use_examples/cifar10_transformerlens.py`.
- All 8 deleted files are gone.
- `multi_arch_showcase.py` runs on CPU in under 2 minutes with no dataset downloads.
- The three architectures in `multi_arch_showcase.py` use identical NI instrumentation — any asymmetry is in the model, not the observer.
- `run_all.py` executes all 5 demos without error.
- No file outside `specific_actions/` references any of the deleted examples.
