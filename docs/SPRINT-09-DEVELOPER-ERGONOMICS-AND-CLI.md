# Sprint 9: Developer Ergonomics, Optional Backends & CLI

**Goal**: Make every artifact NeuroInquisitor captures accessible in formats a developer already knows, add optional storage backends, expose a CLI, and write the developer quick-start guide.

**Prerequisite**: Sprint 8 must be complete. The CLI and quick-start guide assume all built-in analyzers exist and the standard format contracts from Sprint 7 are in place.

## Tasks

- [ ] `NI-DELTA-004` Add optional tensor storage backends.
  - Add `safetensors` as an optional extra (`pip install neuroinquisitor[safetensors]`).
    - Safetensors is a portable, safe format widely used for sharing checkpoints. Good default for exporting a single epoch's weights.
  - Add `zarr` as an optional extra (`pip install neuroinquisitor[zarr]`).
    - Zarr is better than HDF5 for large array stacks and object-store-backed runs.
  - Both backends must produce files readable by their respective libraries without NI installed.
  - Acceptance:
    - `SnapshotCollection.export(epoch=5, format="safetensors", path="epoch5.safetensors")` works when the extra is installed.
    - Clear `ImportError` with install instructions when the extra is missing.
    - Docs explain when to use HDF5 vs Safetensors vs Zarr.

- [ ] `NI-DELTA-005` Add CLI surface.
  - Commands:
    - `ni runs list` — list runs and their epoch ranges.
    - `ni manifest show <run>` — print manifest metadata.
    - `ni snapshot export <run> --epoch N --format [numpy|safetensors|hdf5]` — export a snapshot to a file a developer can open without NI.
    - `ni analyze run <run> --analyzer <name>` — run a built-in analyzer.
    - `ni plugins list` — list registered analyzers with name, version, and required inputs.
  - CLI must call the same public Python APIs; no internal shortcuts.
  - Acceptance:
    - `ni snapshot export` produces a file openable by `np.load` or `safetensors.torch.load_file` depending on format.
    - All commands have `--help` output.

- [ ] `NI-DELTA-006` Write developer quick-start guide.
  - This is not a tutorial about NI internals. It answers the question: *"I have captured data — where do I start?"*
  - Cover five concrete starting points, each as a short runnable example:
    1. **Weights over time**: load epochs 0–20 as numpy arrays, compute something (e.g. L2 norm per layer), plot it. No NI API after the export call.
    2. **Reload a checkpoint**: `model.load_state_dict(col.to_state_dict(epoch=10))`, run a forward pass, do whatever you want with the output.
    3. **Activations from replay**: run `ReplaySession`, get a `Dict[str, Tensor]`, pass directly to a linear classifier or to any tool that accepts hook-style dicts.
    4. **Derived table as a starting point**: open a Parquet file with pandas, filter/plot/export — no NI knowledge required.
    5. **Upload to Neuronpedia**: export activations and upload to Neuronpedia for interactive feature visualization (see NI-DELTA-008).
  - Each example should be self-contained and under 20 lines.
  - Acceptance:
    - All five examples run without error against a toy model (Neuronpedia example skipped if API key absent).
    - A developer reading only this guide (not the full docs) can start an analysis in under 10 minutes.

- [ ] `NI-DELTA-008` Add Neuronpedia integration example.
  - Neuronpedia is a platform for interactive neural network feature visualization, primarily oriented around sparse autoencoder (SAE) features. NI's role is to supply the activations; Neuronpedia does the visualization.
  - Write a short runnable example (under 30 lines) that:
    1. Runs `ReplaySession` to extract activations for a set of inputs.
    2. Exports activations using `ReplaySession.to_numpy()`.
    3. Uploads the activation arrays to Neuronpedia via their public API.
    4. Returns the Neuronpedia dashboard URL for the uploaded features.
  - Do not wrap or abstract the Neuronpedia API — call it directly in the example. The example is documentation, not a library dependency.
  - The example must degrade gracefully: if `NEURONPEDIA_API_KEY` is not set, print a clear message and exit without error.
  - Acceptance:
    - Example runs end-to-end when a valid API key is available.
    - No Neuronpedia SDK or wrapper is added as a package dependency.
    - Example is included in the developer quick-start guide as on-ramp 5.

## Testing

- Backend feature-gate tests: missing optional extras raise `ImportError` with the correct install hint.
- Round-trip backend tests: `safetensors` and `zarr` exports are loadable by their respective libraries without NI installed.
- CLI tests: each command produces correct output and calls the public Python API (not internal shortcuts).
- Quick-start example tests: all five examples in the guide run end-to-end against a toy model (Neuronpedia example skipped when `NEURONPEDIA_API_KEY` is absent).
- Neuronpedia example test: activation arrays exported via `to_numpy()` match the shape and dtype expected by the Neuronpedia upload endpoint.

## Definition of Done

- Optional backends degrade gracefully with actionable install instructions.
- CLI exposes inspection and export without requiring a Python environment.
- Developer quick-start guide gives five concrete, runnable on-ramps into analysis including Neuronpedia upload.
- Every public return value that carries model data is a standard type: `Dict[str, Tensor]`, `Dict[str, np.ndarray]`, `pd.DataFrame`, or a Python scalar. No NI wrapper types in outputs.
- A checkpoint at any epoch is loadable into a stock PyTorch model in one line.
- Neuronpedia upload example runs end-to-end and degrades gracefully without an API key.
