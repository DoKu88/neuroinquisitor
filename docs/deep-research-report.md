# NeuroInquisitor Deep Research Brief

## Where NeuroInquisitor fits

Your repository already has the bones of a strong research substrate. Today, `neuroinquisitor` snapshots all model parameters from `model.named_parameters()` into one HDF5 file per snapshot, maintains an `index.json`, supports lazy load/filter operations by epoch and layer, uses parallel reads for per-layer history, and keeps backends/formats abstract so users can swap storage later. The package is intentionally lean at the core, depending only on PyTorch, NumPy, and h5py, while the examples already cover MNIST, CIFAR-10, and a grokking-style transformer run. That combination makes the project unusually well-positioned as an **open trajectory-observability layer** for neural networks rather than “just another visualization script.” citeturn25view3turn6view2turn7view0turn4view0turn2view0

The broader field is fragmented. Captum is the most mature PyTorch library for attribution, concept methods, and influence methods; TorchLens targets exhaustive extraction of hidden activations and computation graphs for arbitrary PyTorch models; TransformerLens focuses on activation access, editing, and patching for transformers; and Neuronpedia shows that there is now real demand for browser-first, open-source interpretability platforms with activations, UMAP, probes, SAEs, dashboards, exports, and even AI-assisted interpretation. Interactive embedding views are also now standard expectations in tools like TensorBoard’s Projector and FiftyOne’s Embeddings panel. Meanwhile, recent reviews emphasize that interpretability is still an active, diverse field and that mechanistic interpretability in particular still faces scaling and automation challenges. citeturn18search3turn20search11turn20search2turn20search4turn26view3turn15search2turn19search0turn19search1

That context leads to the central product recommendation: **do not try to become a closed, all-in-one replacement for Captum, TransformerLens, TensorBoard, or Neuronpedia.** The better product thesis is to make NeuroInquisitor the open, trajectory-first data plane for model internals across training, with a plugin system and a web GUI layered on top. In other words: raw artifacts and replay at the bottom, mathematical analyzers in the middle, and dashboards or an AI copilot at the top. That position is differentiated, aligned with your current repo design, and much more likely to stay useful as the field changes. citeturn25view3turn18search3turn20search2turn20search4turn19search0

## Mathematical techniques to include

Two filters matter more than anything else for your roadmap. The first is whether a method can run on **weights alone**; the second is whether it needs **data-conditioned replay of activations and/or gradients**. Your current repo handles the first category well already, but most of the highest-value modern methods sit in the second category, which is why I would treat **activation replay and cached derived artifacts** as the next foundational capability. citeturn6view2turn25view3turn10search5turn8search7turn9search7turn11search1turn14search0

| Phase | Technique | What it answers | Data needed | Why it belongs in NeuroInquisitor | Representative use |
|---|---|---|---|---|---|
| Now | **Trajectory geometry and interpolation** | How far weights move, when layers stabilize, whether checkpoints lie on low-loss connecting paths, and whether training dynamics show phase changes | Weights; optional validation loss for interpolation | This is the most natural first-class capability for a checkpoint-first package and directly exploits your epoch/step indexing | Mode connectivity and low-loss paths were studied by Garipov et al. and Draxler et al., making parameter-space trajectory analysis a well-established research direction. citeturn12search4turn12search5 |
| Now | **Spectral analysis and rank metrics** | Are layers becoming lower-rank, more compressible, or more structured over training; how singular values and stable/effective rank evolve | Weights; optionally activations | Rank/spectrum metrics are cheap, general, and excellent for “observability over time” dashboards | SVCCA explicitly used SVD to study intrinsic dimensionality and training dynamics; later work studied rank deficiency and random-matrix structure in trained networks. citeturn23search5turn27search0turn27search2turn27search4 |
| Now | **PCA and UMAP over tensors, channels, and checkpoints** | What low-dimensional structure the weights or features have; which epochs or filters cluster together; which checkpoints are outliers | Weights now; activations later | Users already expect this type of exploratory view, and it is perfect for a layer/epoch GUI | UMAP is a standard dimensionality-reduction method; TensorBoard Projector and FiftyOne both use interactive low-dimensional embeddings for model analysis. citeturn23search0turn26view3turn15search2turn23search3 |
| Now | **SVCCA, PWCCA, and CKA** | When a layer has “converged,” how similar two checkpoints are, whether two models learn similar representations, and how layers align | Activations on a fixed evaluation slice; can approximate with weights for some matrices | This is one of the most useful “math-first” representation modules for cross-epoch comparison | SVCCA was introduced for deep learning dynamics and interpretability; PWCCA extended CCA-based representational comparison; CKA became a robust standard for comparing learned representations. citeturn23search5turn10search1turn8search7 |
| Next | **Decoding hidden states with linear probes and tuned lenses** | What information is linearly recoverable from each layer at each epoch; for LLMs, what token predictions are latent at intermediate layers | Activations, labels, and for LLMs vocabulary logits | This gives users a very interpretable “what lives in this layer?” workflow and is ideal for selected layers/epochs | Linear probes were proposed to understand intermediate layers; tuned lens probes decode how transformer predictions evolve layer by layer. citeturn10search5turn10search3 |
| Next | **Concept methods with TCAV** | Whether a human-defined concept matters for a class or output, and how that changes across epochs/layers | Activations, gradients, and user-defined concept sets | This is a high-value way to keep the package accessible to non-specialists while remaining mathematically grounded | TCAV introduced concept activation vectors; Captum implements TCAV as a reusable concept-interpreter API. citeturn9search7turn18search17turn18search2 |
| Next | **Semantic labeling and feature visualization** | What a neuron/channel appears to represent, which concept best aligns with it, and what input pattern maximally excites it | Activations, model gradients, and optionally concept datasets | This is one of the best bridges from “math” to “human intuition,” especially in vision models | Network Dissection quantifies unit-concept alignment; Distill’s feature-visualization work remains foundational; Lucent packages this workflow for PyTorch. citeturn9search10turn9search2turn8search2turn11search5 |
| Now | **Checkpoint-based data influence with TracIn** | Which training examples most helped or hurt a prediction, and how influence changes across training | Checkpoints, gradients, losses, and access to training data | This is especially well matched to your checkpointing design and is one of the clearest differentiators you can add | TracIn was explicitly designed around saved checkpoints and gradient descent traces; Captum ships practical TracInCP tutorials and APIs. citeturn14search0turn14search3turn18search18 |
| Later | **Sparse autoencoders and dictionary learning** | What sparse, more monosemantic features can be extracted from activations, beyond individual neurons | Large activation corpora, sparse training, and feature browsers | This is one of the strongest current directions in mechanistic interpretability, but it becomes much easier once you have reliable activation replay and caching | Anthropic’s dictionary-learning work and OpenAI’s SAE scaling work show why SAEs are now central in feature-based mechanistic interpretability; Neuronpedia already exposes SAEs and related browsing workflows. citeturn11search1turn20search1turn20search5turn20search4 |
| Later | **Causal interventions with activation patching and causal tracing** | Which activations or modules causally matter for a behavior, not just correlate with it | Activation hooks, clean/corrupted runs, and model-family-specific replay | This is the step from “observing” to “testing causal hypotheses,” especially for transformers | Causal tracing was used to localize factual associations in GPT; TransformerLens provides activation patching as a standard workflow. citeturn11search2turn11search12turn8search1turn8search9 |
| Next | **Sharpness and curvature metrics** | Whether checkpoints move into sharper or flatter regions, and how curvature evolves during training | Model replay, data, gradients, and Hessian approximations | This is valuable for serious observability of training dynamics, but it needs a replay path and careful approximation choices | Recent work analyzed sharpness along gradient-descent trajectories and tracked the top Hessian eigenvalue as a key signal. citeturn12search10turn12search2 |

My recommended shipping order is straightforward. **First wave:** trajectory geometry, spectra/rank, PCA/UMAP, and TracIn. **Second wave:** activation replay, CKA/SVCCA, probes, TCAV, and semantic labeling/feature visualization. **Third wave:** transformer-specific causal tools and SAEs. That ordering fits both the literature and the current strengths of your codebase. citeturn14search0turn23search5turn9search7turn11search1turn11search12turn25view3

## Strengths and weaknesses of the current approach

The strongest part of your current approach is its **simplicity and composability**. One file per snapshot plus an index is easy to reason about, append-friendly, and naturally supports lazy post-hoc analysis. The current `SnapshotCollection` already gives you exactly the sort of layer/epoch slicing a GUI would need, and the per-layer read path is already parallelized. The abstract `Backend` and `Format` interfaces are also the right instinct if you want to keep the ecosystem open rather than trapped inside one storage scheme. citeturn25view3turn7view0turn7view2turn2view0

A second strength is that the project already behaves like a **research substrate** rather than a demo toy. The examples cover small MLPs, CNNs on MNIST and CIFAR-10, and a grokking-style transformer experiment with 15k optimization steps, which is exactly the kind of cross-epoch setting where trajectory analysis becomes scientifically interesting rather than merely decorative. citeturn2view0

The biggest weakness is also the most important design decision in the repository today: the core snapshot logic currently iterates over `model.named_parameters()` and writes parameter tensors plus scalar metadata, but it does not capture activations, gradients, or hooks, and it does not appear to capture buffers such as BatchNorm running statistics. That means the package is currently strongest for **weight observability**, but most modern interpretability methods of highest value—CKA, probes, TCAV, sparse autoencoders, activation patching, tuned lenses—require **data-conditioned internal activations** and often gradients as well. citeturn6view2turn22view0turn24view0turn24view1turn24view2turn8search7turn10search5turn9search7turn11search1turn8search1

A related weakness is that **checkpoint-everything scales linearly in both storage and read cost with parameter count times snapshot count**. That is perfectly reasonable for small and medium models and is still useful for large models when sampling is smart, but it becomes the bottleneck much sooner than most users expect. The right answer is not to abandon checkpointing; it is to add selective capture, replay-based activation extraction, derived-result caching, and alternative storage formats for different artifact types. This is especially important if you want a browser UI rather than notebook-only workflows. That inference follows directly from the repo’s design of writing all parameters into independent snapshot files at each epoch or step. citeturn25view3turn6view2

There is also a product-positioning weakness if you are not careful: if NeuroInquisitor tries to reimplement every attribution method, every transformer-specific circuit method, and every dashboard feature internally, it will end up in direct competition with more mature specialized tools. Captum already covers many attribution, TCAV, and influence methods; TorchLens already solves “extract hidden activations from arbitrary PyTorch models”; TransformerLens already solves many transformer-specific activation access and patching workflows; and Neuronpedia already demonstrates an open-source browser-based interpretability platform with probes, UMAP, SAEs, dashboards, and autointerp. Your comparative advantage is the **time axis across training** plus an **open artifact layer**. citeturn18search3turn20search11turn20search2turn20search4

My blunt recommendation is this: **keep the current checkpoint-first architecture, but stop treating weights as the only first-class internal object.** Make checkpoints the anchor, then add replay-driven activations, gradients, optimizer state, and caches for derived analyses. That preserves what is elegant in the current design while opening the door to the methods the field actually uses. citeturn6view2turn25view3turn19search0

## Design recommendations for an open ecosystem

**Keep the core deliberately small.** The kernel of NeuroInquisitor should be “capture, index, replay, cache, and expose public analysis contracts.” That is much easier to maintain than a giant monolith, and it aligns with the way modern tools are split today: Captum for attribution and concepts, TorchLens for arbitrary activation extraction, TransformerLens for transformer internals, and Neuronpedia for web-native exploration. Adapters to those ecosystems are strategically better than trying to absorb all of them. citeturn18search3turn20search11turn20search2turn20search4

**Add activation replay instead of always-on activation logging.** Storing every activation for every batch and every epoch is usually too expensive. A better design is: save checkpoints and metadata during training; later, let the user choose checkpoint(s), modules, and a dataset slice; replay the model with hooks; save the resulting activations/gradients as derived artifacts. This is the design that unlocks CKA, probes, TCAV, feature visualization, SAEs, and patching without exploding storage. Tools like TorchLens and TransformerLens validate the value of activation-first infrastructure, and TracIn shows how powerful checkpoint-plus-replay workflows can be. citeturn20search11turn20search2turn14search0turn8search7turn10search5turn9search7turn11search1

**Use a real plugin protocol.** Python entry points are a standard way to discover plugins, and pluggy is the battle-tested hook system behind pytest’s ecosystem. That maps extremely well to your “I do not want to make this closed” goal. I would define analyzer hooks such as `discover_inputs`, `run_analysis`, `materialize_artifacts`, and `render_panel_spec`, then let third-party packages register analyzers without ever modifying the NeuroInquisitor core. citeturn16search0turn16search1

**Split storage by artifact type.** Keep HDF5 as the local default for now because it matches the current code and is simple. For tensor-heavy cloud/object-store use cases, add Zarr because it is explicitly designed for large N-dimensional arrays and object stores. For secure tensor serialization and partial/fast reads, add Safetensors. For derived tables—metrics, probe scores, similarity matrices in long-form, metadata, search indexes—use Arrow/Parquet because Arrow is columnar and PyArrow provides multithreaded adapters. If you later want a live reader against a single actively written HDF5 file, SWMR exists, but your current one-file-per-snapshot architecture likely remains simpler for the first GUI release. citeturn21search19turn21search5turn21search12turn21search2turn21search3turn21search7

**Design the GUI as a client of the public APIs, not as special internal logic.** Users now expect linked interactive views: low-dimensional scatterplots, searchable embeddings, selection linking, and view-to-sample navigation. TensorBoard’s Projector, FiftyOne’s Embeddings panel, and Captum Insights all validate that pattern, and Neuronpedia shows it can be extended into a richer interpretability platform. Technically, a FastAPI backend with WebSockets for job status and a lightweight React frontend is a strong fit; more important than the framework choice is that every UI action should map to public analysis endpoints and cached derived artifacts. citeturn26view3turn15search2turn18search1turn18search8turn20search4turn16search2

**Make the first GUI screens brutally practical.** I would start with five panels: a run browser; a layer/epoch selector; a tensor heatmap and summary stats panel; a spectrum/rank panel; and a projection/similarity panel. Add a compare mode that always answers “epoch A vs epoch B” or “layer A vs layer B”, because comparison is where checkpoint workflows become useful. Your existing `select(epochs=..., layers=...)`, `by_epoch`, and `by_layer` affordances are already the right primitives for this. citeturn7view0turn7view2turn25view3

**Treat AI code generation as optional plugin authoring, not as magic.** There is now clear precedent for open-source interpretability platforms that include API-backed explanation workflows, but the right way to keep your ecosystem open is to make the AI assistant produce plain Python analyzers or notebooks that users can inspect, edit, save, and rerun without the assistant. In practice, I would let users enter an API key, ask for an analysis, review the generated code diff, and then execute it in an isolated worker with time/memory limits and read-only access to selected artifacts. The key idea is that AI should generate **normal analyzers in your public plugin format**, not hidden internal code paths. citeturn20search0turn20search4turn16search0turn16search1

**Do not reimplement mature attribution libraries from scratch.** For per-input attribution methods, build adapters to Captum and make NeuroInquisitor responsible for **cross-checkpoint orchestration and provenance** rather than for owning every attribution algorithm. That is a better use of engineering effort and fits how the ecosystem is already organized. citeturn18search10turn18search12turn18search3

## Markdown roadmap for the next build cycle

The roadmap below assumes the product direction above: keep the checkpoint-first core, add replay and derived artifacts, expose a plugin API, then build the GUI and optional AI assistant on top of those public primitives. That sequencing is the safest way to stay open while still moving quickly toward a very usable research tool. citeturn25view3turn16search0turn20search4turn21search5turn21search2

```md
# NeuroInquisitor Next-Step Objectives

## Product objective

Build NeuroInquisitor into an open, trajectory-first interpretability and observability toolkit for PyTorch that:
- captures model state across training,
- replays selected checkpoints to compute activations and gradients on demand,
- exposes a stable plugin API for third-party analyses,
- ships a minimal but powerful web GUI,
- and optionally lets users use an LLM to generate analyzers as ordinary plugins.

## Guiding principles

- Keep the core small and stable.
- Separate raw artifacts from derived artifacts.
- Make every GUI capability available from Python and the API.
- Treat AI-generated code as user-visible, reviewable, and reproducible.
- Prefer adapters to mature ecosystems over reimplementing everything.
- Optimize for selected layers, selected epochs, and selected dataset slices.

## Non-goals

- Do not build a closed platform.
- Do not hard-code analyses into the web app.
- Do not store all activations for all batches by default.
- Do not make the AI assistant mandatory for any workflow.
- Do not try to replace W&B, TensorBoard, Captum, or TransformerLens wholesale.

## Architecture target

### Core layers

- `capture/`
  - checkpoint capture
  - replay capture for activations / gradients / logits / optimizer state
- `artifacts/`
  - manifest schema
  - raw tensor store
  - derived artifact store
- `analysis/`
  - built-in analyzers
  - analyzer registry
- `api/`
  - read-only and job endpoints
- `web/`
  - frontend panels that consume API results
- `plugins/`
  - entry-point-discovered third-party analyzers

### Artifact classes

- RunManifest
- SnapshotRef
- TensorArtifact
- ActivationArtifact
- GradientArtifact
- DerivedTableArtifact
- DerivedTensorArtifact
- AnalysisResult
- PanelSpec

## Sprint Alpha

### Goal

Stabilize the artifact model and prepare the codebase for replay and plugins.

### Tasks

- [ ] `NI-ALPHA-001` Add a versioned manifest schema.
  - Create `src/neuroinquisitor/schema.py`.
  - Define typed models for run metadata, snapshot refs, layer metadata, and derived artifact refs.
  - Include manifest versioning and migration hooks.
  - Acceptance:
    - Existing runs can be read.
    - New runs write a schema version.
    - Tests cover round-trip serialization.

- [ ] `NI-ALPHA-002` Add richer run metadata capture.
  - Extend snapshot metadata support to store:
    - git commit if available,
    - training config,
    - optimizer class name,
    - dtype / device,
    - model class path.
  - Acceptance:
    - Metadata is optional.
    - Missing fields do not break load.
    - Manifest docs updated.

- [ ] `NI-ALPHA-003` Add buffer capture as an optional feature.
  - Add `capture_buffers: bool = False` to capture config.
  - Store buffers under a separate namespace from parameters.
  - Acceptance:
    - Buffers are distinguishable in the manifest.
    - Existing parameter-only paths still work.

- [ ] `NI-ALPHA-004` Add capture policy objects.
  - Create a `CapturePolicy` abstraction that can express:
    - parameter capture,
    - buffer capture,
    - optimizer capture,
    - replay capture requests.
  - Acceptance:
    - Policies are serializable.
    - Policies are referenced by manifest entries.

## Sprint Beta

### Goal

Introduce replay-based activation and gradient extraction.

### Tasks

- [ ] `NI-BETA-001` Implement `ReplaySession`.
  - Create `src/neuroinquisitor/replay.py`.
  - API should accept:
    - checkpoint selector,
    - model factory or loader callback,
    - dataloader / iterable,
    - module selectors,
    - capture kinds (`activations`, `gradients`, `logits`).
  - Acceptance:
    - Replay works for a small MLP and a CNN example.
    - Errors are clear when module names are invalid.

- [ ] `NI-BETA-002` Add hook-based activation capture.
  - Support forward hooks for selected modules.
  - Support reduction modes:
    - raw batch outputs,
    - mean over batch,
    - pooled statistics.
  - Acceptance:
    - Capture does not require storing all batches by default.
    - Captured artifact sizes are reported.

- [ ] `NI-BETA-003` Add gradient capture for selected modules.
  - Support backward hooks or explicit autograd collection.
  - Allow per-example and aggregated modes.
  - Acceptance:
    - Gradient capture works for a classification example.
    - Tests verify shape correctness.

- [ ] `NI-BETA-004` Add dataset slice abstraction.
  - Create a small API for selecting:
    - first N,
    - random N with seed,
    - class-balanced N if labels exist,
    - explicit indices.
  - Acceptance:
    - Slice choice is stored in derived-artifact metadata.

## Sprint Gamma

### Goal

Ship the first built-in analyzers and caching layer.

### Tasks

- [ ] `NI-GAMMA-001` Implement `trajectory_stats` analyzer.
  - Metrics:
    - L2 distance from init,
    - cosine similarity to init / final,
    - update norm per step,
    - velocity / acceleration summaries.
  - Acceptance:
    - Results export to a table artifact.
    - Example notebook or script added.

- [ ] `NI-GAMMA-002` Implement `spectrum_rank` analyzer.
  - Metrics:
    - singular values,
    - effective rank / stable-rank-style summary,
    - spectral norm,
    - Frobenius norm.
  - Acceptance:
    - Works on linear and conv weights.
    - Caches results as derived artifacts.

- [ ] `NI-GAMMA-003` Implement `projection_embed` analyzer.
  - Support PCA first.
  - Add optional UMAP behind an extra dependency.
  - Acceptance:
    - Coordinates saved as a derived table artifact.
    - API returns plot-ready data.

- [ ] `NI-GAMMA-004` Implement `similarity_compare` analyzer.
  - Start with CKA.
  - Add hook point for SVCCA / PWCCA later.
  - Acceptance:
    - Compare epochs within a run.
    - Compare two runs if shapes are compatible.

- [ ] `NI-GAMMA-005` Implement `probe_linear` analyzer.
  - Train a simple linear probe on replayed activations.
  - Save probe metrics and coefficients.
  - Acceptance:
    - Works on a labeled toy dataset.
    - Handles train/val split deterministically.

- [ ] `NI-GAMMA-006` Implement `influence_tracin` analyzer.
  - Start with a practical checkpoint-based approximation.
  - Restrict first version to supported loss shapes and classification.
  - Acceptance:
    - Returns top helpful / harmful examples for a query item.

## Sprint Delta

### Goal

Open the ecosystem through plugins and richer storage options.

### Tasks

- [ ] `NI-DELTA-001` Add analyzer registry with Python entry points.
  - Define an entry-point group such as `neuroinquisitor.analyzers`.
  - Auto-discover analyzers at runtime.
  - Acceptance:
    - A demo external package can register an analyzer.

- [ ] `NI-DELTA-002` Add a hook-based plugin manager.
  - Introduce hook specs for:
    - analyzer registration,
    - artifact materialization,
    - panel spec generation.
  - Acceptance:
    - Plugins can extend without monkeypatching core code.

- [ ] `NI-DELTA-003` Add derived table storage with Parquet.
  - Use Parquet for tables and long-form metrics.
  - Acceptance:
    - Derived tables are queryable without loading raw tensors.

- [ ] `NI-DELTA-004` Add optional tensor-derived storage extras.
  - Add optional extras for:
    - Zarr,
    - Safetensors.
  - Acceptance:
    - Derived tensor artifacts can target different backends.
    - Docs explain when to use each.

- [ ] `NI-DELTA-005` Add CLI surface.
  - Commands:
    - `ni runs list`
    - `ni analyze run`
    - `ni plugins list`
    - `ni manifest show`
  - Acceptance:
    - CLI uses the same public APIs as Python callers.

## Sprint Epsilon

### Goal

Release the first web GUI.

### Tasks

- [ ] `NI-EPSILON-001` Build a FastAPI backend.
  - Endpoints:
    - list runs
    - get manifest
    - list layers
    - list epochs / steps
    - run analyzer job
    - poll job status
    - fetch analysis result
  - Acceptance:
    - API docs load.
    - Basic job lifecycle tested.

- [ ] `NI-EPSILON-002` Add background job execution.
  - Use a simple local worker first.
  - Persist job state in a lightweight store.
  - Acceptance:
    - Long-running analyses do not block API requests.

- [ ] `NI-EPSILON-003` Build the first frontend panels.
  - Panels:
    - run browser
    - layer / epoch selector
    - heatmap panel
    - spectrum panel
    - projection panel
    - similarity matrix panel
  - Acceptance:
    - Selecting a point in projection view updates the detail panel.
    - Compare mode works for two epochs.

- [ ] `NI-EPSILON-004` Add provenance and export.
  - Every result page shows:
    - analyzer name,
    - analyzer version,
    - input selectors,
    - dependency versions,
    - cache key.
  - Acceptance:
    - User can export result metadata and data.

## Sprint Zeta

### Goal

Add optional AI-assisted analyzer generation without closing the ecosystem.

### Tasks

- [ ] `NI-ZETA-001` Define an analyzer-generation contract.
  - The LLM must output:
    - a Python analyzer file,
    - metadata,
    - dependencies,
    - a short explanation,
    - tests when possible.
  - Acceptance:
    - Contract documented and validated by schema.

- [ ] `NI-ZETA-002` Add review-first code generation flow.
  - User sees generated code before execution.
  - User can save it as:
    - temporary analyzer,
    - local plugin,
    - exportable file.
  - Acceptance:
    - No generated code executes automatically.

- [ ] `NI-ZETA-003` Add isolated execution path.
  - Run generated analyzers in an isolated worker process.
  - Restrict accessible paths to selected artifacts.
  - Add timeouts and memory limits.
  - Acceptance:
    - Failed analyzers return clear errors.
    - Core process remains healthy.

- [ ] `NI-ZETA-004` Add reproducibility logging for generated analyzers.
  - Save prompt, model identifier, generated code hash, execution result, and artifact outputs.
  - Acceptance:
    - Generated analyses can be rerun without the LLM.

## Definition of done

A sprint item is done only if:
- public API docstrings are written,
- at least one integration test exists,
- manifest / provenance behavior is covered,
- failure modes are explicit,
- and a small example demonstrates the feature.

## Recommended initial built-ins

Ship these built-ins first:
- trajectory_stats
- spectrum_rank
- projection_embed
- similarity_compare
- probe_linear
- influence_tracin

Ship these after replay is stable:
- tcav_adapter
- feature_viz_adapter
- network_dissection_adapter
- sae_dictionary
- activation_patching_adapter

## Recommended public APIs

```python
from neuroinquisitor import NeuroInquisitor
from neuroinquisitor.analysis import run_analysis
from neuroinquisitor.replay import ReplaySession
from neuroinquisitor.plugins import list_analyzers

col = NeuroInquisitor.load("./runs/exp_a", epochs=range(0, 20), layers="encoder.0.weight")

result = run_analysis(
    analyzer="trajectory_stats",
    run="./runs/exp_a",
    layers=["encoder.0.weight"],
    epochs=range(0, 20),
)

replay = ReplaySession(
    run="./runs/exp_a",
    checkpoint=10,
    model_factory=build_model,
    dataset_slice={"kind": "random", "n": 256, "seed": 7},
    modules=["encoder.block1", "encoder.block2"],
    capture=["activations"],
)
```

## Final implementation note

The product should always preserve this invariant:

**Anything the GUI can do, a Python script can do. Anything the AI can generate, a user can save and own as a normal plugin.**
```

