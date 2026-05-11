# Sprint 12: Qwen 7B Finetuning Showcase

**Goal**: Demonstrate NeuroInquisitor on a real 7B-parameter finetuning run on Modal, capturing selective layer snapshots to S3, and producing concrete interpretability findings about how model internals shift during finetuning. Covers both LoRA/QLoRA and full finetuning paths.

**Prerequisite**: Sprints 10 and 11 complete (bfloat16 fix, layer_filter, S3Backend, SafetensorsFormat).

---

## Tasks

- [ ] `NI-LLM-008` Write Qwen 7B finetuning example.
  - **New file**: `examples/qwen7b_finetune_example.py`

  The example must be runnable as a Modal function. It covers both LoRA and full finetuning paths via a top-level flag.

  **Helper: `make_layer_filter(model, lora_only=False) -> set[str]`**
  - Iterates `model.named_parameters()`, returns names matching any of:
    - Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
    - MLP: `up_proj`, `down_proj`, `gate_proj`
  - Excludes: `embed_tokens`, `lm_head`, `norm` (large, slow-moving, less interpretable)
  - When `lora_only=True`: further filter to names containing `lora_A` or `lora_B`
  - For LoRA, `make_layer_filter(model, lora_only=True)` reduces from ~7B to ~70M params captured (~1% of model, ~140 MB/snapshot).
  - For selective full FT, `make_layer_filter(model, lora_only=False)` captures ~2B params (~4 GB/snapshot bfloat16).

  **Storage setup block**:
  ```python
  s3_backend = S3Backend(
      bucket=os.environ["NI_S3_BUCKET"],
      prefix=f"qwen7b-runs/{run_name}",
      tmp_dir="/tmp/ni_uploads",     # Modal ephemeral SSD
      max_workers=4,
      cleanup_after_upload=True,     # keep /tmp usage bounded
  )
  fmt = SafetensorsFormat()
  ```

  **LoRA path**:
  - Load model in 4-bit via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`
  - Apply `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])` via `peft.get_peft_model()`
  - `observer = NeuroInquisitor(model, backend=s3_backend, format=fmt, layer_filter=make_layer_filter(model, lora_only=True))`
  - Snapshot every N steps: `observer.snapshot(step=global_step, metadata={"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})`
  - `observer.close()` at end of training loop

  **Full finetuning path**:
  - Same structure, no quantization, `layer_filter=make_layer_filter(model, lora_only=False)`
  - Recommend snapshotting less frequently (every epoch or every 500 steps) due to 4 GB snapshot size

  **Modal app structure** (described in comments):
  - `@app.function(gpu="A100-80GB", timeout=7200, secrets=[modal.Secret.from_name("aws-creds")])`
  - AWS credentials injected via Modal secret mapping `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
  - Model weights via Modal volume or downloaded fresh each run from HuggingFace

  - Acceptance:
    - Example runs end-to-end on Modal with a small dataset (e.g., 100 steps) without error.
    - Snapshots appear in S3 after `observer.close()`.
    - Both LoRA and full FT paths are exercised.

- [ ] `NI-LLM-009` Write post-training analysis script.
  - **New file**: `examples/qwen7b_analysis.py`

  A self-contained script that loads snapshots from S3 and produces four analyses. Each analysis section is runnable independently given a populated S3 prefix.

  **Setup**:
  ```python
  col = NeuroInquisitor.load(
      log_dir=".",  # not used — backend provides the index
      backend=S3Backend(bucket=os.environ["NI_S3_BUCKET"], prefix="qwen7b-runs/run-xyz"),
      format=SafetensorsFormat(),
      create_new=False,
  )
  ```

  **Analysis 1 — Weight trajectory per attention head** (`trajectory_stats`):
  ```python
  from neuroinquisitor.analyzers import trajectory_stats
  for layer_name in ["model.layers.0.self_attn.q_proj.weight",
                     "model.layers.15.self_attn.q_proj.weight",
                     "model.layers.31.self_attn.q_proj.weight"]:
      weights = col.by_layer(layer_name)   # {step: np.ndarray}
      df = trajectory_stats(weights)
      # Plot: df["l2_from_init"] and df["update_norm"] over steps
  ```
  Research question: *Which layers move most during finetuning? Do early layers stabilize before later ones?*

  **Analysis 2 — Rank dynamics** (`spectrum_rank`):
  ```python
  from neuroinquisitor.analyzers import spectrum_rank
  dfs = []
  for step in col.epochs:   # epochs = steps in this context
      dfs.append(spectrum_rank(col.by_epoch(step), epoch=step))
  df = pd.concat(dfs)
  # Plot: stable_rank and effective_rank per layer over steps
  ```
  Research question: *Does rank collapse occur in attention matrices? Does LoRA increase or decrease effective rank?*

  **Analysis 3 — Representational shift** (`similarity_compare` / CKA):
  ```python
  from neuroinquisitor.analyzers import similarity_compare
  # Requires replay to get activations — note this needs a dataloader
  # Run ReplaySession at step 0 and step N, compare
  result_0 = ReplaySession(run=..., checkpoint=0, model_factory=..., dataloader=...,
                           modules=["model.layers.15"], capture=["activations"]).run()
  result_N = ReplaySession(run=..., checkpoint=N, ...).run()
  df = similarity_compare(result_0.activations, result_N.activations)
  # CKA diagonal shows self-similarity; off-diagonal shows cross-layer similarity
  ```
  Research question: *How much do representations change from base model to finetuned? Which layers are most affected?*

  **Analysis 4 — Linear probe accuracy** (`probe_linear`):
  ```python
  from neuroinquisitor.analyzers import probe_linear
  result = ReplaySession(..., capture=["activations"]).run()
  df = probe_linear(result.activations, labels=task_labels)
  # Plot: val_accuracy per layer — where does task-relevant info emerge?
  ```
  Research question: *Does task-specific information emerge in earlier or later layers during finetuning?*

  - Acceptance:
    - All four analyses run without error given a populated S3 run prefix and a small dataset for replay.
    - Each analysis produces a `pd.DataFrame` that can be plotted or exported to CSV.
    - Script degrades gracefully when `NI_S3_BUCKET` is not set (prints clear instructions).

---

## Recommended Experimental Design

For researchers running this sprint's example:

| Parameter | LoRA run | Full FT run |
|---|---|---|
| Base model | `Qwen/Qwen2-7B` | `Qwen/Qwen2-7B` |
| Dataset | Alpaca-cleaned (52K) or subset | Same |
| Steps | 500–1000 | 100–200 |
| Snapshot freq | Every 50 steps | Every 25 steps |
| Layers captured | LoRA adapters only | Attn Q/K/V + MLP |
| Snapshot size | ~140 MB | ~4 GB |
| Total storage (20 snapshots) | ~2.8 GB | ~80 GB |
| S3 upload time (est. 100 Mbps) | ~11s | ~320s (async, non-blocking) |
| Modal GPU | A10G (24 GB) | A100-80GB |

---

## Testing

- `examples/qwen7b_finetune_example.py`: smoke test using a tiny model (e.g., `GPT-2` with bfloat16 cast) against a mocked S3 backend — verifies the flow without downloading 7B params.
- `examples/qwen7b_analysis.py`: smoke test using pre-written fixture snapshots (small toy model, stored in `tests/fixtures/`).
- No real S3 or Modal calls in CI — all backend calls mocked.

## Definition of Done

- Both LoRA and full FT paths run end-to-end on Modal with Qwen 7B.
- Four post-training analyses each produce a `pd.DataFrame` from real captured data.
- A researcher can fork the example, change the dataset and model, and get results with no changes to NI library code.
- All four analyses are documented with the research question each answers.
