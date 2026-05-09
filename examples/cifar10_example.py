"""CIFAR-10 classification — full NeuroInquisitor feature showcase.

Demonstrates every implemented capability:
  • CapturePolicy  — capture parameters, buffers, and optimizer state
  • RunMetadata    — attach training provenance to the run
  • Snapshots      — weight + buffer checkpoints with per-epoch metadata
  • SnapshotCollection — by_epoch, by_layer, select, to_state_dict, to_numpy
  • ReplaySession  — activations, gradients, and logits via forward/backward hooks
  • Analyzer registry — register and run a custom weight-statistics analyzer
  • Visualization  — weight-evolution video, replay figure, and loss curves

Run:
    python examples/cifar10_example.py

Requires:
    pip install tqdm petname torchvision matplotlib pandas
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import petname
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from neuroinquisitor import (
    AnalyzerRequest,
    AnalyzerResult,
    AnalyzerSpec,
    CapturePolicy,
    NeuroInquisitor,
    PROVENANCE_COLUMNS,
    ReplaySession,
    RunMetadata,
    get_analyzer,
    list_analyzers,
    register,
    write_derived_table,
)
from cifar10_example_utils import (
    make_video,
    make_replay_video,
    make_combined_video,
    save_loss_curves,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CIFAR10Net(nn.Module):
    """Three-conv-layer CNN for CIFAR-10 (32×32 RGB → 10 classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # 32 → 16 → 8 → 4 (three max-pool ops)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x.flatten(1))
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Custom analyzer — weight statistics
# ---------------------------------------------------------------------------


class WeightStatsRequest(AnalyzerRequest):
    """Request for the weight-statistics analyzer."""
    # Inherits: run (str), layers (list[str]), epochs (list[int])


class WeightStatsResult(AnalyzerResult):
    """Per-layer mean, std, and near-zero sparsity across epochs."""
    rows: list[dict[str, Any]] = []


def _weight_stats_fn(request: WeightStatsRequest) -> WeightStatsResult:
    col = NeuroInquisitor.load(
        request.run,
        epochs=request.epochs or None,
        layers=request.layers or None,
    )
    rows: list[dict[str, Any]] = []
    run_id = Path(request.run).name
    for epoch in col.epochs:
        snap = col.by_epoch(epoch)
        for layer, w in snap.items():
            rows.append({
                "run_id":           run_id,
                "epoch":            epoch,
                "layer":            layer,
                "analyzer_name":    "weight_stats",
                "analyzer_version": "1.0.0",
                "mean":             float(w.mean()),
                "std":              float(w.std()),
                "sparsity":         float((np.abs(w) < 1e-6).mean()),
                "p95_abs":          float(np.percentile(np.abs(w), 95)),
            })
    return WeightStatsResult(
        analyzer_name="weight_stats",
        analyzer_version="1.0.0",
        run=request.run,
        rows=rows,
    )


WEIGHT_STATS_SPEC = AnalyzerSpec(
    name="weight_stats",
    version="1.0.0",
    required_inputs=["weights"],
    output_format="table",
    description="Per-layer mean, std, sparsity, and p95 absolute value across epochs.",
    fn=_weight_stats_fn,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    run_name = petname.generate(words=2, separator="-")
    run_dir  = Path(__file__).parent.parent / "outputs" / "CIFAR10_example" / run_name
    data_dir = Path(__file__).parent.parent / "outputs" / "CIFAR10_example" / "data"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}\n")

    # ── Data ──────────────────────────────────────────────────────────────────

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=pin)

    # ── Model + optimiser ─────────────────────────────────────────────────────

    model     = CIFAR10Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    loss_fn   = nn.CrossEntropyLoss()

    # ── NeuroInquisitor — CapturePolicy + RunMetadata ─────────────────────────
    #
    # CapturePolicy: opt-in to buffers (BatchNorm running stats) and optimizer
    #   state in addition to the default parameter snapshots.
    # RunMetadata: attach provenance so the run is self-describing on disk.

    policy = CapturePolicy(
        capture_parameters=True,
        capture_buffers=True,
        capture_optimizer=True,
        replay_activations=True,
        replay_gradients=True,
    )
    run_meta = RunMetadata(
        training_config={"batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4, "T_max": 60},
        optimizer_class="Adam",
        device=str(device),
        model_class="CIFAR10Net",
    )

    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=policy,
        run_metadata=run_meta,
    )

    # ── Training loop ─────────────────────────────────────────────────────────

    num_epochs = 5
    train_loss_history: list[float] = []
    test_loss_history:  list[float] = []
    accuracy_history:   list[float] = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{num_epochs}", unit="batch", leave=True) as pbar:
            for step, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        total = 0
        test_loss_acc  = torch.zeros(1, device=device)
        correct_acc    = torch.zeros(1, device=device, dtype=torch.long)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                test_loss_acc += loss_fn(out, labels)
                correct_acc   += (out.argmax(1) == labels).sum()
                total         += labels.size(0)

        avg_test_loss = test_loss_acc.item() / len(test_loader)
        acc = correct_acc.item() / total
        test_loss_history.append(avg_test_loss)
        accuracy_history.append(acc)

        tqdm.write(f"  → train loss={avg_train_loss:.4f}  test loss={avg_test_loss:.4f}  acc={acc:.1%}")

        # Snapshot weights, buffers, and optimizer state for this epoch
        observer.snapshot(
            epoch=epoch,
            step=step,
            metadata={"loss": avg_train_loss, "test_loss": avg_test_loss, "accuracy": acc},
        )

    observer.close()
    print(f"\nTraining done. {num_epochs} snapshots saved.")

    # ── SnapshotCollection — post-training queries ─────────────────────────────

    print("\n── SnapshotCollection ──")
    col = NeuroInquisitor.load(run_dir)

    print(f"  Available epochs : {col.epochs}")
    print(f"  Captured layers  : {len(col.layers)} tensors")
    print(f"  Total snapshots  : {len(col)}")

    snap_0 = col.by_epoch(0)
    print(f"  by_epoch(0) keys : {list(snap_0.keys())[:4]} …")

    conv1_trajectory = col.by_layer("conv1.weight")
    print(f"  by_layer('conv1.weight') epochs : {list(conv1_trajectory.keys())}")

    sub = col.select(epochs=[0], layers=["fc1.weight", "fc2.weight"])
    print(f"  select(epochs=[0], layers=[fc1, fc2]) : {sub.layers}")

    arrays = col.to_numpy(epoch=0, layers=["conv1.weight"])
    print(f"  to_numpy(epoch=0, layers=['conv1.weight']) shape : {arrays['conv1.weight'].shape}")

    state = col.to_state_dict(epoch=0)
    restored = CIFAR10Net().to(device)
    restored.load_state_dict(state, strict=False)
    print(f"  to_state_dict(epoch=0) → model restored (strict=False)")

    # ── ReplaySession — activations, gradients, and logits (per epoch) ──────────
    #
    # Replay every saved checkpoint on the same 128 test samples so activations
    # and gradients can be animated over training time.
    # activation_reduction="pool"   → (N, C) tensors; spatial dims collapsed.
    # gradient_mode="aggregated"    → mean gradient over the batch.

    print("\n── ReplaySession ──")
    replay_modules = ["conv1", "conv2", "fc1"]
    replay_history: list[dict[str, dict[str, np.ndarray]]] = []
    final_replay = None

    for epoch in tqdm(range(num_epochs), desc="  Replaying checkpoints", unit="ckpt", leave=True):
        replay = ReplaySession(
            run=run_dir,
            checkpoint=epoch,
            model_factory=CIFAR10Net,
            dataloader=test_loader,
            modules=replay_modules,
            capture=["activations", "gradients", "logits"],
            activation_reduction="pool",
            gradient_mode="aggregated",
            dataset_slice=lambda samples: samples[:128],
            slice_metadata={"description": "first 128 test samples"},
        )
        final_replay = replay.run()
        replay_history.append({
            "activations": final_replay.activations.to_numpy(),
            "gradients":   final_replay.gradients.to_numpy(),
        })

    print(f"  Samples replayed : {final_replay.metadata.n_samples}")
    print(f"  Logits shape     : {final_replay.logits.shape}")
    for name in replay_modules:
        act_shape  = final_replay.activations[name].shape
        grad_shape = final_replay.gradients[name].shape
        print(f"  {name:6s} — activations {tuple(act_shape)}  gradients {tuple(grad_shape)}")

    # ── Analyzer registry ─────────────────────────────────────────────────────

    print("\n── Analyzer registry ──")
    register(WEIGHT_STATS_SPEC)

    for spec in list_analyzers():
        print(f"  {spec.name} v{spec.version} — {spec.description}")

    spec = get_analyzer("weight_stats")
    stats_result = spec.fn(WeightStatsRequest(run=str(run_dir)))
    print(f"  Analyzer produced {len(stats_result.rows)} rows")

    stats_df = pd.DataFrame(stats_result.rows)
    assert PROVENANCE_COLUMNS.issubset(stats_df.columns), "Missing provenance columns"
    stats_path = write_derived_table(stats_df, run_dir / "weight_stats.parquet")
    print(f"  Derived table written → {stats_path.name}")

    # ── Visualizations ────────────────────────────────────────────────────────

    print("\n── Visualizations ──")
    weight_history = [col.by_epoch(e) for e in range(num_epochs)]

    video_path = run_dir / "weights_over_time.mp4"
    print(f"  Generating weight video    → {video_path.name} …")
    result_path = make_video(weight_history, accuracy_history, video_path, fps=4)
    print(f"  Saved: {result_path.name}")

    replay_vid_path = run_dir / "activations_gradients.mp4"
    print(f"  Generating replay video    → {replay_vid_path.name} …")
    replay_result = make_replay_video(replay_history, replay_modules, accuracy_history, replay_vid_path, fps=4)
    print(f"  Saved: {replay_result.name}")

    combined_path = run_dir / "full_dashboard.mp4"
    print(f"  Generating combined video  → {combined_path.name} …")
    combined_result = make_combined_video(weight_history, replay_history, accuracy_history, replay_modules, combined_path, fps=4)
    print(f"  Saved: {combined_result.name}")

    curves_path = run_dir / "loss_curves.png"
    save_loss_curves(train_loss_history, test_loss_history, accuracy_history, curves_path)
    print(f"  Saved: {curves_path.name}")

    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
