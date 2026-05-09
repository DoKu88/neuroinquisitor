"""CIFAR-10 ViT training with NeuroInquisitor + TransformerLens mechanistic interpretability.

Trains a patch-based Vision Transformer (PatchViT) on CIFAR-10 using NeuroInquisitor
for per-epoch weight snapshots, then performs five TransformerLens analyses across
all captured checkpoints.

TransformerLens use cases demonstrated:
  1. run_with_cache     — capture every hook activation in one forward pass; inspect
                          the full hook namespace and activation shapes.
  2. Attention patterns — collect hook_pattern across epochs to see how heads specialize
                          from uniform attention to structured spatial routing.
  3. Logit lens         — project each layer's residual stream through the final
                          classifier to visualise where class information first emerges.
  4. Activation patching — replace hook_resid_post at each layer with activations from
                          the epoch-0 model; find which layers are critical to accuracy.
  5. Weight SVD         — track the top singular values of W_Q and W_K for every head
                          over training to detect head specialisation.

PatchViT exposes TransformerLens hook points at every interesting position:
  hook_embed
  blocks.{i}.hook_resid_pre / hook_resid_post
  blocks.{i}.attn.hook_q / hook_k / hook_v / hook_pattern / hook_z
  blocks.{i}.hook_attn_out / hook_mlp_out
  hook_ln_final

Run:
    python examples/transformerlens_use_examples/cifar10_transformerlens.py

Requires:
    pip install tqdm petname torchvision matplotlib transformer_lens
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import petname
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint, HookedRootModule

from neuroinquisitor import (
    CapturePolicy,
    NeuroInquisitor,
    ReplaySession,
    RunMetadata,
    SnapshotCollection,
)
from cifar10_transformerlens_utils import generate_visualizations

_cfg = yaml.safe_load(
    (Path(__file__).parent.parent / "configs" / "transformerlens_use_examples_cifar10_transformerlens.yaml").read_text()
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with TransformerLens hook points.

    Hook points (all tensors on the active device):
        hook_q, hook_k, hook_v   — per-head projections  (B, n_heads, S, d_head)
        hook_pattern             — softmax weights        (B, n_heads, S, S)
        hook_z                   — weighted value sum     (B, n_heads, S, d_head)
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.hook_q       = HookPoint()
        self.hook_k       = HookPoint()
        self.hook_v       = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_z       = HookPoint()

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        Q = self.hook_q(self._split(self.W_Q(x)))
        K = self.hook_k(self._split(self.W_K(x)))
        V = self.hook_v(self._split(self.W_V(x)))

        pattern = torch.softmax(Q @ K.transpose(-1, -2) * (self.d_head ** -0.5), dim=-1)
        pattern = self.hook_pattern(pattern)

        z   = self.hook_z(pattern @ V)
        out = self.W_O(z.transpose(1, 2).reshape(B, S, self.n_heads * self.d_head))
        return out


class ViTBlock(nn.Module):
    """Pre-norm transformer block with TransformerLens hook points.

    Hook points:
        hook_resid_pre  — residual stream entering  (B, S, d_model)
        hook_attn_out   — attention sublayer output (B, S, d_model)
        hook_mlp_out    — MLP sublayer output       (B, S, d_model)
        hook_resid_post — residual stream leaving   (B, S, d_model)

    Plus all hooks from MultiHeadSelfAttention (prefixed "attn.").
    """

    def __init__(self, d_model: int, n_heads: int, d_mlp: int) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = MLP(d_model, d_mlp)

        self.hook_resid_pre  = HookPoint()
        self.hook_attn_out   = HookPoint()
        self.hook_mlp_out    = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x        = self.hook_resid_pre(x)
        attn_out = self.hook_attn_out(self.attn(self.ln1(x)))
        x        = x + attn_out
        mlp_out  = self.hook_mlp_out(self.mlp(self.ln2(x)))
        x        = self.hook_resid_post(x + mlp_out)
        return x


class PatchViT(HookedRootModule):
    """Vision Transformer for CIFAR-10 built on TransformerLens' HookedRootModule.

    Inheriting HookedRootModule gives run_with_cache and run_with_hooks for free.
    Every HookPoint declared below is automatically registered in self.hook_dict
    by self.setup() and becomes addressable by name in those APIs.

    Full hook namespace:
        hook_embed
        blocks.{i}.hook_resid_pre / hook_resid_post
        blocks.{i}.attn.hook_q / hook_k / hook_v / hook_pattern / hook_z
        blocks.{i}.hook_attn_out / hook_mlp_out
        hook_ln_final
    """

    def __init__(
        self,
        n_classes:  int = 10,
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 4,
        d_mlp:      int = 256,
        patch_size: int = 4,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)

        self.hook_embed    = HookPoint()
        self.blocks        = nn.ModuleList([ViTBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)])
        self.ln_final      = nn.LayerNorm(d_model)
        self.hook_ln_final = HookPoint()
        self.classifier    = nn.Linear(d_model, n_classes)

        self.setup()  # HookedRootModule: traverse children, register all HookPoints

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(self.patchify(x))
        cls    = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = self.hook_embed(torch.cat([cls, tokens], dim=1) + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        cls_out = self.hook_ln_final(self.ln_final(tokens[:, 0]))
        return self.classifier(cls_out)


def model_factory() -> PatchViT:
    return PatchViT()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(data_dir: Path, device: torch.device) -> tuple[DataLoader, DataLoader]:
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
    train_loader = DataLoader(
        train_ds, batch_size=_cfg["train_batch_size"], shuffle=True,  num_workers=2,
        pin_memory=pin, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,  batch_size=_cfg["test_batch_size"], shuffle=False, num_workers=2,
        pin_memory=pin, persistent_workers=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(
    model:        nn.Module,
    optimizer:    optim.Optimizer,
    scheduler:    optim.lr_scheduler.LRScheduler,
    loss_fn:      nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    observer:     NeuroInquisitor,
    device:       torch.device,
    num_epochs:   int,
) -> tuple[list[float], list[float], list[float]]:
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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        total         = 0
        test_loss_acc = torch.zeros(1, device=device)
        correct_acc   = torch.zeros(1, device=device, dtype=torch.long)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                test_loss_acc += loss_fn(out, labels)
                correct_acc   += (out.argmax(1) == labels).sum()
                total         += labels.size(0)

        avg_test_loss = test_loss_acc.item() / len(test_loader)
        acc           = correct_acc.item() / total
        test_loss_history.append(avg_test_loss)
        accuracy_history.append(acc)

        tqdm.write(f"  → train loss={avg_train_loss:.4f}  test loss={avg_test_loss:.4f}  acc={acc:.1%}")

        observer.snapshot(
            epoch=epoch,
            step=step,
            metadata={"loss": avg_train_loss, "test_loss": avg_test_loss, "accuracy": acc},
        )

    observer.close()
    print(f"\nTraining done. {num_epochs} snapshots saved.")
    return train_loss_history, test_loss_history, accuracy_history


# ---------------------------------------------------------------------------
# TransformerLens analysis helpers
# ---------------------------------------------------------------------------


def _load_model(snapshots: SnapshotCollection, epoch: int, device: torch.device) -> PatchViT:
    model = model_factory().to(device)
    model.load_state_dict(snapshots.to_state_dict(epoch), strict=False)
    model.eval()
    return model


def _probe_batch(
    test_loader: DataLoader,
    n_samples:   int,
    device:      torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images_list, labels_list = [], []
    collected = 0
    for images, labels in test_loader:
        take = min(n_samples - collected, images.shape[0])
        images_list.append(images[:take])
        labels_list.append(labels[:take])
        collected += take
        if collected >= n_samples:
            break
    return torch.cat(images_list).to(device), torch.cat(labels_list).to(device)


# --- Use Case 1: run_with_cache ---


def demo_run_with_cache(model: PatchViT, images: torch.Tensor) -> None:
    """Show the complete hook namespace and activation shapes from a single forward pass."""
    print(f"\n── Use Case 1: run_with_cache ──")
    print(f"  Hook points registered : {len(model.hook_dict)}")
    for name in list(model.hook_dict.keys())[:5]:
        print(f"    {name}")
    print("    …")

    with torch.no_grad():
        _, cache = model.run_with_cache(images)

    print(f"  Cached activations     : {len(cache)}")
    for name, tensor in list(cache.items())[:8]:
        print(f"    {name:50s}  {tuple(tensor.shape)}")
    print("    …")


# --- Use Case 2: Attention patterns across epochs ---


def collect_attention_patterns(
    snapshots: SnapshotCollection,
    images:    torch.Tensor,
    n_layers:  int,
    device:    torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Cache hook_pattern for every layer at every epoch.

    Returns a list (one per epoch) of dicts:
        "blocks.{i}.attn.hook_pattern" -> Tensor (batch, n_heads, seq, seq)
    """
    keys     = {f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)}
    results: list[dict[str, torch.Tensor]] = []

    for epoch in tqdm(snapshots.epochs, desc="  Attention patterns", unit="epoch", leave=True):
        model = _load_model(snapshots, epoch, device)
        with torch.no_grad():
            _, cache = model.run_with_cache(images, names_filter=lambda n: n in keys)
        results.append({k: cache[k].cpu() for k in keys if k in cache})

    return results


# --- Use Case 3: Logit lens ---


def compute_logit_lens(
    snapshots: SnapshotCollection,
    images:    torch.Tensor,
    n_layers:  int,
    device:    torch.device,
) -> np.ndarray:
    """Project residual stream at each layer → class probabilities via ln_final + classifier.

    Returns array of shape (n_epochs, n_layers, n_classes) with mean class probabilities
    over the probe batch, revealing at which depth class information becomes readable.
    """
    keys        = {f"blocks.{i}.hook_resid_post" for i in range(n_layers)}
    n_epochs    = len(snapshots.epochs)
    result      = np.zeros((n_epochs, n_layers, 10))

    for ei, epoch in enumerate(tqdm(snapshots.epochs, desc="  Logit lens", unit="epoch", leave=True)):
        model = _load_model(snapshots, epoch, device)
        with torch.no_grad():
            _, cache = model.run_with_cache(images, names_filter=lambda n: n in keys)
            for li in range(n_layers):
                key      = f"blocks.{li}.hook_resid_post"
                cls_resid = cache[key][:, 0, :]  # CLS token: (B, d_model)
                logits   = model.classifier(model.ln_final(cls_resid))
                result[ei, li] = logits.softmax(dim=-1).cpu().numpy().mean(axis=0)

    return result


# --- Use Case 4: Activation patching ---


def activation_patching_accuracy(
    snapshots: SnapshotCollection,
    images:    torch.Tensor,
    labels:    torch.Tensor,
    n_layers:  int,
    device:    torch.device,
) -> np.ndarray:
    """Patch epoch-0 residual stream into the final model, one layer at a time.

    Returns accuracy array of shape (n_layers,).  A large accuracy drop when patching
    layer i means the final model relies on something that layer-i learned late in training.
    """
    final_epoch = snapshots.epochs[-1]
    early_epoch = snapshots.epochs[0]

    model_final = _load_model(snapshots, final_epoch, device)
    model_early = _load_model(snapshots, early_epoch, device)

    keys = {f"blocks.{i}.hook_resid_post" for i in range(n_layers)}

    with torch.no_grad():
        _, cache_early = model_early.run_with_cache(images, names_filter=lambda n: n in keys)

    accs = np.zeros(n_layers)
    for li in range(n_layers):
        key   = f"blocks.{li}.hook_resid_post"
        patch = cache_early[key].to(device)

        def make_patch_hook(p: torch.Tensor):
            def hook(value: torch.Tensor, hook=None) -> torch.Tensor:
                return p
            return hook

        with torch.no_grad():
            patched_logits = model_final.run_with_hooks(
                images,
                fwd_hooks=[(key, make_patch_hook(patch))],
            )
        accs[li] = (patched_logits.argmax(1) == labels).float().mean().item()

    return accs


# --- Use Case 5: Weight SVD ---


def compute_weight_svd(
    snapshots: SnapshotCollection,
    n_layers:  int,
    n_heads:   int,
    d_head:    int,
) -> dict[str, np.ndarray]:
    """Decompose W_Q and W_K per head across all checkpoints.

    Each weight matrix is (d_model, d_model); reshaped per-head to (d_head, d_model)
    then SVD'd to track spectral evolution.

    Returns dict with:
        "sigma_Q": ndarray (n_epochs, n_layers, n_heads, d_head)
        "sigma_K": ndarray (n_epochs, n_layers, n_heads, d_head)
    """
    epochs  = snapshots.epochs
    d_model = n_heads * d_head
    sig_Q   = np.zeros((len(epochs), n_layers, n_heads, d_head))
    sig_K   = np.zeros((len(epochs), n_layers, n_heads, d_head))

    for ei, epoch in enumerate(tqdm(epochs, desc="  Weight SVD", unit="epoch", leave=True)):
        weights = snapshots.by_epoch(epoch)
        for li in range(n_layers):
            W_Q = weights[f"blocks.{li}.attn.W_Q.weight"]  # (d_model, d_model)
            W_K = weights[f"blocks.{li}.attn.W_K.weight"]
            W_Q_h = W_Q.reshape(n_heads, d_head, d_model)
            W_K_h = W_K.reshape(n_heads, d_head, d_model)
            for hi in range(n_heads):
                sig_Q[ei, li, hi] = np.linalg.svd(W_Q_h[hi], compute_uv=False)
                sig_K[ei, li, hi] = np.linalg.svd(W_K_h[hi], compute_uv=False)

    return {"sigma_Q": sig_Q, "sigma_K": sig_K}


# ---------------------------------------------------------------------------
# Analysis entry point
# ---------------------------------------------------------------------------


def analyze(
    snapshots:  SnapshotCollection,
    run_dir:    Path,
    test_loader: DataLoader,
    num_epochs: int,
    n_layers:   int,
    n_heads:    int,
    d_head:     int,
    device:     torch.device,
) -> dict:
    print("\n── SnapshotCollection ──")
    print(f"  Available epochs : {snapshots.epochs}")
    print(f"  Captured layers  : {len(snapshots.layers)} tensors")
    print(f"  Total snapshots  : {len(snapshots)}")
    print(f"  by_epoch(0) keys : {list(snapshots.by_epoch(0).keys())[:4]} …")
    print(f"  by_layer('blocks.0.attn.W_Q.weight') epochs : {list(snapshots.by_layer('blocks.0.attn.W_Q.weight').keys())}")

    # Grab a fixed probe batch reused across all analyses for comparability
    probe_images, probe_labels = _probe_batch(test_loader, n_samples=256, device=device)

    print("\n── ReplaySession (NI baseline) ──")
    replay_modules = ["blocks.0", f"blocks.{n_layers - 1}"]
    result = ReplaySession(
        run=run_dir,
        checkpoint=num_epochs - 1,
        model_factory=model_factory,
        dataloader=DataLoader(
            TensorDataset(probe_images.cpu(), probe_labels.cpu()),
            batch_size=64,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        ),
        modules=replay_modules,
        capture=["activations", "gradients", "logits"],
        activation_reduction="pool",
        gradient_mode="aggregated",
        dataset_slice=lambda s: s[:128],
        slice_metadata={"description": "first 128 test samples"},
    ).run()
    print(f"  Samples replayed : {result.metadata.n_samples}")
    print(f"  Logits shape     : {result.logits.shape}")
    for name in replay_modules:
        if name in result.activations:
            print(f"  {name:20s}  activations {tuple(result.activations[name].shape)}")

    # ── Use Case 1 ──
    final_model = _load_model(snapshots, num_epochs - 1, device)
    demo_run_with_cache(final_model, probe_images[:16])

    # ── Use Case 2 ──
    print(f"\n── Use Case 2: Attention Pattern Evolution ──")
    attn_patterns = collect_attention_patterns(
        snapshots, probe_images[:32], n_layers, device,
    )
    first_pat = next(iter(attn_patterns[0].values()))
    print(f"  Epochs collected : {len(attn_patterns)}")
    print(f"  Pattern shape    : {tuple(first_pat.shape)}  (batch, n_heads, seq, seq)")

    # ── Use Case 3 ──
    print(f"\n── Use Case 3: Logit Lens ──")
    logit_lens_probs = compute_logit_lens(
        snapshots, probe_images[:64], n_layers, device,
    )
    print(f"  Array shape : {logit_lens_probs.shape}  (epochs, layers, classes)")
    dominant = CIFAR10_CLASSES[logit_lens_probs[-1, -1].argmax()]
    print(f"  Final epoch, deepest layer — dominant class: {dominant}")

    # ── Use Case 4 ──
    print(f"\n── Use Case 4: Activation Patching ──")
    patch_accs = activation_patching_accuracy(
        snapshots, probe_images[:128], probe_labels[:128], n_layers, device,
    )
    for li, acc in enumerate(patch_accs):
        print(f"  Epoch-0 patch at layer {li}  →  accuracy {acc:.1%}")

    # ── Use Case 5 ──
    print(f"\n── Use Case 5: Weight SVD ──")
    svd_data = compute_weight_svd(snapshots, n_layers, n_heads, d_head)
    print(f"  sigma_Q shape : {svd_data['sigma_Q'].shape}  (epochs, layers, heads, singular values)")
    top3 = svd_data["sigma_Q"][-1, 0, 0, :3]
    print(f"  Final epoch, layer 0, head 0 — top-3 σ(W_Q) : {top3.round(3)}")

    return {
        "attn_patterns":    attn_patterns,
        "logit_lens_probs": logit_lens_probs,
        "patch_accs":       patch_accs,
        "svd_data":         svd_data,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    N_LAYERS   = _cfg["n_layers"]
    N_HEADS    = _cfg["n_heads"]
    D_MODEL    = _cfg["d_model"]
    D_HEAD     = D_MODEL // N_HEADS
    D_MLP      = _cfg["d_mlp"]
    PATCH_SIZE = _cfg["patch_size"]
    NUM_EPOCHS = _cfg["num_epochs"]

    run_name = petname.generate(words=2, separator="-")
    run_dir  = Path(__file__).parent.parent.parent / "outputs" / "CIFAR10_TransformerLens" / run_name
    data_dir = Path(__file__).parent.parent.parent / "outputs" / "CIFAR10_TransformerLens" / "data"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name : {run_name}")
    print(f"Run dir  : {run_dir}/")
    print(f"Device   : {device}\n")

    train_loader, test_loader = load_data(data_dir, device)

    model     = PatchViT(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_mlp=D_MLP, patch_size=PATCH_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=_cfg["lr"], weight_decay=_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=_cfg["label_smoothing"])

    print(f"Model params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Hook points  : {len(model.hook_dict)}")
    print(f"Hook sample  : {list(model.hook_dict.keys())[:4]} …\n")

    policy = CapturePolicy(
        capture_parameters=True,
        capture_buffers=True,
        capture_optimizer=False,
        replay_activations=True,
        replay_gradients=True,
    )
    run_meta = RunMetadata(
        training_config={
            "batch_size": _cfg["train_batch_size"], "lr": _cfg["lr"], "weight_decay": _cfg["weight_decay"],
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            "d_mlp": D_MLP, "patch_size": PATCH_SIZE,
        },
        optimizer_class="AdamW",
        device=str(device),
        model_class="PatchViT",
    )
    observer = NeuroInquisitor(
        model,
        log_dir=run_dir,
        compress=True,
        create_new=True,
        capture_policy=policy,
        run_metadata=run_meta,
    )

    train_losses, test_losses, accuracy_history = train(
        model, optimizer, scheduler, loss_fn,
        train_loader, test_loader, observer,
        device, NUM_EPOCHS,
    )

    snapshots    = NeuroInquisitor.load(run_dir)
    analysis_out = analyze(
        snapshots, run_dir, test_loader,
        NUM_EPOCHS, N_LAYERS, N_HEADS, D_HEAD, device,
    )

    generate_visualizations(
        analysis_out["attn_patterns"],
        analysis_out["logit_lens_probs"],
        analysis_out["patch_accs"],
        analysis_out["svd_data"],
        accuracy_history, train_losses, test_losses,
        N_LAYERS, N_HEADS,
        run_dir,
    )
    print(f"\nAll outputs in: {run_dir}/")


if __name__ == "__main__":
    main()
