from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gaussian_transformer import GaussianTransformer
from utils import gaussian_measure_nll, load_data


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    """Return all trainable parameters concatenated into one vector."""
    return torch.cat(
        [
            p.detach().reshape(-1).cpu()
            for p in model.parameters()
            if p.requires_grad
        ]
    )


def parameter_shapes(
    model: torch.nn.Module,
) -> list[tuple[str, tuple[int, ...], int]]:
    """Record trainable-parameter names, shapes and flattened sizes."""
    return [
        (name, tuple(p.shape), p.numel())
        for name, p in model.named_parameters()
        if p.requires_grad
    ]


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
# ---------------------------------------------------------------------------
# Pretraining on fixed D
# ---------------------------------------------------------------------------

def pretrain_model(
    model: GaussianTransformer,
    loader: DataLoader,
    device: torch.device,
    sigma_y_cholesky: torch.Tensor,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float | None,
) -> list[float]:
    """
    Pretrain the transformer on the SAME fixed dataset D.

    Returns the batch-mean NLL history.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: list[float] = []

    for epoch in range(1, num_epochs + 1):

        model.train()

        running_nll = 0.0
        num_batches = 0

        for source, target in loader:

            source = source.to(
                device,
                non_blocking=True,
            )

            target = target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            mean = model(source)

            loss = gaussian_measure_nll(
                y=target,
                mean=mean,
                Ly=sigma_y_cholesky,
            )

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

            optimizer.step()

            running_nll += loss.item()
            num_batches += 1

        epoch_nll = (
            running_nll / max(num_batches, 1)
        )

        history.append(epoch_nll)

        print(
            f"[pretrain] "
            f"epoch {epoch:04d}/{num_epochs:04d}"
            f" | NLL = {epoch_nll:.6f}",
            flush=True,
        )

    return history

# ---------------------------------------------------------------------------
# Plot pretrained model predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_pretrained_predictions(
    model: GaussianTransformer,
    source: torch.Tensor,
    target: torch.Tensor,
    latent: torch.Tensor,
    B0: torch.Tensor,
    device: torch.device,
    output_path: Path,
) -> None:
    """
    Plot pretrained-model predictions for the first four training measures.

    Each panel shows:
        - source point cloud X_i
        - target point cloud Y_i
        - predicted target mean m(X_i)
        - empirical target mean
        - true population mean B0 Z_i
    """

    model.eval()

    n_examples = min(
        4,
        source.shape[0],
    )

    # -----------------------------------------------------------------------
    # Predict first four training measures
    # -----------------------------------------------------------------------

    source_batch = source[
        :n_examples
    ].to(device)

    predicted_means = (
        model(source_batch)
        .cpu()
    )

    source_plot = source[
        :n_examples
    ].cpu()

    target_plot = target[
        :n_examples
    ].cpu()

    latent_plot = latent[
        :n_examples
    ].cpu()

    B0 = B0.cpu()

    # Empirical target means
    empirical_target_means = (
        target_plot.mean(dim=1)
    )

    # True population means B0 Z_i
    true_population_means = (
        latent_plot @ B0.T
    )

    # -----------------------------------------------------------------------
    # Numerical diagnostics
    # -----------------------------------------------------------------------

    prediction_error_truth = (
        torch.linalg.vector_norm(
            predicted_means
            - true_population_means,
            dim=1,
        )
    )

    prediction_error_empirical = (
        torch.linalg.vector_norm(
            predicted_means
            - empirical_target_means,
            dim=1,
        )
    )

    print(
        "\nPretrained-model diagnostic:",
        flush=True,
    )

    for i in range(n_examples):

        print(
            f"Measure {i}: "
            f"||prediction - B0 Z|| = "
            f"{prediction_error_truth[i].item():.6f}"
            f" | "
            f"||prediction - Ybar|| = "
            f"{prediction_error_empirical[i].item():.6f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    if source.shape[-1] != 2:
        raise ValueError(
            "Pretrained prediction plot currently requires d=2."
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 10),
    )

    axes = axes.flatten()

    for i in range(n_examples):

        ax = axes[i]

        X = source_plot[i].numpy()
        Y = target_plot[i].numpy()

        predicted_mean = (
            predicted_means[i]
            .numpy()
        )

        empirical_mean = (
            empirical_target_means[i]
            .numpy()
        )

        true_mean = (
            true_population_means[i]
            .numpy()
        )

        # ---------------------------------------------------------------
        # Point clouds
        # ---------------------------------------------------------------

        ax.scatter(
            X[:, 0],
            X[:, 1],
            s=15,
            alpha=0.35,
            label="Source",
        )

        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            s=15,
            alpha=0.35,
            label="Target",
        )

        # ---------------------------------------------------------------
        # Pretrained transformer prediction
        # ---------------------------------------------------------------

        ax.scatter(
            predicted_mean[0],
            predicted_mean[1],
            s=140,
            marker="x",
            linewidths=3,
            label=r"Prediction $m(X_i)$",
            zorder=10,
        )

        # ---------------------------------------------------------------
        # Empirical target mean
        # ---------------------------------------------------------------

        ax.scatter(
            empirical_mean[0],
            empirical_mean[1],
            s=130,
            marker="o",
            facecolors="none",
            linewidths=2,
            label=r"Empirical $\bar Y_i$",
            zorder=11,
        )

        # ---------------------------------------------------------------
        # True population mean
        # ---------------------------------------------------------------

        ax.scatter(
            true_mean[0],
            true_mean[1],
            s=160,
            marker="+",
            linewidths=3,
            label=r"True $B_0Z_i$",
            zorder=12,
        )

        ax.set_title(
            f"Training measure {i}"
        )

        ax.set_xlabel(
            "Coordinate 1"
        )

        ax.set_ylabel(
            "Coordinate 2"
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            alpha=0.2,
        )

    # Hide unused panels if fewer than four measures exist.
    for i in range(
        n_examples,
        4,
    ):
        axes[i].axis("off")

    # -----------------------------------------------------------------------
    # Shared legend
    # -----------------------------------------------------------------------

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(
            0.5,
            0.98,
        ),
    )

    fig.suptitle(
        "Pretrained transformer predictions on training data",
        fontsize=15,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.92]
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Pretrained prediction plot saved to: "
        f"{output_path.resolve()}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# SGD trajectory + SWA mean
# ---------------------------------------------------------------------------

def collect_trajectory(
    model: GaussianTransformer,
    loader: DataLoader,
    device: torch.device,
    sigma_y_cholesky: torch.Tensor,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    snapshot_every: int,
    max_snapshots: int,
    grad_clip: float | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[float],
]:
    """
    Continue SGD from the pretrained solution.

    Following the PCA construction in the paper:

      - collect snapshots during continued SGD;
      - maintain the running SWA mean;
      - form deviations from the SWA mean;
      - retain at most max_snapshots deviations.

    Returns
    -------
    swa_mean:
        Flattened SWA parameter vector.

    deviations:
        Matrix [M, D] used for PCA.

    snapshot_losses:
        NLL at each retained snapshot.
    """

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    swa_mean: torch.Tensor | None = None
    num_snapshots = 0

    deviations: deque[torch.Tensor] = deque(
        maxlen=max_snapshots
    )

    snapshot_losses: deque[float] = deque(
        maxlen=max_snapshots
    )

    for epoch in range(1, num_epochs + 1):

        model.train()

        running_nll = 0.0
        num_batches = 0

        for source, target in loader:

            source = source.to(
                device,
                non_blocking=True,
            )

            target = target.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            mean = model(source)

            loss = gaussian_measure_nll(
                y=target,
                mean=mean,
                Ly=sigma_y_cholesky,
            )

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

            optimizer.step()

            running_nll += loss.item()
            num_batches += 1

        epoch_nll = (
            running_nll / max(num_batches, 1)
        )

        if epoch % snapshot_every == 0:

            w_i = flatten_parameters(model)

            if swa_mean is None:
                swa_mean = w_i.clone()
            else:
                swa_mean = (
                    num_snapshots * swa_mean
                    + w_i
                ) / (num_snapshots + 1)

            num_snapshots += 1

            deviation = (
                w_i - swa_mean
            )

            deviations.append(
                deviation.clone()
            )

            snapshot_losses.append(
                epoch_nll
            )

            print(
                f"[trajectory] "
                f"epoch {epoch:04d}/{num_epochs:04d}"
                f" | NLL = {epoch_nll:.6f}"
                f" | snapshot {num_snapshots}"
                f" | retained "
                f"{len(deviations)}/{max_snapshots}",
                flush=True,
            )

    if swa_mean is None:
        raise RuntimeError(
            "No snapshots were collected. "
            "Increase num_epochs or reduce snapshot_every."
        )

    deviation_matrix = torch.stack(
        list(deviations),
        dim=0,
    )

    return (
        swa_mean,
        deviation_matrix,
        list(snapshot_losses),
    )


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def compute_pca_subspace(
    deviations: torch.Tensor,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute

        A = U S V^T

    and return the scaled PCA basis

        P = V_K diag(S_K).

    The Bayesian coordinates therefore have prior

        phi ~ N(0, I).
    """

    M, D = deviations.shape

    if M < 2:
        raise ValueError(
            "Need at least two snapshots for PCA."
        )

    rank = min(rank, M, D)

    if rank < 1:
        raise ValueError(
            "rank must be at least 1."
        )

    _, singular_values, Vh = torch.linalg.svd(
        deviations,
        full_matrices=False,
    )

    singular_values = singular_values[
        :rank
    ]

    Vh = Vh[:rank]

    basis = (
        Vh.T
        * singular_values.unsqueeze(0)
    )

    all_variance = (
        torch.linalg.svdvals(deviations)
        .square()
        .sum()
    )

    explained_ratio = (
        singular_values.square()
        / torch.clamp(
            all_variance,
            min=torch.finfo(
                torch.float32
            ).eps,
        )
    )

    return (
        basis,
        singular_values,
        explained_ratio,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Pretrain a Gaussian transformer on a fixed dataset, "
            "collect an SGD trajectory, and construct the PCA "
            "subspace for Bayesian subspace inference."
        )
    )

    parser.add_argument(
        "--fixed-data",
        type=Path,
        required=True,
        help=(
            "Fixed dataset containing source, target, latent, "
            "and dataset_config."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints"
            / "pca_subspace_fixed_D.pt"
        ),
    )

    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--trajectory-epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--trajectory-lr",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--pca-rank",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--pretrained-prediction-figure",
        type=Path,
        default=None,
        help="Output path for the pretrained prediction diagnostic plot.",
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        "PCA subspace from fixed training dataset D",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        f"Fixed dataset : "
        f"{args.fixed_data.resolve()}",
        flush=True,
    )

    print(
        f"Output        : "
        f"{args.output.resolve()}",
        flush=True,
    )

    print(
        f"Device        : {device}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Load fixed D
    # -----------------------------------------------------------------------

    print(
        "Loading fixed dataset...",
        flush=True,
    )

    fixed_data = (
        load_data(
            args.fixed_data, 
            required_keys={"source", "target", "latent", "dataset_config"},
            )
    )
    
    source = fixed_data["source"].float()
    target = fixed_data["target"].float()
    latent = fixed_data["latent"].float()
    dataset_config = fixed_data["dataset_config"]

    print(
        f"Source shape: {tuple(source.shape)}",
        flush=True,
    )

    print(
        f"Target shape: {tuple(target.shape)}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            source,
            target,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    sigma_y = torch.as_tensor(
        dataset_config["Sigma_Y"],
        dtype=torch.float32,
    )

    Ly = torch.linalg.cholesky(
        sigma_y
    ).to(device)

    model = GaussianTransformer(
        d=source.shape[-1],
        Ly=Ly.detach().cpu(),
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        dropout=0.0,
    ).to(device)

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {n_parameters:,}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Pretraining on fixed D
    # -----------------------------------------------------------------------

    print(
        "\nStarting pretraining...",
        flush=True,
    )

    pretrain_history = pretrain_model(
        model=model,
        loader=train_loader,
        device=device,
        sigma_y_cholesky=Ly,
        num_epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    # -----------------------------------------------------------------------
    # Optional pretrained-model diagnostic
    # -----------------------------------------------------------------------

    if args.pretrained_prediction_figure is not None:
        pretrained_figure_path = args.pretrained_prediction_figure
        
        print(
            "\nPlotting pretrained-model predictions...",
            flush=True,
            )
        
        plot_pretrained_predictions(
            model=model,
            source=source,
            target=target,
            latent=latent,
            B0=torch.as_tensor(
                dataset_config["B0"],
                dtype=torch.float32,
            ),
            device=device,
            output_path=pretrained_figure_path,
        )
    # -----------------------------------------------------------------------
    # SGD trajectory
    # -----------------------------------------------------------------------

    print(
        "\nStarting SGD trajectory collection...",
        flush=True,
    )

    (
        swa_mean,
        deviations,
        snapshot_losses,
    ) = collect_trajectory(
        model=model,
        loader=train_loader,
        device=device,
        sigma_y_cholesky=Ly,
        num_epochs=args.trajectory_epochs,
        learning_rate=args.trajectory_lr,
        weight_decay=args.weight_decay,
        snapshot_every=args.snapshot_every,
        max_snapshots=args.max_snapshots,
        grad_clip=args.grad_clip,
    )

    print(
        "\nTrajectory collection complete.",
        flush=True,
    )

    print(
        f"Deviation matrix: {tuple(deviations.shape)}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # PCA
    # -----------------------------------------------------------------------

    print(
        "\nComputing PCA...",
        flush=True,
    )

    (
        basis,
        singular_values,
        explained_ratio,
    ) = compute_pca_subspace(
        deviations=deviations,
        rank=args.pca_rank,
    )

    print(
        f"PCA basis shape: {tuple(basis.shape)}",
        flush=True,
    )

    print(
        "Singular values:",
        singular_values.numpy(),
        flush=True,
    )

    print(
        "Explained variance ratios:",
        explained_ratio.numpy(),
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Save compact Bayesian-subspace checkpoint
    # -----------------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        # Core Bayesian subspace representation.
        "swa_mean": swa_mean,
        "pca_basis": basis,
        "singular_values": singular_values,
        "explained_variance_ratio": explained_ratio,

        # Diagnostics only; no raw trajectory vectors are saved.
        "snapshot_losses": torch.tensor(
            snapshot_losses,
            dtype=torch.float64,
        ),

        "pretrain_history": pretrain_history,

        "pca_rank": int(basis.shape[1]),
        "num_snapshots": int(deviations.shape[0]),
        "num_parameters": int(swa_mean.numel()),

        "parameter_info": parameter_shapes(model),

        "model_config": {
            "d": source.shape[-1],
            "hidden_dim": 128,
            "num_heads": 4,
            "num_layers": 3,
            "ff_dim": 256,
            "dropout": 0.0,
        },

        "dataset_config": dataset_config,

        "fixed_data_path": str(
            args.fixed_data.resolve()
        ),

        "trajectory_config": {
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_lr": args.pretrain_lr,
            "trajectory_epochs": args.trajectory_epochs,
            "trajectory_lr": args.trajectory_lr,
            "weight_decay": args.weight_decay,
            "snapshot_every": args.snapshot_every,
            "max_snapshots": args.max_snapshots,
            "batch_size": args.batch_size,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
    }

    torch.save(
        checkpoint,
        args.output,
    )

    metadata = {
        "output": str(
            args.output.resolve()
        ),
        "fixed_data": str(
            args.fixed_data.resolve()
        ),
        "pca_rank": int(basis.shape[1]),
        "num_snapshots": int(
            deviations.shape[0]
        ),
        "num_parameters": int(
            swa_mean.numel()
        ),
        "singular_values": (
            singular_values.tolist()
        ),
        "explained_variance_ratio": (
            explained_ratio.tolist()
        ),
        "pretrain_history": pretrain_history,
        "snapshot_losses": snapshot_losses,
        "model_config": checkpoint[
            "model_config"
        ],
        "trajectory_config": checkpoint[
            "trajectory_config"
        ],
    }

    save_json(
        args.output.with_suffix(".json"),
        metadata,
    )

    print(
        "\nPCA checkpoint saved to:",
        args.output.resolve(),
        flush=True,
    )


if __name__ == "__main__":
    main()
