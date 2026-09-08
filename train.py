from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.gaussian_transformer import GaussianTransformer
from datasets import GaussianLocationDataset
from utils import gaussian_measure_nll



# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: GaussianTransformer,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    history: dict[str, list[float]],
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    path:
        Destination checkpoint path, typically ending in ``.pt``.
    model:
        Model to save.
    optimizer:
        Optimizer to save. Can be ``None`` when only model state is needed.
    epoch:
        Last completed epoch.
    history:
        Training/validation history.
    extra:
        Optional additional metadata/configuration.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "history": history,
        # Ly is currently a plain Tensor attribute rather than a registered
        # buffer, so store it explicitly in the checkpoint.
        "Ly": model.Ly.detach().cpu(),
        "extra": extra or {},
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: GaussianTransformer,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> tuple[GaussianTransformer, torch.optim.Optimizer | None, int, dict[str, list[float]], dict[str, Any]]:
    """
    Load a checkpoint into an existing model (and optionally optimizer).

    Returns
    -------
    model, optimizer, epoch, history, extra
        ``epoch`` is the last completed epoch stored in the checkpoint.
    """

    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore Ly explicitly because GaussianTransformer currently stores it
    # as a plain tensor attribute rather than a registered buffer.
    model.Ly = checkpoint["Ly"].to(device)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    history = checkpoint.get(
        "history",
        {"train_nll": [], "val_nll": []},
    )
    extra = checkpoint.get("extra", {})

    return model, optimizer, epoch, history, extra


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_m2m(
    model: GaussianTransformer,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate average negative log-likelihood per measure.

    ``gaussian_measure_nll`` returns one scalar already averaged across the
    measures in the batch, after summing over target points. We therefore
    aggregate by the number of measures in each batch.
    """

    model.eval()

    total_nll = 0.0
    total_points = 0

    for batch in dataloader:
        source = batch[0].to(device)
        target = batch[1].to(device)

        batch_nll = gaussian_measure_nll(
            y=target,
            mean=model(source),
            Ly=model.Ly,
        )

        batch_size = target.shape[0]

        total_nll += batch_nll.item() * batch_size
        total_points += batch_size

    return total_nll / max(total_points, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_m2m(
    train_dataset,
    model,
    val_dataset=None,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    grad_clip: float | None = 10.0,
    print_every: int = 1,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
):
    """
    Train the Gaussian M2M transformer by maximum likelihood.

    Parameters
    ----------
    train_dataset:
        Training dataset returning ``(source, target, Z)``.
    model:
        GaussianTransformer instance.
    val_dataset:
        Optional validation dataset.
    checkpoint_path:
        Optional path used for periodic checkpoints.
    checkpoint_every:
        Save every this many epochs. If ``None``, checkpoints are only saved
        at the end of training when ``checkpoint_path`` is supplied.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = model.to(device)

    # GaussianTransformer currently keeps Ly as a normal Tensor attribute,
    # so make sure it is on the same device as the model inputs.
    model.Ly = model.Ly.to(device)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_nll": [],
        "val_nll": [],
    }

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    for epoch in range(1, num_epochs + 1):

        model.train()

        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:

            # Both current datasets return their source and target first:
            #
            # GaussianLocationDataset: X, Y, Z
            # ShapeDataset:            source, target
            source = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            loss = gaussian_measure_nll(
                y=target,
                mean=model(source),
                Ly=model.Ly,
            )

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        train_nll = running_loss / max(num_batches, 1)
        history["train_nll"].append(train_nll)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------

        if val_loader is not None:
            val_nll = evaluate_m2m(
                model,
                val_loader,
                device,
            )
            history["val_nll"].append(val_nll)
        else:
            val_nll = None

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------

        if epoch % print_every == 0:
            if val_nll is None:
                print(
                    f"Epoch {epoch:03d}/{num_epochs:03d} "
                    f"| train NLL = {train_nll:.5f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d}/{num_epochs:03d} "
                    f"| train NLL = {train_nll:.5f} "
                    f"| val NLL = {val_nll:.5f}"
                )

        # -------------------------------------------------------------------
        # Periodic checkpoint
        # -------------------------------------------------------------------

        if (
            checkpoint_path is not None
            and checkpoint_every is not None
            and epoch % checkpoint_every == 0
        ):
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                extra={
                    "device": str(device),
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
            )

    # -----------------------------------------------------------------------
    # Final checkpoint
    # -----------------------------------------------------------------------

    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=num_epochs,
            history=history,
            extra={
                "device": str(device),
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    return model, history


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Train the Gaussian-location M2M transformer on a simple 2D example."""

    # Reproducibility for model initialization/optimizer randomness.
    torch.manual_seed(0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -----------------------------------------------------------------------
    # Data-generating parameters
    # -----------------------------------------------------------------------

    d = 2
    num_points = 128

    beta = torch.zeros(d)

    Sigma_Z = torch.tensor([
        [1.0, 0.4],
        [0.4, 1.0],
    ])

    Sigma_X = 0.25 * torch.eye(d)
    Sigma_Y = 0.10 * torch.eye(d)

    # Nontrivial linear map from latent source location to target location.
    B0 = torch.tensor([
        [1.2, 0.2],
        [-0.1, 0.8],
    ])

    Ly = torch.linalg.cholesky(Sigma_Y)

    # -----------------------------------------------------------------------
    # Train/validation datasets
    # -----------------------------------------------------------------------

    train_dataset = GaussianLocationDataset(
        num_measures=1000,
        num_samples=num_points,
        beta=beta,
        Sigma_Z=Sigma_Z,
        Sigma_X=Sigma_X,
        Sigma_Y=Sigma_Y,
        B0=B0,
        seed=123,
    )

    val_dataset = GaussianLocationDataset(
        num_measures=200,
        num_samples=num_points,
        beta=beta,
        Sigma_Z=Sigma_Z,
        Sigma_X=Sigma_X,
        Sigma_Y=Sigma_Y,
        B0=B0,
        seed=456,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = GaussianTransformer(
        d=d,
        Ly=Ly,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        dropout=0.0,
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    checkpoint_path = PROJECT_ROOT / "checkpoints" / "gaussian_transformer.pt"

    model, history = train_m2m(
        train_dataset=train_dataset,
        model=model,
        val_dataset=val_dataset,
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-3,
        weight_decay=0.0,
        device=device,
        num_workers=0,
        grad_clip=10.0,
        print_every=1,
        checkpoint_path=checkpoint_path,
        checkpoint_every=10,
    )

    print("Training complete.")
    print(f"Best/last validation NLL: {history['val_nll'][-1]:.5f}")


if __name__ == "__main__":
    main()
