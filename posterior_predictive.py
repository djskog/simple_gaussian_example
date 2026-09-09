from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gaussian_transformer import GaussianTransformer
from models.subspace_model import SubspaceModel
from utils import load_data


def build_subspace_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> SubspaceModel:

    model_config = checkpoint["model_config"]
    dataset_config = checkpoint["dataset_config"]

    sigma_y = torch.as_tensor(
        dataset_config["Sigma_Y"],
        dtype=torch.float32,
    )

    model = GaussianTransformer(
        d=model_config["d"],
        Ly=sigma_y,
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
    ).to(device)

    return SubspaceModel(
        base_model=model,
        swa_mean=checkpoint["swa_mean"],
        pca_basis=checkpoint["pca_basis"],
        parameter_info=checkpoint["parameter_info"],
        device=device,
    )


# ---------------------------------------------------------------------------
# Posterior predictive means
# ---------------------------------------------------------------------------

def evaluate_posterior_means(
    subspace_model: SubspaceModel,
    phi_samples: torch.Tensor,
    source: torch.Tensor,
    device: torch.device,
    num_models: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Evaluate

        m_phi(X^0)

    for posterior samples phi.

    Parameters
    ----------
    phi_samples:
        [S, K]

    source:
        [n_test, N, d]

    num_models:
        Number of posterior draws to evaluate.
        None means use every posterior draw.

    Returns
    -------
    posterior_means:
        [S_used, n_test, d]

        posterior_means[s, i]
        is the predicted target mean for test source i under
        posterior sample phi_s.

    phi_indices:
        [S_used]

        Indices of the posterior samples used.
    """

    if phi_samples.ndim != 2:
        raise ValueError(
            "phi_samples must have shape [S, K]."
        )

    if source.ndim != 3:
        raise ValueError(
            "source must have shape [n_test, N, d]."
        )

    n_posterior = phi_samples.shape[0]

    if num_models is None:
        num_models = n_posterior

    if num_models < 1:
        raise ValueError(
            "num_models must be positive."
        )

    rng = np.random.default_rng(seed)

    replace = num_models > n_posterior

    phi_indices = rng.choice(
        n_posterior,
        size=num_models,
        replace=replace,
    )

    source = source.to(device)

    posterior_means = []

    print(
        f"Evaluating {num_models} posterior models "
        f"on {source.shape[0]} test measures...",
        flush=True,
    )

    for j, phi_index in enumerate(
        phi_indices,
        start=1,
    ):

        phi = phi_samples[
            int(phi_index)
        ].to(device)

        subspace_model.set_phi(phi)

        subspace_model.base_model.eval()

        with torch.no_grad():

            mean = subspace_model.base_model(
                source
            )

        posterior_means.append(
            mean.cpu()
        )

        if (
            j % 50 == 0
            or j == num_models
        ):
            print(
                f"Posterior models: "
                f"{j}/{num_models}",
                flush=True,
            )

    posterior_means = torch.stack(
        posterior_means,
        dim=0,
    )

    return (
        posterior_means,
        torch.as_tensor(
            phi_indices,
            dtype=torch.long,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the posterior predictive distribution of the "
            "target mean for many unseen source measures."
        )
    )

    parser.add_argument(
        "--pca",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--posterior",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help=(
            "Test-data checkpoint containing source and optionally "
            "target and latent."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--num-predictive-models",
        type=int,
        default=None,
        help=(
            "Number of posterior phi samples to evaluate. "
            "Default: all posterior samples."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    args = parser.parse_args()

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
        "Posterior predictive mean evaluation",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    print(
        "Loading PCA checkpoint...",
        flush=True,
    )

    pca_checkpoint = load_data(
        args.pca,
        required_keys={
            "swa_mean",
            "pca_basis",
            "parameter_info",
            "model_config",
            "dataset_config",
        },
    )

    print(
        "PCA checkpoint loaded.",
        flush=True,
    )

    print(
        "Loading posterior samples...",
        flush=True,
    )

    posterior = load_data(
        args.posterior,
        required_keys={
            "phi_samples",
        },
    )

    print(
        "Posterior samples loaded.",
        flush=True,
    )

    print(
        "Loading test data...",
        flush=True,
    )

    test_data = load_data(
        args.test_data,
        required_keys={
            "source",
            "latent",
            "dataset_config",
        },
    )

    print(
        "Test data loaded.",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Test data
    # -----------------------------------------------------------------------

    source = test_data[
        "source"
    ].float()
    
    latent = test_data[
            "latent"
        ].float()
    
    B0 = test_data["dataset_config"]["B0"]

    phi_samples = posterior[
        "phi_samples"
    ].float()

    print(
        f"Test source shape : "
        f"{tuple(source.shape)}",
        flush=True,
    )

    print(
        f"Phi sample shape  : "
        f"{tuple(phi_samples.shape)}",
        flush=True,
    )

    print(
        f"Device             : {device}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    print(
        "Constructing subspace model...",
        flush=True,
    )

    subspace_model = build_subspace_model(
        pca_checkpoint,
        device,
    )

    print(
        f"PCA dimension K = "
        f"{subspace_model.subspace_dim}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Evaluate posterior predictive mean
    # -----------------------------------------------------------------------

    (
        posterior_means,
        phi_indices,
    ) = evaluate_posterior_means(
        subspace_model=subspace_model,
        phi_samples=phi_samples,
        source=source,
        device=device,
        num_models=args.num_predictive_models,
        seed=args.seed,
    )

    # ---------------------------------------------------------------
    # Posterior moments of target mean
    # ---------------------------------------------------------------

    predictive_mean = posterior_means.mean(
        dim=0
    )

    predictive_std = posterior_means.std(
        dim=0
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "posterior_means": posterior_means,
        "predictive_mean": predictive_mean,
        "predictive_std": predictive_std,
        "selected_phi_indices": phi_indices,
        "source": source,
        "latent": latent,
        "B0": B0,
        "pca_checkpoint": str(
            args.pca.resolve()
        ),
        "posterior_checkpoint": str(
            args.posterior.resolve()
        ),
        "test_data": str(
            args.test_data.resolve()
        ),
        "seed": args.seed,
    }

    torch.save(
        output,
        args.output,
    )

    metadata = {
        "posterior_means_shape": list(
            posterior_means.shape
        ),
        "predictive_mean_shape": list(
            predictive_mean.shape
        ),
        "predictive_std_shape": list(
            predictive_std.shape
        ),
        "num_test_measures": int(
            source.shape[0]
        ),
        "num_posterior_models": int(
            posterior_means.shape[0]
        ),
    }

    args.output.with_suffix(
        ".json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved posterior predictive means to: "
        f"{args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()