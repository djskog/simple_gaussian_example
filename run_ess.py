from __future__ import annotations

import argparse
import json
import random
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
from utils import gaussian_measure_nll


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fixed_data(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)

    required = {"source", "target", "dataset_config"}
    missing = required.difference(data.keys())
    if missing:
        raise KeyError(
            "Fixed-data file is missing: "
            + ", ".join(sorted(missing))
        )
    return data


def build_subspace_model(
    pca_checkpoint: dict[str, Any],
    device: torch.device,
) -> SubspaceModel:
    model_config = pca_checkpoint["model_config"]
    dataset_config = pca_checkpoint["dataset_config"]

    sigma_y = torch.as_tensor(
        dataset_config["Sigma_Y"],
        dtype=torch.float32,
        device = device,
    )

    base_model = GaussianTransformer(
        d=model_config["d"],
        Ly=sigma_y,
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
    ).to(device)

    return SubspaceModel(
        base_model=base_model,
        swa_mean=pca_checkpoint["swa_mean"],
        pca_basis=pca_checkpoint["pca_basis"],
        parameter_info=pca_checkpoint["parameter_info"],
        device=device,
    )


def make_fixed_batches(
    source: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
):
    n = source.shape[0]
    for start in range(0, n, batch_size):
        yield source[start:start + batch_size], target[start:start + batch_size]


def make_log_likelihood(
    subspace_model: SubspaceModel,
    source: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    batch_size: int,
):
    """
    Return log p(D | phi), where
        w(phi) = w_hat + P phi.
    """

    sigma_y_chol = subspace_model.base_model.Ly

    def log_likelihood(phi: torch.Tensor) -> float:
        subspace_model.set_phi(phi)
        subspace_model.base_model.eval()

        total_log_likelihood = 0.0

        with torch.no_grad():
            for source_batch, target_batch in make_fixed_batches(
                source,
                target,
                batch_size,
            ):
                source_batch = source_batch.to(device)
                target_batch = target_batch.to(device)

                mean = subspace_model.base_model(source_batch)

                # gaussian_measure_nll returns an average over measures
                # of the sum over points, so multiply back by batch size.
                nll = gaussian_measure_nll(
                    y=target_batch,
                    mean=mean,
                    Ly=sigma_y_chol,
                )

                total_log_likelihood -= nll.item() * source_batch.shape[0]

        return total_log_likelihood

    return log_likelihood


def elliptical_slice_step(
    phi: torch.Tensor,
    current_log_likelihood: float,
    log_likelihood,
    temperature: float,
    prior_std: float,
    max_bracket_steps: int = 10000,
):
    """
    Elliptical slice sampling for

        phi ~ N(0, prior_std^2 I).

    The Gaussian prior is incorporated directly into the elliptical
    proposal, so the slice threshold contains only the likelihood.
    """

    if prior_std <= 0:
        raise ValueError(
            "prior_std must be positive."
        )

    # Draw from the Gaussian prior:
    #
    # nu ~ N(0, prior_std^2 I)
    #
    nu = (
        prior_std
        * torch.randn_like(phi)
    )

    log_slice = (
        current_log_likelihood / temperature
        + torch.log(
            torch.rand(
                (),
                device=phi.device,
            )
        ).item()
    )

    theta = (
        2.0
        * np.pi
        * torch.rand(
            (),
            device=phi.device,
        ).item()
    )

    theta_min = theta - 2.0 * np.pi
    theta_max = theta

    for _ in range(max_bracket_steps):

        theta_tensor = torch.tensor(
            theta,
            dtype=phi.dtype,
            device=phi.device,
        )

        proposal = (
            phi * torch.cos(theta_tensor)
            + nu * torch.sin(theta_tensor)
        )

        proposal_ll = log_likelihood(
            proposal
        )

        if (
            proposal_ll / temperature
            >= log_slice
        ):
            return (
                proposal,
                proposal_ll,
            )

        if theta < 0.0:
            theta_min = theta
        else:
            theta_max = theta

        theta = (
            theta_min
            + random.random()
            * (theta_max - theta_min)
        )

    raise RuntimeError(
        "ESS exceeded max_bracket_steps."
    )


def run_ess(
    log_likelihood,
    dimension: int,
    device: torch.device,
    prior_std: float,
    burn_in: int,
    num_samples: int,
    thin: int,
    temperature: float,
    seed: int,
):
    set_seed(seed)

    phi = torch.zeros(
        dimension,
        dtype=torch.float32,
        device=device,
    )

    current_ll = log_likelihood(phi)

    saved = []
    log_likelihoods = []

    total_iterations = burn_in + num_samples * thin

    print(
        f"Starting ESS burn-in: {burn_in} iterations",
        flush=True,
    )
    
    for iteration in range(1, total_iterations + 1):

        phi, current_ll = elliptical_slice_step(
            phi=phi,
            current_log_likelihood=current_ll,
            log_likelihood=log_likelihood,
            temperature=temperature,
            prior_std=prior_std,
        )

        # -------------------------------------------------------------
        # Burn-in completed
        # -------------------------------------------------------------

        if iteration == burn_in:
            print(
                "\n"
                "============================================================",
                flush=True,
            )
            print("Burn-in complete", flush=True)
            print(
                f"Completed {burn_in} burn-in iterations.",
                flush=True
            )
            print(
                f"Current log likelihood = {current_ll:.4f}",
                flush=True,
            )
            print(
                f"Current ||phi|| = "
                f"{torch.linalg.vector_norm(phi).item():.4f}",
                flush=True,
            )
            print(
                "Starting posterior sampling...",
                flush=True,
            )
            print(
                "============================================================"
                "\n",
                flush=True,
            )

        # -------------------------------------------------------------
        # Save posterior samples
        # -------------------------------------------------------------

        if (
            iteration > burn_in
            and (iteration - burn_in) % thin == 0
        ):
            saved.append(
                phi.detach().cpu().clone()
            )

            log_likelihoods.append(
                current_ll
            )

            num_saved = len(saved)

            # Print every 50 posterior samples.
            if (
                num_saved % 50 == 0
                or num_saved == num_samples
            ):
                print(
                    f"Posterior samples: "
                    f"{num_saved}/{num_samples}"
                    f" | log L = {current_ll:.4f}"
                    f" | ||phi|| = "
                    f"{torch.linalg.vector_norm(phi).item():.4f}",
                    flush=True,
                )

    print(
        "\n"
        "============================================================",
        flush=True,
    )
    print("ESS complete", flush=True)
    print(
        f"Collected {len(saved)} posterior samples.",
        flush=True,
    )
    print(
        "============================================================",
        flush=True
    )

    return (
        torch.stack(saved),
        torch.tensor(
            log_likelihoods,
            dtype=torch.float64,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elliptical slice sampling in the PCA subspace."
    )

    parser.add_argument(
        "--fixed-data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--pca",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "posterior_phi.pt",
    )
    
    parser.add_argument(
        "--prior-std",
        type=float,
        default=1.0,
        help=(
            "Standard deviation of the Gaussian PCA prior: "
            "phi ~ N(0, prior_std^2 I)."
        ),
    )

    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    
    print("Hi", flush=True)

    print("Starting run_ess.py", flush=True)

    if args.device is None:
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(args.device)

    print(
        f"Device: {device}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Load fixed data
    # -------------------------------------------------------------

    print(
        "Loading fixed dataset...",
        flush=True,
    )

    fixed_data = load_fixed_data(
        args.fixed_data
    )

    print(
        "Fixed dataset loaded.",
        flush=True,
    )

    # -------------------------------------------------------------
    # Load PCA checkpoint
    # -------------------------------------------------------------

    print(
        "Loading PCA checkpoint...",
        flush=True,
    )

    pca_checkpoint = torch.load(
        args.pca,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "PCA checkpoint loaded.",
        flush=True,
    )

    # -------------------------------------------------------------
    # Load tensors
    # -------------------------------------------------------------

    source = fixed_data["source"].float()
    target = fixed_data["target"].float()

    print(
        f"Source shape: {tuple(source.shape)}",
        flush=True,
    )

    print(
        f"Target shape: {tuple(target.shape)}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Construct subspace model
    # -------------------------------------------------------------

    print(
        "Constructing subspace model...",
        flush=True,
    )

    subspace_model = build_subspace_model(
        pca_checkpoint,
        device,
    )

    print(
        "Subspace model constructed.",
        flush=True,
    )

    K = subspace_model.subspace_dim

    print(
        f"PCA dimension K = {K}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Construct likelihood
    # -------------------------------------------------------------

    print(
        "Constructing fixed-data likelihood...",
        flush=True,
    )

    log_likelihood = make_log_likelihood(
        subspace_model=subspace_model,
        source=source,
        target=target,
        device=device,
        batch_size=args.batch_size,
    )

    print(
        "Likelihood constructed.",
        flush=True,
    )

    # -------------------------------------------------------------
    # ESS configuration
    # -------------------------------------------------------------

    print(
        "============================================================",
        flush=True,
    )

    print(
        "Elliptical slice sampling",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        f"Fixed data          : {args.fixed_data}",
        flush=True,
    )

    print(
        f"PCA checkpoint      : {args.pca}",
        flush=True,
    )

    print(
        f"PCA dimension K     : {K}",
        flush=True,
    )

    print(
        f"Prior std           : {args.prior_std}",
        flush=True,
    )
        
    print(
        f"Temperature         : {args.temperature}",
        flush=True,
    )

    print(
        f"Burn-in             : {args.burn_in}",
        flush=True,
    )

    print(
        f"Posterior samples   : {args.num_samples}",
        flush=True,
    )

    print(
        f"Thinning            : {args.thin}",
        flush=True,
    )

    print(
        f"Device              : {device}",
        flush=True,
    )

    # -------------------------------------------------------------
    # Run ESS
    # -------------------------------------------------------------

    print(
        "Starting ESS...",
        flush=True,
    )

    phi_samples, sample_ll = run_ess(
        log_likelihood=log_likelihood,
        dimension=K,
        device=device,
        prior_std=args.prior_std,
        burn_in=args.burn_in,
        num_samples=args.num_samples,
        thin=args.thin,
        temperature=args.temperature,
        seed=args.seed,
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    print(
        "ESS finished. Saving posterior samples...",
        flush=True,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "phi_samples": phi_samples,
        "temperature": args.temperature,
        "prior_std": args.prior_std,
        "burn_in": args.burn_in,
        "thin": args.thin,
        "num_samples": args.num_samples,
        "seed": args.seed,
    }

    torch.save(
        output,
        args.output,
    )

    print(
        f"Saved posterior samples: "
        f"{args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
