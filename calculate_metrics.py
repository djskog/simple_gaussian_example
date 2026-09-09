from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_df_label, bayes_optimal_target_mean


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictive_result(
    df: float,
    checkpoint_dir: Path,
) -> dict:

    label = get_df_label(df)

    path = (
        checkpoint_dir
        / f"{label}_posterior_predictive.pt"
    )

    print(
        f"Loading {label}...",
        flush=True,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


# ---------------------------------------------------------------------------
# Energy distance
# ---------------------------------------------------------------------------

def energy_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Empirical energy distance between two multivariate samples.

    x: [n, d]
    y: [m, d]
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            "x and y must have shape [n, d]."
        )

    xx = torch.cdist(
        x,
        x,
        p=2,
    )

    yy = torch.cdist(
        y,
        y,
        p=2,
    )

    xy = torch.cdist(
        x,
        y,
        p=2,
    )

    term_xy = 2.0 * xy.mean()
    term_xx = xx.mean()
    term_yy = yy.mean()

    return (
        term_xy
        - term_xx
        - term_yy
    ).item()


# ---------------------------------------------------------------------------
# Posterior moments
# ---------------------------------------------------------------------------

def posterior_statistics(
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    samples: [S, d]

    Returns:
        mean: [d]
        covariance: [d, d]
    """

    mean = samples.mean(
        dim=0
    )

    centred = (
        samples
        - mean.unsqueeze(0)
    )

    covariance = (
        centred.T @ centred
        / (samples.shape[0] - 1)
    )

    return mean, covariance


# ---------------------------------------------------------------------------
# Empirical 95% coverage
# ---------------------------------------------------------------------------

def empirical_mahalanobis_coverage(
    samples: torch.Tensor,
    truth: torch.Tensor,
    coverage: float = 0.95,
) -> bool:
    """
    Test whether `truth` lies in the empirical Gaussian/Mahalanobis
    credible ellipsoid constructed from the posterior samples.

    Note:
        This uses the posterior mean and covariance only for defining the
        ellipsoid; no Gaussian model is assumed for the reported posterior
        samples themselves.

    samples: [S, d]
    truth:   [d]
    """

    mean, covariance = posterior_statistics(
        samples
    )

    d = samples.shape[1]

    # Small ridge for numerical stability.
    ridge = (
        1e-8
        * torch.eye(
            d,
            dtype=samples.dtype,
        )
    )

    covariance = covariance + ridge

    diff = truth - mean

    mahalanobis_sq = (
        diff
        @ torch.linalg.solve(
            covariance,
            diff,
        )
    ).item()

    # Calibrate the radius empirically from the posterior samples.
    centred = (
        samples
        - mean.unsqueeze(0)
    )

    sample_mahalanobis_sq = torch.sum(
        centred
        * torch.linalg.solve(
            covariance,
            centred.T,
        ).T,
        dim=1,
    )

    radius = torch.quantile(
        sample_mahalanobis_sq,
        coverage,
    ).item()

    return mahalanobis_sq <= radius


# ---------------------------------------------------------------------------
# Metrics for one df
# ---------------------------------------------------------------------------

def calculate_metrics_for_df(
    samples: torch.Tensor,
    baseline_samples: torch.Tensor,
    bayes_optimal: torch.Tensor,
    true_mean: torch.Tensor,
) -> dict[str, float]:

    # Posterior predictive mean
    posterior_mean = (
        samples.mean(dim=0)
    )

    # Baseline posterior predictive mean
    baseline_mean = (
        baseline_samples.mean(dim=0)
    )

    # -----------------------------------------------------------------------
    # Bias relative to Bayes-optimal prediction
    # -----------------------------------------------------------------------

    bias = torch.linalg.vector_norm(
        posterior_mean
        - bayes_optimal
    ).item()

    # -----------------------------------------------------------------------
    # Difference from Gaussian-trained posterior mean
    # -----------------------------------------------------------------------

    mean_shift = torch.linalg.vector_norm(
        posterior_mean
        - baseline_mean
    ).item()

    # -----------------------------------------------------------------------
    # Posterior uncertainty
    # -----------------------------------------------------------------------

    _, covariance = posterior_statistics(
        samples
    )

    uncertainty = (
        torch.trace(covariance)
        .item()
    )

    # -----------------------------------------------------------------------
    # Normalised bias
    # -----------------------------------------------------------------------

    normalised_bias = (
        bias
        / (
            np.sqrt(
                max(
                    uncertainty,
                    1e-12,
                )
            )
        )
    )

    # -----------------------------------------------------------------------
    # Coverage of realised true population mean
    # -----------------------------------------------------------------------

    coverage_true = empirical_mahalanobis_coverage(
        samples,
        true_mean,
        coverage=0.95,
    )

    # -----------------------------------------------------------------------
    # Energy distance from Gaussian-trained posterior
    # -----------------------------------------------------------------------

    energy = energy_distance(
        samples,
        baseline_samples,
    )

    return {
        "posterior_mean_0": posterior_mean[0].item(),
        "posterior_mean_1": posterior_mean[1].item(),

        "bias_to_bayes_optimal": bias,
        "mean_shift_from_gaussian": mean_shift,

        "posterior_uncertainty_trace": uncertainty,
        "normalised_bias": normalised_bias,

        "coverage_true_mean_95": float(
            coverage_true
        ),

        "energy_distance_from_gaussian": energy,
    }


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def calculate_df_sweep_metrics(
    *,
    checkpoint_dir: Path,
    dfs: list[float],
    n_test_measures: int | None = None,
) -> Path:

    output = (
        checkpoint_dir
        / "df_sweep_metrics.pt"
    )

    dfs = sorted(dfs)

    if not dfs:
        raise ValueError(
            "No dfs supplied."
        )

    # -----------------------------------------------------------------------
    # Load results
    # -----------------------------------------------------------------------

    predictive_results = {}

    for df in dfs:

        predictive_results[df] = (
            load_predictive_result(
                df,
                checkpoint_dir,
            )
        )

    # -----------------------------------------------------------------------
    # Identify Gaussian baseline
    # -----------------------------------------------------------------------

    gaussian_df = np.inf

    if gaussian_df not in predictive_results:
        raise ValueError(
            "The df list must contain np.inf for the Gaussian baseline."
        )

    gaussian_result = (
        predictive_results[gaussian_df]
    )

    baseline_samples = (
        gaussian_result[
            "posterior_means"
        ]
    )

    source = gaussian_result[
        "source"
    ]

    latent = gaussian_result[
        "latent"
    ]

    B0 = gaussian_result[
        "B0"
    ]

    beta = gaussian_result[
        "beta"
    ]

    Sigma_Z = gaussian_result[
        "Sigma_Z"
    ]

    Sigma_X = gaussian_result[
        "Sigma_X"
    ]

    # -----------------------------------------------------------------------
    # Test count
    # -----------------------------------------------------------------------

    available_test_measures = (
        source.shape[0]
    )

    if n_test_measures is None:
        n_test_measures = available_test_measures

    if n_test_measures > available_test_measures:
        raise ValueError(
            f"Requested {n_test_measures} test measures, but only "
            f"{available_test_measures} are available."
        )

    # -----------------------------------------------------------------------
    # Bayes-optimal predictions and realised means
    # -----------------------------------------------------------------------

    bayes_optimal = []

    true_means = []

    for i in range(
        n_test_measures
    ):

        bayes_mean = bayes_optimal_target_mean(
                X=source[i],
                beta=beta,
                Sigma_Z=Sigma_Z,
                Sigma_X=Sigma_X,
                B0=B0,
            )

        bayes_optimal.append(
            bayes_mean
        )

        true_means.append(
            B0
            @ latent[i]
        )

    bayes_optimal = torch.stack(
        bayes_optimal
    )

    true_means = torch.stack(
        true_means
    )

    # -----------------------------------------------------------------------
    # Metric arrays
    # -----------------------------------------------------------------------

    metric_names = [
        "posterior_mean_0",
        "posterior_mean_1",
        "bias_to_bayes_optimal",
        "mean_shift_from_gaussian",
        "posterior_uncertainty_trace",
        "normalised_bias",
        "coverage_true_mean_95",
        "energy_distance_from_gaussian",
    ]

    metric_values = {
        name: torch.empty(
            len(dfs),
            n_test_measures,
            dtype=torch.float64,
        )
        for name in metric_names
    }

    # -----------------------------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------------------------

    for df_idx, df in enumerate(
        dfs
    ):

        print(
            f"Calculating metrics for {get_df_label(df)}...",
            flush=True,
        )

        samples_all = (
            predictive_results[df][
                "posterior_means"
            ]
        )

        baseline_all = (
            baseline_samples
        )

        for test_idx in range(
            n_test_measures
        ):

            samples = samples_all[
                :,
                test_idx,
                :,
            ]

            baseline_samples_i = (
                baseline_all[
                    :,
                    test_idx,
                    :,
                ]
            )

            metrics = calculate_metrics_for_df(
                samples=samples,
                baseline_samples=baseline_samples_i,
                bayes_optimal=bayes_optimal[
                    test_idx
                ],
                true_mean=true_means[
                    test_idx
                ],
            )

            for name in metric_names:

                metric_values[name][
                    df_idx,
                    test_idx
                ] = metrics[name]

    # -----------------------------------------------------------------------
    # Aggregate across test measures
    # -----------------------------------------------------------------------

    aggregate = {}

    for name, values in metric_values.items():

        aggregate[name] = {
            "mean": values.mean(
                dim=1
            ),
            "std": values.std(
                dim=1
            ),
        }

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "dfs": torch.tensor(
            dfs,
            dtype=torch.float64,
        ),
        "metric_values": metric_values,
        "aggregate": aggregate,
        "bayes_optimal": bayes_optimal,
        "true_means": true_means,
        "n_test_measures": n_test_measures,
    }

    torch.save(
        result,
        output,
    )

    metadata = {
        "dfs": dfs,
        "n_test_measures": n_test_measures,
        "metric_names": metric_names,
        "output": str(
            output.resolve()
        ),
    }

    output.with_suffix(
        ".json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved metrics to: {output.resolve()}",
        flush=True,
    )

    return output


def plot_df_sweep_metrics(
    metrics_path: str | Path,
    figure_dir: str | Path,
    dpi: int = 300,
) -> None:
    """
    Plot aggregate metrics against df and save one figure per metric.

    The metrics file is expected to contain:

        "dfs"

        "aggregate" -> {
            metric_name: {
                "mean": [n_df],
                "std":  [n_df],
            }
        }

    The mean is taken over test measures and the standard deviation across
    test measures is shown as an error bar.

    A Gaussian baseline is represented by df = infinity.
    """

    metrics_path = Path(metrics_path)
    figure_dir = Path(figure_dir)

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file does not exist: {metrics_path}"
        )

    print(
        f"Loading metrics from {metrics_path}...",
        flush=True,
    )

    metrics = torch.load(
        metrics_path,
        map_location="cpu",
        weights_only=False,
    )

    dfs = metrics["dfs"].numpy()

    aggregate = metrics["aggregate"]

    # ------------------------------------------------------------------
    # Human-readable labels
    # ------------------------------------------------------------------

    def df_label(df):
        if np.isinf(df):
            return "Gaussian"
        return rf"$t_{{{df:g}}}$"

    # ------------------------------------------------------------------
    # Plot each metric
    # ------------------------------------------------------------------

    for metric_name, metric_data in aggregate.items():

        means = metric_data["mean"].numpy()
        stds = metric_data["std"].numpy()

        fig, ax = plt.subplots(
            figsize=(8, 5),
        )

        # --------------------------------------------------------------
        # Plot points with uncertainty across test measures
        # --------------------------------------------------------------

        finite_mask = np.isfinite(dfs)

        finite_dfs = dfs[finite_mask]
        finite_means = means[finite_mask]
        finite_stds = stds[finite_mask]

        gaussian_mask = np.isinf(dfs)

        # Student-t sweep
        if finite_mask.any():
            ax.errorbar(
                finite_dfs,
                finite_means,
                yerr=finite_stds,
                fmt="o-",
                capsize=4,
                label="Student-$t$",
            )

        # Gaussian baseline
        if gaussian_mask.any():
            gaussian_mean = means[gaussian_mask][0]
            gaussian_std = stds[gaussian_mask][0]

            ax.errorbar(
                [dfs[gaussian_mask][0]],
                [gaussian_mean],
                yerr=[gaussian_std],
                fmt="s",
                capsize=4,
                label="Gaussian",
            )

        # --------------------------------------------------------------
        # Formatting
        # --------------------------------------------------------------

        ax.set_xlabel(
            r"Degrees of freedom $\nu$"
        )

        ax.set_ylabel(
            metric_name.replace("_", " ")
        )

        ax.set_title(
            metric_name.replace("_", " ").title()
        )

        ax.grid(
            alpha=0.2,
        )

        ax.legend()

        # A log x-axis is useful because df has a large dynamic range.
        if finite_mask.any():
            ax.set_xscale("log")

        plt.tight_layout()

        # --------------------------------------------------------------
        # Save
        # --------------------------------------------------------------

        output_path = (
            figure_dir
            / f"metric_{metric_name}_vs_df.png"
        )

        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(
            f"Saved: {output_path}",
            flush=True,
        )

        plt.close(fig)

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Calculate and plot posterior predictive metrics "
            "across a df sweep."
        )
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dfs",
        nargs="+",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--n-test-measures",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------------

    metrics_path = calculate_df_sweep_metrics(
        checkpoint_dir=args.checkpoint_dir,
        dfs=args.dfs,
        n_test_measures=args.n_test_measures,
    )

    # ---------------------------------------------------------------
    # Plot metrics
    # ---------------------------------------------------------------

    plot_df_sweep_metrics(
        metrics_path=metrics_path,
        figure_dir=args.figure_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()