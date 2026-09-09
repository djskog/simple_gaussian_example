from __future__ import annotations

import argparse
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
# Load one posterior predictive result
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
        f"  Loading {label}...",
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
# Load and compile all predictions
# ---------------------------------------------------------------------------

def compile_prediction_data(
    checkpoint_dir: Path,
    dfs: list[float],
    n_plots: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict,
    torch.Tensor
]:
    """
    Load all posterior predictive results and organise them by test measure.

    Returns
    -------
    source:
        [n_plots, n_points, d]

    true_means:
        [n_plots, d]

    posterior_means:
        [n_plots, n_df, S, d]

        posterior_means[i, j, s, :]
        = prediction for test measure i,
          training-data df dfs[j],
          posterior sample s.
    """

    results = []

    source = None
    latent = None
    dataset_config = None

    print(
        "Loading posterior predictive results...",
        flush=True,
    )

    for df in dfs:

        result = load_predictive_result(
            df,
            checkpoint_dir,
        )

        # ---------------------------------------------------------------
        # Common test data
        # ---------------------------------------------------------------

        if dataset_config is None:

            source = result["source"][:n_plots]

            latent = result["latent"][:n_plots]

            dataset_config = result["dataset_config"]

        # ---------------------------------------------------------------
        # IMPORTANT:
        #
        # posterior_means has shape
        #
        #     [S, n_test, d]
        #
        # so slice the SECOND dimension to select test measures.
        # ---------------------------------------------------------------

        posterior_means = result["posterior_means"][:, :n_plots, :]

        results.append(
            posterior_means
        )

    if source is None:
        raise ValueError(
            "No dfs were supplied."
        )

    # Each result:
    #
    #     [S, n_plots, d]
    #
    # Stack:
    #
    #     [n_df, S, n_plots, d]
    #

    posterior_means = torch.stack(
        results,
        dim=0,
    )

    # Reorder:
    #
    #     [n_plots, n_df, S, d]
    #

    posterior_means = posterior_means.permute(
        2,
        0,
        1,
        3,
    )


    return (
        source,
        latent,
        dataset_config,
        posterior_means,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictive_df_sweep(
    checkpoint_dir: Path,
    figure_dir: Path,
    dfs: list[float],
    n_plots: int = 4,
    bins: int = 40,
    dpi: int = 300,
) -> None:
    """
    Create one figure per test measure.

    Each figure has one panel per df.

    The panels within a figure share the same axis limits so that the
    posterior predictive distributions can be compared directly.
    """

    dfs = sorted(dfs)

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source, latent, dataset_config, posterior_means = compile_prediction_data(
        checkpoint_dir=checkpoint_dir,
        dfs=dfs,
        n_plots=n_plots,
    )

    true_means = latent @ dataset_config["B0"].T
    bayes_means = torch.stack([
        bayes_optimal_target_mean(
            X=source[i],
            beta=dataset_config["beta"],
            Sigma_Z=dataset_config["Sigma_Z"],
            Sigma_X=dataset_config["Sigma_X"],
            B0=dataset_config["B0"],
        )
        for i in range(n_plots)
    ])
    # -----------------------------------------------------------------------
    # Plot one figure for each test measure
    # -----------------------------------------------------------------------

    for measure_idx in range(n_plots):

        true_mean = true_means[measure_idx].numpy()
    
        
        bayes_mean = bayes_means[measure_idx].numpy()

        # ================================================================
        # Collect samples from all dfs
        #
        # posterior_means has shape:
        #
        #     [n_test, n_df, S, d]
        #
        # ================================================================

        all_samples = []

        for df_idx in range(
            len(dfs)
        ):

            samples = (
                posterior_means[
                    measure_idx,
                    df_idx,
                ]
                .numpy()
            )

            if samples.shape[1] != 2:
                raise ValueError(
                    "This plotting script expects "
                    "two-dimensional target means."
                )

            all_samples.append(
                samples
            )

        all_samples = np.concatenate(
            all_samples,
            axis=0,
        )

        x = all_samples[:, 0]
        y = all_samples[:, 1]

        # ================================================================
        # Common plotting limits for all dfs
        # ================================================================

        x_low, x_high = np.quantile(
            x,
            [0.01, 0.99],
        )

        y_low, y_high = np.quantile(
            y,
            [0.01, 0.99],
        )

        # Make sure the true mean is visible.
        x_low = min(
            x_low,
            true_mean[0],
            bayes_mean[0],
        )

        x_high = max(
            x_high,
            true_mean[0],
            bayes_mean[0],
        )

        y_low = min(
            y_low,
            true_mean[1],
            bayes_mean[1],
        )

        y_high = max(
            y_high,
            true_mean[1],
            bayes_mean[1],
        )

        x_range = x_high - x_low
        y_range = y_high - y_low

        x_pad = max(
            0.1 * x_range,
            1e-4,
        )

        y_pad = max(
            0.1 * y_range,
            1e-4,
        )

        x_min = x_low - x_pad
        x_max = x_high + x_pad

        y_min = y_low - y_pad
        y_max = y_high + y_pad

        plot_range = [
            [x_min, x_max],
            [y_min, y_max],
        ]

        # ================================================================
        # Figure
        # ================================================================

        fig, axes = plt.subplots(
            1,
            len(dfs),
            figsize=(
                4 * len(dfs),
                4.5,
            ),
            squeeze=False,
        )

        axes = axes[0]

        # ================================================================
        # One panel per df
        # ================================================================

        for df_idx, (
            ax,
            df,
        ) in enumerate(
            zip(
                axes,
                dfs,
            )
        ):

            samples = (
                posterior_means[
                    measure_idx,
                    df_idx,
                ]
                .numpy()
            )

            posterior_mean = (
                samples.mean(axis=0)
            )

            # ------------------------------------------------------------
            # Empirical 2D histogram
            # ------------------------------------------------------------

            counts, x_edges, y_edges = (
                np.histogram2d(
                    samples[:, 0],
                    samples[:, 1],
                    bins=bins,
                    range=plot_range,
                )
            )

            if counts.sum() > 0:

                probabilities = (
                    counts / counts.sum()
                )

            else:

                probabilities = counts

            ax.imshow(
                probabilities.T,
                origin="lower",
                extent=[
                    x_edges[0],
                    x_edges[-1],
                    y_edges[0],
                    y_edges[-1],
                ],
                aspect="equal",
                interpolation="nearest",
                cmap="viridis",
            )

            # ------------------------------------------------------------
            # Transformer posterior mean
            # ------------------------------------------------------------

            ax.scatter(
                posterior_mean[0],
                posterior_mean[1],
                s=100,
                marker="x",
                linewidths=3,
                color="red",
                zorder=10,
            )

            # ------------------------------------------------------------
            # True population mean
            # ------------------------------------------------------------

            ax.scatter(
                true_mean[0],
                true_mean[1],
                s=160,
                marker="+",
                linewidths=3,
                color="white",
                zorder=11,
            )
            
            ax.scatter(
                bayes_mean[0],
                bayes_mean[1],
                s=140,
                marker="*",
                color="orange",
                edgecolor="black",
                linewidth=0.7,
                zorder=12,
            )

            # ------------------------------------------------------------
            # Title
            # ------------------------------------------------------------

            title = (
                "Gaussian"
                if np.isinf(df)
                else rf"$t_{{{df:g}}}$"
            )

            ax.set_title(
                title,
                fontsize=13,
            )

            ax.set_xlabel(
                "Target mean coordinate 1"
            )

            ax.set_ylabel(
                "Target mean coordinate 2"
            )

            ax.set_xlim(
                x_min,
                x_max,
            )

            ax.set_ylim(
                y_min,
                y_max,
            )

            ax.grid(
                alpha=0.15,
            )

        # ================================================================
        # Common legend
        # ================================================================

        axes[0].scatter(
            [],
            [],
            s=100,
            marker="x",
            linewidths=3,
            color="red",
            label="Transformer posterior mean",
        )

        axes[0].scatter(
            [],
            [],
            s=160,
            marker="+",
            linewidths=3,
            color="white",
            label=r"True $B_0 Z^0$",
        )
        
        axes[0].scatter(
            [],
            [],
            s=140,
            marker="*",
            color="orange",
            edgecolor="black",
            linewidth=0.7,
            label=r"Bayes optimal $E[Y^0\mid X^0]$",
        )

        fig.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(
                0.5,
                1.02,
            ),
        )

        # ================================================================
        # Overall title
        # ================================================================

        fig.suptitle(
            rf"Posterior predictive distribution of $m_\phi(X^0)$"
            f" — test measure {measure_idx}",
            fontsize=16,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.91]
        )

        # ================================================================
        # Save
        # ================================================================

        output_path = (
            figure_dir
            / (
                "posterior_predictive_"
                f"df_sweep_measure_{measure_idx}.png"
            )
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Plot posterior predictive target-mean heatmaps "
            "for a Gaussian / Student-t df sweep."
        )
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "figures",
    )

    parser.add_argument(
        "--dfs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--n-plots",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    dfs = [
        float(df)
        for df in args.dfs
    ]

    plot_predictive_df_sweep(
        checkpoint_dir=args.checkpoint_dir,
        figure_dir=args.figure_dir,
        dfs=dfs,
        n_plots=args.n_plots,
        bins=args.bins,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()