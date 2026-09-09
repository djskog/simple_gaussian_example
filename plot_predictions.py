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
    
from utils import get_df_label


def load_predictive_result(
    df: float,
    checkpoint_dir: Path) -> dict:
    
    label = get_df_label(df)
    
    path = (
        checkpoint_dir
        / f"{label}_posterior_predictive.pt"
    )

    print(f"  Loading {label}...", flush=True)
    
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

def get_test_data(
    predictive_result: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    if "source" not in predictive_result:
        raise KeyError(
            "Predictive checkpoint does not contain 'source'."
        )

    if "latent" not in predictive_result:
        raise KeyError(
            "Predictive checkpoint does not contain 'latent'."
        )
        
    if "B0" not in predictive_result:
            raise KeyError(
                "Predictive checkpoint does not contain 'B0'."
            )
    
    return predictive_result["source"], predictive_result["latent"], predictive_result["B0"]

def get_test_and_compile_prediction_data(
    checkpoint_dir: Path,
    dfs: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    
    n_dfs = len(dfs)
    predictive_results = [None] * n_dfs
    
    print("Loading posterior predictive results...", flush=True)
    
    # ------------------------------------------------------------------
    # Test data are shared across all training-data distributions.
    # ------------------------------------------------------------------
    predictive_result = load_predictive_result(dfs[n_dfs - 1], checkpoint_dir)
    source, latent, B0 = get_test_data(predictive_result)

    #compile predictions for all dfs
    predictive_results[0] = predictive_result["posterior_means"]
    for i, df in enumerate(dfs[:-1]):
        predictive_result = load_predictive_result(df, checkpoint_dir)
        predictive_results[i] = predictive_result["posterior_means"]


    return source, latent, B0, predictive_results



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

    Each figure has one panel per df, with common axis limits across all
    panels in that figure. The heatmap is the empirical 2D histogram of
    posterior predictive target-mean samples.
    """
    dfs = sorted(dfs)
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source, latent, B0, predictive_results = get_test_and_compile_prediction_data(
        checkpoint_dir=checkpoint_dir,
        dfs=dfs,
    )

    if n_plots > source.shape[0]:
        raise ValueError(
            f"n_plots={n_plots}, but only "
            f"{source.shape[0]} test measures are available."
        )

    # ------------------------------------------------------------------
    # Plot one figure for each test measure.
    # ------------------------------------------------------------------

    for measure_idx in range(n_plots):

        true_mean = (
            B0 @ latent[measure_idx]
        ).numpy()

        # Collect all samples so all df panels share the same axis scale.
        all_samples = []

        for i, df in enumerate(dfs):
            samples = (
                predictive_results[i][:, measure_idx, :]
                .numpy()
            )

            if samples.shape[1] != 2:
                raise ValueError(
                    "This plotting script expects two-dimensional target means."
                )

            all_samples.append(samples)

        all_samples = np.concatenate(
            all_samples,
            axis=0,
        )

        x = all_samples[:, 0]
        y = all_samples[:, 1]

        x_low, x_high = np.quantile(
            x,
            [0.01, 0.99],
        )

        y_low, y_high = np.quantile(
            y,
            [0.01, 0.99],
        )

        # Ensure the true population mean is visible.
        x_low = min(x_low, true_mean[0])
        x_high = max(x_high, true_mean[0])
        y_low = min(y_low, true_mean[1])
        y_high = max(y_high, true_mean[1])

        x_range = x_high - x_low
        y_range = y_high - y_low

        x_pad = max(0.1 * x_range, 1e-4)
        y_pad = max(0.1 * y_range, 1e-4)

        x_min = x_low - x_pad
        x_max = x_high + x_pad
        y_min = y_low - y_pad
        y_max = y_high + y_pad

        plot_range = [
            [x_min, x_max],
            [y_min, y_max],
        ]

        # --------------------------------------------------------------
        # Figure: Gaussian + each Student-t df.
        # --------------------------------------------------------------

        fig, axes = plt.subplots(
            1,
            len(dfs),
            figsize=(4 * len(dfs), 4.5),
            squeeze=False,
        )

        axes = axes[0]

        for ax, df in zip(axes, dfs):

            samples = (
                predictive_results[i][:, measure_idx, :]
                .numpy()
            )

            posterior_mean = samples.mean(
                axis=0
            )

            counts, x_edges, y_edges = (
                np.histogram2d(
                    samples[:, 0],
                    samples[:, 1],
                    bins=bins,
                    range=plot_range,
                )
            )

            probabilities = (
                counts / counts.sum()
                if counts.sum() > 0
                else counts
            )

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

            ax.scatter(
                posterior_mean[0],
                posterior_mean[1],
                s=100,
                marker="x",
                linewidths=3,
                color="red",
                zorder=10,
            )

            ax.scatter(
                true_mean[0],
                true_mean[1],
                s=160,
                marker="+",
                linewidths=3,
                color="white",
                zorder=11,
            )

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

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.grid(alpha=0.15)

        # Shared legend.
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

        fig.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
        )

        fig.suptitle(
            rf"Posterior predictive distribution of $m_\phi(X^0)$"
            f" — test measure {measure_idx}",
            fontsize=16,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.91]
        )

        output_path = (
            figure_dir
            / f"posterior_predictive_df_sweep_measure_{measure_idx}.png"
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
            "Plot posterior predictive target-mean heatmaps for a "
            "Gaussian / Student-t df sweep."
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
    
    dfs = [float(df) for df in args.dfs]

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
