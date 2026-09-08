import torch
import numpy as np
from pathlib import Path
from typing import Any

def _sample_multivariate_t(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    df: float,
    num_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample from a multivariate Student-t distribution whose covariance is
    `covariance`.

    For df > 2,

        Cov(T) = df / (df - 2) * scale,

    so the Student-t scale matrix is

        scale = (df - 2) / df * covariance.
    """

    if df <= 2:
        raise ValueError(
            "df must be greater than 2 when covariance is specified."
        )

    d = mean.numel()

    scale = (
        (df - 2.0) / df
    ) * covariance

    L = torch.linalg.cholesky(scale)

    gaussian = torch.randn(
        num_samples,
        d,
        generator=generator,
        dtype=torch.float32,
    )

    # Gamma(shape=df/2, rate=1/2) = chi-square(df) / 2.
    gamma = torch.distributions.Gamma(
        torch.tensor(df / 2.0),
        torch.tensor(0.5),
    )

    chi2 = (
        2.0
        * gamma.sample((num_samples,))
    )

    return (
        mean.unsqueeze(0)
        + gaussian @ L.T
        / torch.sqrt(
            chi2.unsqueeze(1) / df
        )
    )

def gaussian_measure_nll(
    y,
    mean,
    Ly,
):
    """
    y:       [batch, N, d]
    mean:    [batch, d]
    Ly: [d, d]

    Returns:
        mean NLL per measure
    """

    d = y.shape[-1]

    residual = y - mean.unsqueeze(1)

    # Solve Ly z = residual^T
    whitened = torch.linalg.solve_triangular(
        Ly,
        residual.reshape(-1, d).T,
        upper=False,
    ).T

    quadratic = whitened.square().sum(dim=-1)

    logdet = 2.0 * torch.log(
        torch.diagonal(Ly)
    ).sum()

    constant = d * torch.log(
        torch.tensor(2.0 * torch.pi, device=y.device)
    )

    pointwise_nll = 0.5 * (
        quadratic
        + logdet
        + constant
    )

    # average over points, then measures
    return pointwise_nll.reshape(
        y.shape[0],
        y.shape[1],
    ).sum(dim=1).mean()
    
    
def bayes_optimal_target_mean(
    X,
    beta,
    Sigma_Z,
    Sigma_X,
    B0,
):
    """
    Exact E[Y | X] for the Gaussian location model.

    X: [N, d]

    Returns:
        [d]
    """

    N = X.shape[0]

    precision_Z = torch.linalg.inv(Sigma_Z)
    precision_X = torch.linalg.inv(Sigma_X)

    posterior_cov_Z = torch.linalg.inv(
        precision_Z
        + N * precision_X
    )

    posterior_mean_Z = (
        posterior_cov_Z
        @ (
            precision_Z @ beta
            + precision_X @ X.sum(dim=0)
        )
    )

    return B0 @ posterior_mean_Z

def get_df_label(df: float) -> str:
    
    if np.isinf(df):
        df_label = "gaussian"
    else:
        df_label = f"t_df{df:g}"
    
    return df_label

def load_data(
    path: str | Path,
    required_keys: set[str] | list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """
    Load selected entries from a saved .pt dictionary.

    Parameters
    ----------
    path:
        Path to the saved .pt file.

    required_keys:
        Keys to retrieve from the file.

    Returns
    -------
    dict
        Dictionary containing only the requested keys.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected {path} to contain a dictionary, "
            f"got {type(data).__name__}."
        )

    required_keys = list(required_keys)

    missing = set(required_keys).difference(
        data.keys()
    )

    if missing:
        raise KeyError(
            f"File {path} is missing required keys: "
            + ", ".join(sorted(missing))
        )

    return {
        key: data[key]
        for key in required_keys
    }