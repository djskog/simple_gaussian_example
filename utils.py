import torch

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