import numpy as np
import torch
from torch.utils.data import Dataset
from utils import _sample_multivariate_t


# ---------------------------------------------------------------------------
# Shape data
# ---------------------------------------------------------------------------

def sample_circle(
    num_points: int,
    radius_range=(1.0, 1.0),
    shift_range=(-1.0, 1.0),
):
    """
    Sample points approximately uniformly around a circle.

    Returns
    -------
    points : torch.Tensor
        Shape [num_points, 2].
    """
    r = np.random.uniform(*radius_range)

    theta = np.random.uniform(
        0.0,
        2.0 * np.pi,
        num_points,
    )

    x_shift = np.random.uniform(*shift_range)
    y_shift = np.random.uniform(*shift_range)

    x = x_shift + r * np.cos(theta)
    y = y_shift + r * np.sin(theta)

    return torch.tensor(
        np.column_stack((x, y)),
        dtype=torch.float32,
    )


def sample_square(
    num_points: int,
    side_range=(2.0, 2.0),
    shift_range=(-1.0, 1.0),
):
    """
    Sample points uniformly from the boundary of a square.

    Returns
    -------
    points : torch.Tensor
        Shape [num_points, 2].
    """
    side_length = np.random.uniform(*side_range)
    h = side_length / 2.0

    x_shift = np.random.uniform(*shift_range)
    y_shift = np.random.uniform(*shift_range)

    coord = np.random.uniform(
        -h,
        h,
        num_points,
    )

    edge = np.random.randint(
        0,
        2,
        size=num_points,
    )

    sign = (
        2 * np.random.randint(
            0,
            2,
            size=num_points,
        )
        - 1
    )

    x = (
        x_shift
        + edge * coord
        + (1 - edge) * sign * h
    )

    y = (
        y_shift
        + (1 - edge) * coord
        + edge * sign * h
    )

    return torch.tensor(
        np.column_stack((x, y)),
        dtype=torch.float32,
    )


def kernel_interaction_points(
    points: torch.Tensor,
    eta: float = 0.3,
    sigma: float = 0.01,
    h: float = 0.75,
    dt: float = 0.05,
    num_steps: int = 50,
):
    """
    Kernel interaction process.

    Parameters
    ----------
    points:
        [N, 2] point cloud.

    eta:
        Repulsion strength.

    sigma:
        Noise scale.

    h:
        Kernel bandwidth.

    dt:
        Time step.

    num_steps:
        Number of interaction steps.

    Returns
    -------
    torch.Tensor
        Transformed point cloud of shape [N, 2].
    """
    x = points.clone()

    sqrt_dt = np.sqrt(dt)

    for _ in range(num_steps):
        d2 = torch.cdist(
            x,
            x,
            p=2,
        ).pow(2)

        A = torch.exp(
            -d2 / (2.0 * h * h)
        )

        A = A / (
            A.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-12
        )

        local_mean = A @ x
        drift = eta * (x - local_mean) * dt

        noise = (
            sigma
            * sqrt_dt
            * torch.randn_like(x)
        )

        x = x + drift + noise

    return x


class ShapeDataset(Dataset):
    """
    Synthetic shape-to-shape M2M dataset.

    Each example consists of:
        source = corrupted target shape
        target = clean target shape
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_points: int = 128,
        sigma: float = 0.05,
        seed: int | None = None,
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_samples

    def _sample_circle(
        self,
        num_points: int,
    ):
        r = self.rng.uniform(1.0, 1.0)

        theta = self.rng.uniform(
            0.0,
            2.0 * np.pi,
            num_points,
        )

        x_shift = self.rng.uniform(-1.0, 1.0)
        y_shift = self.rng.uniform(-1.0, 1.0)

        x = x_shift + r * np.cos(theta)
        y = y_shift + r * np.sin(theta)

        return torch.tensor(
            np.column_stack((x, y)),
            dtype=torch.float32,
        )

    def _sample_square(
        self,
        num_points: int,
    ):
        side_length = self.rng.uniform(2.0, 2.0)
        h = side_length / 2.0

        x_shift = self.rng.uniform(-1.0, 1.0)
        y_shift = self.rng.uniform(-1.0, 1.0)

        coord = self.rng.uniform(
            -h,
            h,
            num_points,
        )

        edge = self.rng.integers(
            0,
            2,
            size=num_points,
        )

        sign = (
            2
            * self.rng.integers(
                0,
                2,
                size=num_points,
            )
            - 1
        )

        x = (
            x_shift
            + edge * coord
            + (1 - edge) * sign * h
        )

        y = (
            y_shift
            + (1 - edge) * coord
            + edge * sign * h
        )

        return torch.tensor(
            np.column_stack((x, y)),
            dtype=torch.float32,
        )

    def __getitem__(self, idx):
        if self.rng.random() < 0.5:
            target = self._sample_circle(self.num_points)
        else:
            target = self._sample_square(self.num_points)

        source = kernel_interaction_points(
            target,
            sigma=self.sigma,
        )

        return source, target


# ---------------------------------------------------------------------------
# Gaussian location M2M data
# ---------------------------------------------------------------------------

class RandomGaussianLocationDataset(Dataset):
    """
    Gaussian location M2M dataset with fixed latent populations.

    Population-level model:

        Z_i ~ N(beta, Sigma_Z)

        X_ij | Z_i ~ N(Z_i, Sigma_X)

        Y_ij | Z_i ~ N(B0 Z_i, Sigma_Y)

    The latent Z_i are sampled once during initialization, but each call to
    __getitem__ generates a fresh realization of X_i and Y_i.
    """

    def __init__(
        self,
        num_measures: int = 1000,
        num_samples: int = 128,
        beta=None,
        Sigma_Z=None,
        Sigma_X=None,
        Sigma_Y=None,
        B0=None,
        seed: int | None = None,
    ):
        if beta is None:
            beta = torch.zeros(2)

        if Sigma_Z is None:
            Sigma_Z = torch.eye(len(beta))

        if Sigma_X is None:
            Sigma_X = torch.eye(len(beta))

        if Sigma_Y is None:
            Sigma_Y = torch.eye(len(beta))

        if B0 is None:
            B0 = torch.eye(len(beta))

        self.num_measures = num_measures
        self.num_samples = num_samples

        self.beta = torch.as_tensor(beta, dtype=torch.float32)
        self.Sigma_Z = torch.as_tensor(
            Sigma_Z,
            dtype=torch.float32,
        )
        self.Sigma_X = torch.as_tensor(
            Sigma_X,
            dtype=torch.float32,
        )
        self.Sigma_Y = torch.as_tensor(
            Sigma_Y,
            dtype=torch.float32,
        )
        self.B0 = torch.as_tensor(
            B0,
            dtype=torch.float32,
        )

        self.d = self.beta.shape[0]

        expected_matrix_shape = (self.d, self.d)

        for name, matrix in [
            ("Sigma_Z", self.Sigma_Z),
            ("Sigma_X", self.Sigma_X),
            ("Sigma_Y", self.Sigma_Y),
            ("B0", self.B0),
        ]:
            if matrix.shape != expected_matrix_shape:
                raise ValueError(
                    f"{name} must have shape {expected_matrix_shape}, "
                    f"got {matrix.shape}"
                )

        try:
            self.Lz = torch.linalg.cholesky(self.Sigma_Z)
        except RuntimeError as exc:
            raise ValueError("Sigma_Z must be positive definite.") from exc

        try:
            self.Lx = torch.linalg.cholesky(self.Sigma_X)
        except RuntimeError as exc:
            raise ValueError("Sigma_X must be positive definite.") from exc

        try:
            self.Ly = torch.linalg.cholesky(self.Sigma_Y)
        except RuntimeError as exc:
            raise ValueError("Sigma_Y must be positive definite.") from exc

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        eps_z = torch.randn(
            self.num_measures,
            self.d,
            generator=generator,
        )

        self.Z = (
            self.beta.unsqueeze(0)
            + eps_z @ self.Lz.T
        )

        self._generator = generator

    def __len__(self):
        return self.num_measures

    def _sample_source(
        self,
        Z: torch.Tensor,
        generator=None,
    ):
        eps_x = torch.randn(
            self.num_samples,
            self.d,
            generator=generator,
        )

        return Z.unsqueeze(0) + eps_x @ self.Lx.T

    def _sample_target(
        self,
        Z: torch.Tensor,
        generator=None,
    ):
        mean_y = self.B0 @ Z

        eps_y = torch.randn(
            self.num_samples,
            self.d,
            generator=generator,
        )

        return mean_y.unsqueeze(0) + eps_y @ self.Ly.T

    def sample_observation(self, idx: int):
        if not 0 <= idx < self.num_measures:
            raise IndexError(
                f"Index {idx} out of range."
            )

        Z = self.Z[idx]

        X = self._sample_source(Z, generator=None)
        Y = self._sample_target(Z, generator=None)

        return X, Y, Z.clone()

    def __getitem__(self, idx: int):
        return self.sample_observation(idx)

    def get_latent(self, idx: int):
        if not 0 <= idx < self.num_measures:
            raise IndexError(
                f"Index {idx} out of range."
            )

        return self.Z[idx].clone()

    def sample_multiple_observations(
        self,
        idx: int,
        num_replicates: int,
    ):
        if num_replicates < 1:
            raise ValueError(
                "num_replicates must be at least 1."
            )

        Z = self.get_latent(idx)

        X = []
        Y = []

        for _ in range(num_replicates):
            X.append(self._sample_source(Z))
            Y.append(self._sample_target(Z))

        return (
            torch.stack(X),
            torch.stack(Y),
            Z,
        )


# ---------------------------------------------------------------------------
# Fixed Gaussian location M2M data
# ---------------------------------------------------------------------------

class GaussianLocationDataset(Dataset):
    """
    Fixed Gaussian location M2M dataset.

    Population-level model:

        Z_i ~ N(beta, Sigma_Z)

        X_ij | Z_i ~ N(Z_i, Sigma_X)

        Y_ij | Z_i ~ N(B0 Z_i, Sigma_Y)

    Crucially, all Z_i, X_i and Y_i are sampled once during initialization.
    Therefore:

        dataset[i]

    always returns exactly the same tensors.

    This is the recommended dataset class when defining a fixed likelihood
    p(D | theta) for Bayesian inference or MCMC.
    """

    def __init__(
        self,
        num_measures: int = 1000,
        num_samples: int = 128,
        beta=None,
        Sigma_Z=None,
        Sigma_X=None,
        Sigma_Y=None,
        B0=None,
        seed: int | None = None,
    ):
        if beta is None:
            beta = torch.zeros(2)

        if Sigma_Z is None:
            Sigma_Z = torch.eye(len(beta))

        if Sigma_X is None:
            Sigma_X = torch.eye(len(beta))

        if Sigma_Y is None:
            Sigma_Y = torch.eye(len(beta))

        if B0 is None:
            B0 = torch.eye(len(beta))

        self.num_measures = num_measures
        self.num_samples = num_samples

        self.beta = torch.as_tensor(beta, dtype=torch.float32)
        self.Sigma_Z = torch.as_tensor(Sigma_Z, dtype=torch.float32)
        self.Sigma_X = torch.as_tensor(Sigma_X, dtype=torch.float32)
        self.Sigma_Y = torch.as_tensor(Sigma_Y, dtype=torch.float32)
        self.B0 = torch.as_tensor(B0, dtype=torch.float32)

        self.d = self.beta.shape[0]

        expected_matrix_shape = (self.d, self.d)

        for name, matrix in [
            ("Sigma_Z", self.Sigma_Z),
            ("Sigma_X", self.Sigma_X),
            ("Sigma_Y", self.Sigma_Y),
            ("B0", self.B0),
        ]:
            if matrix.shape != expected_matrix_shape:
                raise ValueError(
                    f"{name} must have shape {expected_matrix_shape}, "
                    f"got {matrix.shape}"
                )

        try:
            self.Lz = torch.linalg.cholesky(self.Sigma_Z)
            self.Lx = torch.linalg.cholesky(self.Sigma_X)
            self.Ly = torch.linalg.cholesky(self.Sigma_Y)
        except RuntimeError as exc:
            raise ValueError(
                "Sigma_Z, Sigma_X and Sigma_Y must be positive definite."
            ) from exc

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        # Draw latent populations once.
        eps_z = torch.randn(
            self.num_measures,
            self.d,
            generator=generator,
        )

        self.Z = (
            self.beta.unsqueeze(0)
            + eps_z @ self.Lz.T
        )

        # Draw all observations once.
        eps_x = torch.randn(
            self.num_measures,
            self.num_samples,
            self.d,
            generator=generator,
        )

        eps_y = torch.randn(
            self.num_measures,
            self.num_samples,
            self.d,
            generator=generator,
        )

        self.X = (
            self.Z[:, None, :]
            + eps_x @ self.Lx.T
        )

        target_means = self.Z @ self.B0.T

        self.Y = (
            target_means[:, None, :]
            + eps_y @ self.Ly.T
        )

    def __len__(self):
        return self.num_measures

    def __getitem__(self, idx: int):
        if not 0 <= idx < self.num_measures:
            raise IndexError(
                f"Index {idx} out of range."
            )

        return (
            self.X[idx],
            self.Y[idx],
            self.Z[idx],
        )

    def get_latent(self, idx: int):
        if not 0 <= idx < self.num_measures:
            raise IndexError(
                f"Index {idx} out of range."
            )

        return self.Z[idx].clone()

    def get_all(self):
        """
        Return the complete fixed dataset.

        Returns
        -------
        X : [num_measures, num_samples, d]
        Y : [num_measures, num_samples, d]
        Z : [num_measures, d]
        """
        return self.X, self.Y, self.Z


# ---------------------------------------------------------------------------
# Fixed Student-t location M2M data
# ---------------------------------------------------------------------------

class StudentTLocationDataset(Dataset):
    """
    Fixed Student-t location M2M dataset.

    Population-level model:

        Z_i ~ N(beta, Sigma_Z)

        X_ij | Z_i ~ t_df(Z_i, Sigma_X)

        Y_ij | Z_i ~ t_df(B0 Z_i, Sigma_Y)

    All observations are sampled once during initialization, so __getitem__
    is deterministic.

    Parameterisation
    ----------------
    Sigma_X and Sigma_Y are interpreted as the desired covariance matrices
    of the Student-t distributions, not their scale matrices.

    For df > 2, the Student-t scale matrix is therefore

        scale = ((df - 2) / df) * covariance.

    Consequently the training X and Y have the same covariance matrices as
    the corresponding Gaussian model while having heavier tails.

    This is useful for the misspecification experiment in which the learner
    still uses a Gaussian likelihood.
    """

    def __init__(
        self,
        num_measures: int = 1000,
        num_samples: int = 128,
        df: float = 5.0,
        beta=None,
        Sigma_Z=None,
        Sigma_X=None,
        Sigma_Y=None,
        B0=None,
        seed: int | None = None,
    ):
        if df <= 2:
            raise ValueError(
                "df must be greater than 2 when Sigma_X and Sigma_Y "
                "are specified as covariance matrices."
            )

        if beta is None:
            beta = torch.zeros(2)

        if Sigma_Z is None:
            Sigma_Z = torch.eye(len(beta))

        if Sigma_X is None:
            Sigma_X = torch.eye(len(beta))

        if Sigma_Y is None:
            Sigma_Y = torch.eye(len(beta))

        if B0 is None:
            B0 = torch.eye(len(beta))

        self.num_measures = num_measures
        self.num_samples = num_samples
        self.df = float(df)

        self.beta = torch.as_tensor(beta, dtype=torch.float32)
        self.Sigma_Z = torch.as_tensor(Sigma_Z, dtype=torch.float32)
        self.Sigma_X = torch.as_tensor(Sigma_X, dtype=torch.float32)
        self.Sigma_Y = torch.as_tensor(Sigma_Y, dtype=torch.float32)
        self.B0 = torch.as_tensor(B0, dtype=torch.float32)

        self.d = self.beta.shape[0]

        expected_matrix_shape = (self.d, self.d)

        for name, matrix in [
            ("Sigma_Z", self.Sigma_Z),
            ("Sigma_X", self.Sigma_X),
            ("Sigma_Y", self.Sigma_Y),
            ("B0", self.B0),
        ]:
            if matrix.shape != expected_matrix_shape:
                raise ValueError(
                    f"{name} must have shape {expected_matrix_shape}, "
                    f"got {matrix.shape}"
                )

        try:
            self.Lz = torch.linalg.cholesky(self.Sigma_Z)
            self.Lx = torch.linalg.cholesky(self.Sigma_X)
            self.Ly = torch.linalg.cholesky(self.Sigma_Y)
        except RuntimeError as exc:
            raise ValueError(
                "Sigma_Z, Sigma_X and Sigma_Y must be positive definite."
            ) from exc

        # Student-t scale matrices giving the requested covariance.
        covariance_to_scale = (self.df - 2.0) / self.df

        self.StudentTScale_X = covariance_to_scale * self.Sigma_X
        self.StudentTScale_Y = covariance_to_scale * self.Sigma_Y

        self.Lx_t = torch.linalg.cholesky(self.StudentTScale_X)
        self.Ly_t = torch.linalg.cholesky(self.StudentTScale_Y)

        # Use an independent NumPy RNG so all observations are generated once
        # and deterministically from the supplied seed.
        self.rng = np.random.default_rng(seed)

        # Fixed latent populations.
        eps_z = torch.tensor(
            self.rng.standard_normal(
                size=(self.num_measures, self.d)
            ),
            dtype=torch.float32,
        )

        self.Z = (
            self.beta.unsqueeze(0)
            + eps_z @ self.Lz.T
        )

        # Generate all Student-t observations once.
        self.X = torch.empty(
            self.num_measures,
            self.num_samples,
            self.d,
            dtype=torch.float32,
        )

        self.Y = torch.empty_like(self.X)

        for i in range(self.num_measures):
            self.X[i] = self._sample_multivariate_t(
                mean=self.Z[i],
                chol_scale=self.Lx_t,
                num_samples=self.num_samples,
            )

            self.Y[i] = self._sample_multivariate_t(
                mean=self.B0 @ self.Z[i],
                chol_scale=self.Ly_t,
                num_samples=self.num_samples,
            )

    def _sample_multivariate_t(
        self,
        mean: torch.Tensor,
        chol_scale: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Sample from multivariate Student-t using the normal / chi-square
        representation.
        """

        normal = torch.tensor(
            self.rng.standard_normal(
                size=(num_samples, self.d)
            ),
            dtype=torch.float32,
        )

        chi2 = self.rng.chisquare(
            self.df,
            size=num_samples,
        )

        chi2 = torch.tensor(
            chi2,
            dtype=torch.float32,
        )

        gaussian_part = normal @ chol_scale.T

        return (
            mean.unsqueeze(0)
            + gaussian_part
            / torch.sqrt(chi2.unsqueeze(1) / self.df)
        )

    def __len__(self):
        return self.num_measures

    def __getitem__(self, idx: int):
        if not 0 <= idx < self.num_measures:
            raise IndexError(
                f"Index {idx} out of range."
            )

        return (
            self.X[idx],
            self.Y[idx],
            self.Z[idx],
        )

    def get_latent(self, idx: int):
        if not 0 <= idx < self.num_measures:
            raise IndexError(
                f"Index {idx} out of range."
            )

        return self.Z[idx].clone()

    def get_all(self):
        """
        Return the complete fixed dataset.

        Returns
        -------
        X : [num_measures, num_samples, d]
        Y : [num_measures, num_samples, d]
        Z : [num_measures, d]
        """
        return self.X, self.Y, self.Z


def generate_location_data(
    *,
    num_measures: int,
    num_points: int,
    beta: torch.Tensor,
    Sigma_Z: torch.Tensor,
    Sigma_X: torch.Tensor,
    Sigma_Y: torch.Tensor,
    B0: torch.Tensor,
    df: float = 5.0,
    seed: int | None = None,
):
    """
    Generate a fixed location M2M dataset.

    Latent model:

        Z_i ~ N(beta, Sigma_Z)

    Gaussian observations:

        X_ij | Z_i ~ N(Z_i, Sigma_X)
        Y_ij | Z_i ~ N(B0 Z_i, Sigma_Y)

    Student-t observations:

        X_ij | Z_i ~ t_df(Z_i, Sigma_X)
        Y_ij | Z_i ~ t_df(B0 Z_i, Sigma_Y)

    For the Student-t case, Sigma_X and Sigma_Y denote the covariance
    matrices of the Student-t distributions, not their scale matrices.

    Returns
    -------
    dict with keys:

        "source"
        "target"
        "latent"
        "dataset_config"
    """

    beta = torch.as_tensor(
        beta,
        dtype=torch.float32,
    )

    Sigma_Z = torch.as_tensor(
        Sigma_Z,
        dtype=torch.float32,
    )

    Sigma_X = torch.as_tensor(
        Sigma_X,
        dtype=torch.float32,
    )

    Sigma_Y = torch.as_tensor(
        Sigma_Y,
        dtype=torch.float32,
    )

    B0 = torch.as_tensor(
        B0,
        dtype=torch.float32,
    )

    d = beta.numel()

    expected_shape = (d, d)

    for name, matrix in [
        ("Sigma_Z", Sigma_Z),
        ("Sigma_X", Sigma_X),
        ("Sigma_Y", Sigma_Y),
        ("B0", B0),
    ]:
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {tuple(matrix.shape)}."
            )

    if num_measures < 1:
        raise ValueError(
            "num_measures must be positive."
        )

    if num_points < 1:
        raise ValueError(
            "num_points must be positive."
        )

    if df <= 2:
        raise ValueError(
            "df must be greater than 2."
        )

    # One generator controls the entire experiment.
    generator = torch.Generator()

    if seed is not None:
        generator.manual_seed(seed)

    # Cholesky factors.
    Lz = torch.linalg.cholesky(
        Sigma_Z
    )

    Lx = torch.linalg.cholesky(
        Sigma_X
    )

    Ly = torch.linalg.cholesky(
        Sigma_Y
    )

    # -----------------------------------------------------------------------
    # Latent populations
    # -----------------------------------------------------------------------

    eps_z = torch.randn(
        num_measures,
        d,
        generator=generator,
    )

    latent = (
        beta.unsqueeze(0)
        + eps_z @ Lz.T
    )

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    source = torch.empty(
        num_measures,
        num_points,
        d,
        dtype=torch.float32,
    )

    target = torch.empty_like(
        source
    )

    for i in range(num_measures):

        z_i = latent[i]

        target_mean = B0 @ z_i

        if np.isinf(df):

            # Gaussian case
            eps_x = torch.randn(
                num_points,
                d,
                generator=generator,
            )

            eps_y = torch.randn(
                num_points,
                d,
                generator=generator,
            )

            source[i] = (
                z_i.unsqueeze(0)
                + eps_x @ Lx.T
            )

            target[i] = (
                target_mean.unsqueeze(0)
                + eps_y @ Ly.T
            )

        else:

            source[i] = _sample_multivariate_t(
                mean=z_i,
                covariance=Sigma_X,
                df=df,
                num_samples=num_points,
                generator=generator,
            )

            target[i] = _sample_multivariate_t(
                mean=target_mean,
                covariance=Sigma_Y,
                df=df,
                num_samples=num_points,
                generator=generator,
            )
            

    dataset_config = {
            "num_measures": num_measures,
            "beta": beta,
            "Sigma_Z": Sigma_Z,
            "Sigma_X": Sigma_X,
            "Sigma_Y": Sigma_Y,
            "B0": B0,
            "df": df,
        }

    return {
        "source": source,
        "target": target,
        "latent": latent,
        "dataset_config": dataset_config,
    }