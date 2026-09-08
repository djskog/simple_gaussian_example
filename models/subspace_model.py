from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class SubspaceModel:
    """
    Wrapper around a deterministic neural network whose trainable weights are
    restricted to an affine PCA subspace

        w(phi) = w_hat + P phi,

    where:
        w_hat : [D]       - shift / SWA weight vector
        P     : [D, K]    - scaled PCA basis
        phi     : [K]       - low-dimensional Bayesian parameter

    This follows the subspace-inference construction used in
    Izmailov et al. / "Subspace Inference for Bayesian Deep Learning".

    The underlying network remains an ordinary deterministic PyTorch module.
    Before each evaluation, its trainable parameters are overwritten with
    w(phi).

    Notes
    -----
    This implementation is intentionally designed for posterior evaluation
    and elliptical-slice sampling, where gradients with respect to phi are not
    required. Loading w(phi) into the network uses copy_ under no_grad.

    For gradient-based inference in phi (e.g. HMC), a functional model using
    torch.func.functional_call should be used instead.
    """

    def __init__(
        self,
        base_model: nn.Module,
        swa_mean: torch.Tensor,
        pca_basis: torch.Tensor,
        parameter_info: list[tuple[str, tuple[int, ...], int]] | None = None,
        device: str | torch.device | None = None,
    ):
        self.base_model = base_model

        if device is not None:
            self.device = torch.device(device)
            self.base_model.to(self.device)
        else:
            try:
                self.device = next(self.base_model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self.swa_mean = torch.as_tensor(
            swa_mean,
            dtype=torch.float32,
            device=self.device,
        ).detach()

        self.pca_basis = torch.as_tensor(
            pca_basis,
            dtype=torch.float32,
            device=self.device,
        ).detach()

        if self.swa_mean.ndim != 1:
            raise ValueError(
                f"swa_mean must have shape [D], got {tuple(self.swa_mean.shape)}"
            )

        if self.pca_basis.ndim != 2:
            raise ValueError(
                "pca_basis must have shape [D, K], "
                f"got {tuple(self.pca_basis.shape)}"
            )

        D, K = self.pca_basis.shape

        if self.swa_mean.numel() != D:
            raise ValueError(
                "swa_mean and pca_basis disagree: "
                f"swa_mean has {self.swa_mean.numel()} parameters, "
                f"but basis has first dimension {D}."
            )

        self.num_parameters = D
        self.subspace_dim = K
        self.parameter_info = parameter_info

        # Build the parameter slices once. This avoids repeatedly traversing
        # the model when reconstructing w(phi).
        self._parameter_slices = self._build_parameter_slices()

        expected_num_parameters = sum(
            numel for _, _, numel in self._parameter_slices
        )

        if expected_num_parameters != D:
            raise ValueError(
                "PCA subspace dimension does not match the supplied model. "
                f"Model has {expected_num_parameters} trainable parameters, "
                f"but swa_mean has {D}."
            )

    # ------------------------------------------------------------------
    # Parameter bookkeeping
    # ------------------------------------------------------------------

    def _build_parameter_slices(
        self,
    ) -> list[tuple[str, tuple[int, ...], int]]:
        """
        Determine the flattening order and parameter shapes.

        If parameter_info was saved by collect_pca_trajectory.py, verify that
        it agrees with the current model ordering.
        """

        current_info = [
            (name, tuple(param.shape), param.numel())
            for name, param in self.base_model.named_parameters()
            if param.requires_grad
        ]

        if self.parameter_info is not None:
            saved_info = [
                (name, tuple(shape), int(numel))
                for name, shape, numel in self.parameter_info
            ]

            if saved_info != current_info:
                raise ValueError(
                    "The PCA checkpoint's parameter_info does not match "
                    "the current model. The model architecture or parameter "
                    "ordering may have changed."
                )

        return current_info

    def flatten_parameters(self) -> torch.Tensor:
        """Return the current trainable network weights as one vector."""
        return torch.cat(
            [
                p.detach().reshape(-1)
                for p in self.base_model.parameters()
                if p.requires_grad
            ]
        )

    def reconstruct_weights(
        self,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the full network weight vector

            w(phi) = w_hat + P phi.

        Parameters
        ----------
        phi:
            Shape [K] or [batch, K].

        Returns
        -------
        weights:
            Shape [D] or [batch, D].
        """
        phi = torch.as_tensor(
            phi,
            dtype=self.pca_basis.dtype,
            device=self.device,
        )

        if phi.ndim == 1:
            if phi.numel() != self.subspace_dim:
                raise ValueError(
                    f"phi must have {self.subspace_dim} entries, "
                    f"got {phi.numel()}."
                )

            return self.swa_mean + self.pca_basis @ phi

        if phi.ndim == 2:
            if phi.shape[1] != self.subspace_dim:
                raise ValueError(
                    f"phi must have shape [batch, {self.subspace_dim}], "
                    f"got {tuple(phi.shape)}."
                )

            return self.swa_mean.unsqueeze(0) + phi @ self.pca_basis.T

        raise ValueError(
            f"phi must be one- or two-dimensional, got ndim={phi.ndim}"
        )

    def load_weights(
        self,
        flat_weights: torch.Tensor,
    ) -> None:
        """
        Copy one flattened weight vector into the underlying network.
        """

        flat_weights = torch.as_tensor(
            flat_weights,
            dtype=torch.float32,
            device=self.device,
        )

        if flat_weights.ndim != 1:
            raise ValueError(
                "flat_weights must have shape [D]."
            )

        if flat_weights.numel() != self.num_parameters:
            raise ValueError(
                f"Expected {self.num_parameters} weights, "
                f"got {flat_weights.numel()}."
            )

        offset = 0

        with torch.no_grad():
            for name, shape, numel in self._parameter_slices:
                # Find the matching parameter by name.
                param = dict(self.base_model.named_parameters())[name]

                chunk = flat_weights[
                    offset : offset + numel
                ].view(shape)

                param.copy_(chunk.to(
                    device=param.device,
                    dtype=param.dtype,
                ))

                offset += numel

    def set_phi(
        self,
        phi: torch.Tensor,
    ) -> None:
        """
        Set the underlying transformer to the weights corresponding to phi.
        """
        weights = self.reconstruct_weights(phi)

        if weights.ndim != 1:
            raise ValueError(
                "set_phi accepts a single subspace point phi with shape [K]."
            )

        self.load_weights(weights)

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def __call__(
        self,
        phi: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Evaluate the base network at subspace coordinate phi.
        """
        self.set_phi(phi)
        return self.base_model(*args, **kwargs)

    def forward(
        self,
        phi: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.__call__(phi, *args, **kwargs)

    def to(
        self,
        device: str | torch.device,
    ) -> "SubspaceModel":
        """Move the underlying network and subspace tensors to a device."""
        self.device = torch.device(device)
        self.base_model.to(self.device)
        self.swa_mean = self.swa_mean.to(self.device)
        self.pca_basis = self.pca_basis.to(self.device)
        return self

    def train(self, mode: bool = True) -> "SubspaceModel":
        self.base_model.train(mode)
        return self

    def eval(self) -> "SubspaceModel":
        self.base_model.eval()
        return self

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def zero_coordinate(
        self,
    ) -> torch.Tensor:
        """
        Return phi=0.

        This corresponds exactly to the SWA shift point w_hat.
        """
        return torch.zeros(
            self.subspace_dim,
            dtype=self.pca_basis.dtype,
            device=self.device,
        )

    def get_weights(
        self,
    ) -> torch.Tensor:
        """Return the currently loaded full parameter vector."""
        return self.flatten_parameters()

    def state_at(
        self,
        phi: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Return a copy of the underlying state_dict after setting phi.

        Useful for inspection or temporarily storing a sampled model.
        """
        self.set_phi(phi)
        return {
            key: value.detach().clone()
            for key, value in self.base_model.state_dict().items()
        }


# ---------------------------------------------------------------------------
# Loading helper
# ---------------------------------------------------------------------------


def load_subspace_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    device: str | torch.device = "cpu",
) -> tuple[SubspaceModel, dict[str, Any]]:
    """
    Load a PCA-subspace checkpoint produced by collect_pca_trajectory.py.

    Parameters
    ----------
    checkpoint_path:
        Path to pca_subspace.pt.

    model:
        Fresh instance of the deterministic transformer architecture with the
        same parameter ordering as the model used to construct the PCA basis.

    device:
        Device on which the network and subspace are stored.

    Returns
    -------
    subspace_model:
        A SubspaceModel instance.

    checkpoint:
        The full PCA checkpoint dictionary.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"PCA subspace checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "swa_mean",
        "pca_basis",
        "parameter_info",
    }

    missing = required.difference(checkpoint.keys())

    if missing:
        raise KeyError(
            "PCA checkpoint is missing required entries: "
            + ", ".join(sorted(missing))
        )

    subspace_model = SubspaceModel(
        base_model=model,
        swa_mean=checkpoint["swa_mean"],
        pca_basis=checkpoint["pca_basis"],
        parameter_info=checkpoint["parameter_info"],
        device=device,
    )

    # Start at the SWA point phi=0.
    subspace_model.set_phi(
        subspace_model.zero_coordinate()
    )

    return subspace_model, checkpoint


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        "SubspaceModel defines w(phi) = w_hat + P phi.\n"
        "Use load_subspace_checkpoint(...) with a matching transformer."
    )
