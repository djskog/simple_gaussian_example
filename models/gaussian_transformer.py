import torch
import torch.nn as nn


class GaussianTransformer(nn.Module):
    def __init__(
        self,
        d,
        Ly,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        
        self.Ly = Ly

        self.input_projection = nn.Linear(d, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d),
        )

    def forward(self, x):
        """
        x: [batch, num_points, d]

        returns:
            mean: [batch, d]
        """

        h = self.input_projection(x)

        h = self.encoder(h)

        # permutation-invariant pooling
        pooled = h.mean(dim=1)

        mean = self.output_head(pooled)

        return mean