"""Core module implementing NeRF models."""
from typing import Union
import torch
from torch import nn


class Embedder:
    """An auxiliary class providing positional encoding feature.
    The output encoding includes original input and utilizes sin, cos embedding functions.
    This positional encoding uses evenly distributed frequency bands in log-scale."""

    def __init__(
        self,
        input_dim: int,
        num_freqs: int,
        max_freq: Union[int, float],
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            input feature dimension.
        num_freqs : int
            desired number of frequency bands.
        max_freq : Union[int, float]
            log_2(MAX_FREQUENCY). The frequency bands will span from 2^0 to 2^max_freq.
        """
        self.input_dim = input_dim
        # Implemented the log-sampling.
        self.freq_bands = torch.pow(2, torch.linspace(0.0, max_freq, steps=num_freqs))
        self.out_dim = input_dim * (2 * num_freqs + 1)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return embeddings of given inputs

        Parameters
        ----------
        inputs : torch.Tensor
            Shape: [..., self.input_dim]

        Returns
        -------
        torch.Tensor
            embedding; [..., self.out_dim]
        """
        assert inputs.size(-1) == self.input_dim, "Input dimension doesn't match."
        input_times_freq_band = inputs.unsqueeze(-1) * self.freq_bands
        sin_embedding = torch.sin(input_times_freq_band)
        cos_embedding = torch.cos(input_times_freq_band)
        embedding = torch.cat(
            [inputs.unsqueeze(-1), sin_embedding, cos_embedding], dim=-1
        )
        flattened_embedding = embedding.flatten(-2, -1)
        assert (
            flattened_embedding.size(-1) == self.out_dim
        ), "Output dimension doesn't match."
        return flattened_embedding


class NeRF(nn.Module):
    """Model architecture follows default model in the NeRF paper.
    8 linear layers with one skip connection at 4th layer.
    """

    def __init__(
        self,
        xyz_channel: int,
        viewdir_channel: int,
        hidden_dim: int = 256,
        *args,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        xyz_channel : int
            Dimension of xyz input embeddings
        viewdir_channel : int
            Dimension of viewing direction input embeddings
        hidden_dim : int, optional
            Width of hidden layers, by default 256
        """
        super().__init__(*args, **kwargs)
        self.xyz_channel = xyz_channel
        self.viewdir_channel = viewdir_channel
        self.hidden_dim = hidden_dim

        self.MLP_1 = nn.Sequential(
            nn.Linear(self.xyz_channel, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.MLP_2 = nn.Sequential(
            nn.Linear(self.xyz_channel + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.feature_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.alpha_linear = nn.Linear(self.hidden_dim, 1)
        self.view_linear = nn.Linear(
            self.viewdir_channel + self.hidden_dim,
            self.hidden_dim // 2,
        )
        self.rgb_linear = nn.Linear(self.hidden_dim // 2, 3)

    def forward(self, x):
        inputs, views = torch.split(x, [self.xyz_channel, self.viewdir_channel], dim=-1)
        h1 = self.MLP_1(inputs)
        h2 = self.MLP_2(torch.cat([inputs, h1], dim=-1))

        alpha = self.alpha_linear(h2)
        feature = self.feature_linear(h2)
        h3 = self.view_linear(torch.cat([views, feature], dim=-1))
        rgb = self.rgb_linear(h3)

        return torch.cat([rgb, alpha], dim=-1)
