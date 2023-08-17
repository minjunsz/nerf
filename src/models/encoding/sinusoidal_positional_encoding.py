from typing import Union

import torch


class SinusoidalEmbedder:
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
