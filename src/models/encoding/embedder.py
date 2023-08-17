"""Interface for various embedding methods."""
from typing import Protocol

import torch


class Embedder(Protocol):
    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def input_dim(self) -> int:
        ...

    @property
    def out_dim(self) -> int:
        ...
