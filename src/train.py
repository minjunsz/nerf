"""This module includes whole training pipeline for NeRF."""
from pathlib import Path
from src.models.nerf import Embedder, NeRF
from src.dataset.lego import LegoDataset
from src.utils.config_parser import get_config


def instantiate_model(
    xyz_max_freq: int = 10,
    viewdir_max_freq: int = 4,
    hidden_dim: int = 256,
):
    """Instantiate embedders and core NeRF models.
    Parameters
    ----------
    xyz_max_freq : int, optional
        log_2(MAX_FREQ) for xyz positional encoding, by default 10
    viewdir_max_freq : int, optional
        log_2(MAX_FREQ) for viewing direction encoding, by default 4
    hidden_dim : int, optional
        width of hidden layer, by default 256

    Returns
    -------
    tuple[NeRF, NeRF, Embedder, Embedder]
        Return tuple; (coarse_model, fine_model, xyz_embedder, viewdir_embedder)
    """
    xyz_embedder = Embedder(
        input_dim=3,
        num_freqs=xyz_max_freq + 1,
        max_freq=xyz_max_freq,
    )
    viewdir_embedder = Embedder(
        input_dim=3,
        num_freqs=viewdir_max_freq + 1,
        max_freq=viewdir_max_freq,
    )
    coarse_model = NeRF(
        xyz_channel=xyz_embedder.out_dim,
        viewdir_channel=viewdir_embedder.out_dim,
        hidden_dim=hidden_dim,
    )

    fine_model = NeRF(
        xyz_channel=xyz_embedder.out_dim,
        viewdir_channel=viewdir_embedder.out_dim,
        hidden_dim=hidden_dim,
    )

    return coarse_model, fine_model, xyz_embedder, viewdir_embedder


def train():
    config = get_config()
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise Exception("No such data directory")
    train_data = LegoDataset(data_path, "train", half_res=True)
    val_data = LegoDataset(data_path, "val", half_res=True)
    test_data = LegoDataset(data_path, "test", half_res=True)


if __name__ == "__main__":
    train()
