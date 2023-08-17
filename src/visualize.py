from pathlib import Path

import imageio
import numpy as np
import torch

from src.dataset.lego import read_camera_params
from src.utils.config_parser import get_config
from src.utils.generate_render_pose import c2w
from src.utils.get_rays import get_rays
from src.utils.rendering import render_rays

from .train import instantiate_model


def visualize(run_id: int, iteration: int):
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")

    config = get_config()
    data_path = Path(config.data_path)
    height, width, focal_length = read_camera_params(data_path, "train", half_res=True)
    K = torch.tensor(
        [
            [focal_length, 0, 0.5 * width],
            [0, focal_length, 0.5 * height],
            [0, 0, 1],
        ]
    )
    near, far = 2.0, 6.0

    coarse_model, fine_model, xyz_embedder, viewdir_embedder = instantiate_model()
    checkpoint = torch.load(
        Path.cwd() / "checkpoints" / str(run_id) / f"iter_{iteration}.pt"
    )
    coarse_model.load_state_dict(checkpoint["coarse_model"])
    fine_model.load_state_dict(checkpoint["fine_model"])

    coarse_model.eval()
    fine_model.eval()

    mat = torch.tensor(
        [
            [
                0.4429636299610138,
                0.31377720832824707,
                -0.8398374915122986,
                -3.385493516921997,
            ],
            [
                -0.8965396881103516,
                0.1550314873456955,
                -0.41494810581207275,
                -1.6727094650268555,
            ],
            [0.0, 0.936754584312439, 0.3499869406223297, 1.4108426570892334],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )  # r02

    rays_o, rays_d = get_rays(height, width, K, c2w=c2w(4.0, 60.0, 50.0))
    # rays_o, rays_d = get_rays(height, width, K, c2w=mat)
    rays_o, rays_d = rays_o.flatten(0, 1), rays_d.flatten(0, 1)
    render_result = render_rays(
        coarse_model,
        fine_model,
        rays_o,
        rays_d,
        xyz_embedder,
        viewdir_embedder,
        near=near,
        far=far,
        render_only=True,
    )

    img = render_result["render"].rgb_map.reshape(height, width, 3)
    img = (img.numpy() * 255).astype(np.uint8)
    imageio.imwrite(Path.cwd() / f"render_{iteration}.png", img)
    print("done")


if __name__ == "__main__":
    with torch.no_grad():
        visualize(1, 90)
