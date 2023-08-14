from pathlib import Path

import imageio
import numpy as np
import torch

from src.dataset.lego import read_camera_params
from src.utils.config_parser import get_config
from src.utils.generate_render_pose import c2w
from src.utils.get_rays import get_rays
from src.utils.rendering import render_img

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
                -0.9993038773536682,
                -0.0326223149895668,
                0.018094748258590698,
                0.07294226437807083,
            ],
            [
                0.037304628640413284,
                -0.8738756775856018,
                0.4847160875797272,
                1.9539530277252197,
            ],
            [0.0, 0.48505374789237976, 0.8744843602180481, 3.5251591205596924],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # rays_o, rays_d = get_rays(height, width, K, c2w=c2w(4.0, 30.0, 0.0))
    rays_o, rays_d = get_rays(height, width, K, c2w=mat)
    rays_o, rays_d = rays_o.flatten(0, 1), rays_d.flatten(0, 1)
    render_result = render_img(
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
        visualize(2, 85)
