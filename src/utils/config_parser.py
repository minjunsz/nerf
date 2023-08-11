import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    data_path: str
    project_name: str


def get_config() -> Config:
    default_config_path = Path.home() / "codes/nerf/configs/config.yaml"

    parser = argparse.ArgumentParser(
        description="simple argparse for the config file path"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="path to the config file in .yaml format.",
        default=str(default_config_path),
        nargs="?",
    )
    args = parser.parse_args()
    with open(args.config_path) as f:
        cfg_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Config(**cfg_yaml)
    return config
