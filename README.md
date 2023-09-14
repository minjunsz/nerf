# Implementation of NeRF Model

This is an implementation of the original NeRF model. [paper](https://arxiv.org/abs/2003.08934)

I referenced
- [pytorch-implementation by yenchenlin](https://github.com/yenchenlin/nerf-pytorch)
- [pytorch-lightning implementation by kwea123](https://github.com/kwea123/nerf_pl)

```
.
├── configs                 # Storing config files for experiments
├── data                    # dedicated directory for data
├── notebook                # collection of interactive jupyter notebooks
├── notes                   # Simple markdown files what I studied during the implementation
└── src                     # most codes goes into this directory
    ├── dataset             # Impl. pytorch dataset with Blender Lego data.
    ├── models              # Pytorch lightning implementation of the nerf model.
    └── utils               # This directory includes volumetric rendering and 3D samplings. Other auxiliary codes goes into this directory as well.

```

<!-- │   ├── benchmarks          # Load and stress tests
│   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
│   └── unit                # Unit tests
└── ... -->
