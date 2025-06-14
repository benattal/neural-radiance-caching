# Table of contents
-----
  * [Introduction](#introduction)
  * [Installation](#installation)
  * [Datasets](#datasets)
  * [Quick start](#quick-start)
------

# Introduction

This repository contains the code for two papers:

1. [Flash Cache: Reducing Bias in Radiance Cache Based Inverse Rendering](https://benattal.github.io/flash-cache/)
2. [Neural Inverse Rendering from Propagating Light](https://benattal.github.io/flash-cache/)

## Flash Cache: Reducing Bias in Radiance Cache Based Inverse Rendering

### [Website](https://benattal.github.io/flash-cache/) |  [Paper](https://benattal.github.io/flash-cache/flash_cache.pdf)

The goal of *Flash Cache* is to recover the geometry, materials, and potentially unknown lighting of a scene from a set of conventional images. It works by modeling light transport using *radiance caching*, a technique that can accelerate physically-based rendering. It implements several techniques to improve speed and reduce bias in radiance caching.

## Neural Inverse Rendering from Propagating Light

### [Website](https://benattal.github.io/flash-cache/) |  [Paper](https://benattal.github.io/flash-cache/flash_cache.pdf)

*Neural Inverse Rendering from Propagating Light* is an extension of *Flash Cache* that models time-resolved light transport, and performs inverse rendering from ultrafast videos that capture light in flight.

# Installation

To install all required dependences, run

```bash
bash install_environment.sh
```

# Datasets
TBA

# Quick Start

At a high level, this system works by:

1. Training a ``Cache'' or NeRF of a scene, which gives an initial estimate of geometry, and radiance leaving every point.
2. Training a physically-based ``Material model'', which predicts outgoing illumination by integrating the cache against a Disney-GGX BRDF.

To train both models simultaneously, run

```
bash scripts/train.sh --scene <scene_name> --stage material_light_from_scratch_resample --batch_size 1024 --render_chunk_size 1024
```

Intermediate images will be written, by default, to `~/checkpoints/yobo_results/synthetic/<scene_name>_<stage>`. 

Try running the above with the scene name set to `hotdog`. You should see results in `~/checkpoints/yobo_results/synthetic/hotdog_material_light_from_scratch_resample`.

# Running Flash Cache

In our paper, we train and evaluate flash cache on the following scenes from the TensoIR-synthetic dataset: `hotdog`, `lego`, `armadillo`, and `ficus`. We also train and evaluate on the following scenes from the open illumination dataset: `obj_02_egg`, `obj_04_stone`, `obj_05_bird`, `obj_17_box`, `obj_26_pumpkin`, `obj_29_hat`, `obj_35_cup`, `obj_36_sponge`, `obj_42_banana`, `obj_48_bucket`.

In order to perform evaluation for a specific scene, run:

```
bash scripts/eval.sh --scene <scene_name> --stage material_light_from_scratch_resample --render_chunk_size 1024 --render_repeats N
```

where the physically-based renderings are averaged `N` times.

# Running Neural Inverse Rendering from Propagating Light

In our paper, we train and evaluate inv prop on the following scenes from our synthetic dataset: `cornell`, `pots`, `peppers`, and `kitchen`. We also train and evaluate on the following captured scenes: `statue`, `spheres`, `globe`, `house`.

In order to perform evaluation for a specific scene, run:

```
bash scripts/eval.sh --scene <scene_name> --stage material_light_from_scratch_resample --render_chunk_size 1024 --render_repeats N
```

where the physically-based renderings are averaged `N` times.

# Citation

```bibtex
@inproceedings{attal2024flash,
  title={Flash cache: Reducing bias in radiance cache based inverse rendering},
  author={Attal, Benjamin and Verbin, Dor and Mildenhall, Ben and Hedman, Peter and Barron, Jonathan T and O’Toole, Matthew and Srinivasan, Pratul P},
  booktitle={European Conference on Computer Vision},
  pages={20--36},
  year={2024},
  organization={Springer}
}
```

```bibtex
@inproceedings{malik2025neural,
  title={Neural Inverse Rendering from Propagating Light},
  author={Malik, Anagh and Attal, Benjamin and Xie, Andrew, and O’Toole, Matthew and Lindell, David},
  booktitle={CVPR},
  year={2025},
}
```