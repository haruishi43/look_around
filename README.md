# LookAround

FindView Environment for Look-Around Agents.
Agents looking around to find "something". 

```
@article{ishikawa2023findview,
  title={FindView: Precise Target View Localization Task for Look Around Agents},
  author={Ishikawa, Haruya and Aoki Yoshimitsu},
  journal={arXiv preprint arXiv:},
  year={2023}
}

```

## Installation

```Bash
pip install -r requirements.txt
```

- Install `mycv` or `mmcv`

## Basic Preparations

### Organizing the directories

Since the experiment outputs many videos as well as checkpoints, I recommend using symoblic links:
- Checkpoint save directory `./results/checkpoints` need to be created or symbolic linked before running the training script.
- Video save directory `./results/videos` need to be created or symbolic linked before running the evaluation script.
- Benchmark directory `./results/benchmarks` need to be created or symoblic linked before running the benchmark script.
- Make sure to download 360 image datasets:
  - SUN360 (`sun360`): download link is not currently online
  - 360-Indoor (`wacv360indoor`): foloow the official instructions [here](https://aliensunmin.github.io/project/360-dataset/)
  - Pano3D (`pano3d`): follow the official instructions [here](https://vcl3d.github.io/Pano3D/)
- Make sure to place of symbolic link each data under `./data` and name them accordingly (SUN360 should be `sun360`, 360-Indoor should be `wacv360indoor`, etc...)

### Dataset

The dataset split used in the paper will be released

The dataset could also be created using the script below:

```Bash
python tools/make_findview_sun360_alpha_dataset.py --config configs/_base_/datasets/sun360_alpha_indoor.py
```

Types of dataset:
- `sun360`:
  - Versions:
    - `alpha`: original (dataset used in conference)
    - `beta`: revised dataset (smaller evaluation splits)
  - Categories: `indoor` or `outdoor`
- `wacv360indoor`:
  - Versions: `alpha`


## Tasks

- `FindView`: Given a target image, look around to find the view
  - target image is sampled from the same equirectangular image
- `FindCorrupted`: Given a corrupted target image, look around to find the uncorrupted view
  - target image is sampled from the same equirectangular image but is augmented (corrupted)
  - the agent will have to find the view that closely resembles the target image


## Scriptsand useful commands

### Print config

```Bash
python tools/print_config.py <config>
```

### Run the environment interactively or some agent/policy

```Bash
python scripts/run_*.py --config <config>
```

### Jupyter Lab

```Bash
pip install jupyterlab

jupyter lab --no-browser --port 8888
```

## Type Checking

```Bash
mypy . --ignore-missing-imports
```
