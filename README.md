# LookAround

Agents looking around to find "something".

## Installation

```Bash
pip install -r requirements.txt
```

- Install `mycv` or `mmcv`

## Preperation

- Checkpoint save directory `./results/checkpoints` need to be created or symbolic linked before running the training script.
- Video save directory `./results/videos` need to be created or symbolic linked before running the evaluation script.


### Make Dataset

```Bash
python scripts/make_sun360_dataset.py --config configs/_base_/datasets/sun360_alpha_indoor.py
```

## Training

Start Training:

```Bash
CUDA_VISIBLE_DEVICES=0, python findview_baselines/run_ppo.py --config configs/run_3.py --mode train
CUDA_VISIBLE_DEVICES=1, python findview_baselines/run_ppo.py --config configs/run_4.py --mode train
```

Resume Training:

```Bash

```

Tensorboard:

```Bash
# running inside docker container
tensorboard --logdir tb --port 8889 --bind_all
```

## Test

Temporary test using `trainer` (make sure that `num_envs=1`):

```Bash
CUDA_VISIBLE_DEVICES=0, python findview_baselines/run_ppo.py --config configs/run_3.py --mode test --options num_envs=1
```

Test using the benchmark code (WIP):

```Bash

```


## Other scripts

### Print config

```Bash
python scripts/print_config.py <config>
```

### Run the environment interactively or some agent/policy

```Bash
python scripts/run_*.py --config <config>
```

## Type Checking

```Bash
mypy . --ignore-missing-imports
```
