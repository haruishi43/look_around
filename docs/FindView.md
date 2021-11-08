# FindView Task

## Training

Start Training:

```Bash
CUDA_VISIBLE_DEVICES=0, python findview_baselines/run_ppo.py --config configs/findview/sun360/rl/basic_1_alpha_indoor.py --mode train
```

It is required that the VRAM is atleast 24GB. To train using smaller VRAM, use `basic_1_half_alpha_indoor.py` which requires around 11GB.

Resume Training:

```Bash
CUDA_VISIBLE_DEVICES=0, python findview_baselines/run_ppo.py --config configs/findview/sun360/rl/basic_1_alpha_indoor.py --mode train --options trainer.resume=True
```

Tensorboard:

```Bash
# running inside docker container
tensorboard --logdir tb --port 8889 --bind_all
```

## Test

To test the trained RL agents on `test` split of the dataset, run `--mode eval` along with changes to `validator` configuration using `--options`.

```Bash
CUDA_VISIBLE_DEVICES=0, python findview_baselines/run_ppo.py --config configs/findview/sun360/rl/basic_1_alpha_indoor.py --mode eval --options validator.num_envs=1
```

### Benchmarking

Benchmark code for `ppo`:

```Bash
CUDA_VISIBLE_DEVICES=0, python findview_baselines/agents/ppo.py --config configs/findview/sun360/benchmarks/ppo_alpha_indoor.py --ckpt-path results/checkpoints/sun360_alpha_indoor/basic/run_1/ckpt.best.pth
```

Benchmark code for `feature matching`:

```Bash
OPENBLAS_NUM_THREADS=8, python findview_baselines/agents/feature_matching.py --config configs/findv
iew/sun360/benchmarks/fm_alpha_indoor.py
```

__NOTE__: since `numpy`'s threads are not limited, you would need to set the environment variables before running the script.
