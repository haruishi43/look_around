_base_ = [
    './findview_agents/ppo.py',
    './rl_trainer.py',
]
run_id = 4
num_envs = 16
num_updates = 20000
num_ckpts = -1
ckpt_interval = 500
total_num_steps = -1.0
log_interval = 10
verbose = True
train = dict(
    device=0,
    is_torch=True,
    resume=False,
)
val = dict(
    device=0,
    is_torch=True,
    episode_count=-1,
)
test = dict(
    device=0,
    is_torch=True,
    ckpt_path="",
    use_ckpt_cfg=True,
    episode_count=20,
)
