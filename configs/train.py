_base_ = ['./findview_baselines/ppo.py']
run_id = 1
num_envs = 16
num_updates = 10000
num_ckpts = 200
ckpt_interval = -1
total_num_steps = -1.0
log_interval = 10
results_root = "./results"
log_root = "./logs"
tb_root = "./tb"
video_option = ["disk"]
ckpt_dir = "{results_root}/checkpoints/run_{run_id}"
video_dir = "{results_root}/videos/run_{run_id}"
tb_dir = "{tb_root}/run_{run_id}"
log_file = "{log_root}/train_run_{run_id}.log"
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
    episode_count=10,
)
