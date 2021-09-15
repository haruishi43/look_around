_base_ = [
    './_base_/datasets/sun360_alpha_indoor.py',
    './_base_/findview_sim.py',
    './findview_agents/ppo.py',
]
run_id = 4
results_root = "./results"
log_root = "./logs"
tb_root = "./tb"
video_option = ["disk"]
ckpt_dir = "{results_root}/checkpoints/run_{run_id}"
video_dir = "{results_root}/videos/run_{run_id}"
tb_dir = "{tb_root}/run_{run_id}"
log_file = "{log_root}/{split}_run_{run_id}.log"
verbose = True
test = dict(
    device=0,
    is_torch=True,
    ckpt_path="",
    use_ckpt_cfg=True,
    episode_count=20,
)
