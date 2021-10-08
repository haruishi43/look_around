_base_ = [
    '../rl_envs/basic.py',
    '../agents/feature_matching.py',
    './base.py',
]
results_root = "./results"
log_root = "./logs"
video_option = ["disk"]
video_dir = "{results_root}/videos/run_fm"
log_file = "{log_root}/run_fm.log"
verbose = True
