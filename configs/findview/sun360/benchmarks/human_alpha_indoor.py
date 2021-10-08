_base_ = [
    '../rl_env/basic.py',
    './base.py',
]
results_root = "./results"
log_root = "./logs"
video_option = ["disk"]
video_dir = "{results_root}/videos/run_human"
log_file = "{log_root}/run_human.log"
verbose = True
