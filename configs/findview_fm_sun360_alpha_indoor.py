_base_ = [
    './rl_envs/basic.py',
    './findview_agents/feature_matching.py'
]
results_root = "./results"
log_root = "./logs"
video_option = ["disk"]
video_dir = "{results_root}/videos/run_fm"
log_file = "{log_root}/run_fm.log"
verbose = True
