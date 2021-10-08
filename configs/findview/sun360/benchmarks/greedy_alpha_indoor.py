_base_ = [
    '../rl_envs/basic.py',
    '../agents/greedy.py',
    './base.py',
]
results_root = "./results"
log_root = "./logs"
video_option = ["disk"]
video_dir = "{results_root}/videos/run_greedy"
log_file = "{log_root}/run_greedy.log"
verbose = True
