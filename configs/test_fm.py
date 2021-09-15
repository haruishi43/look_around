_base_ = [
    './_base_/datasets/sun360_alpha_indoor.py',
    './_base_/findview_sim.py',
    './findview_agents/feature_matching.py'
]
results_root = "./results"
log_root = "./logs"
video_option = ["disk"]
video_dir = "{results_root}/videos/run_fm"
log_file = "{log_root}/run_fm.log"
verbose = True
