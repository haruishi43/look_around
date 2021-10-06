_base_ = [
    '../findview_agents/ppo.py',
    '../rl_envs/basic.py',
    '../trainers/base.py'
]
dataset = dict(
    difficulty='easy',
    bounded=False,
)
rl_env = dict(
    name='Basic',
    success_reward=100.0,
    slack_reward=-0.01,
    end_type='bell',
    end_type_param=10,
)
base_trainer = dict(
    run_id=9876,
    num_envs=16,
    num_updates=200,
    ckpt_interval=50,
    log_interval=10,
    ckpt_dir="{results_root}/checkpoints/run_{run_id}",
    video_dir="{results_root}/videos/test_run_{run_id}",
    tb_dir="{tb_root}/run_{run_id}",
    log_file="{log_root}/{split}_test_run_{run_id}.log",
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=10,
)
val = dict(
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    num_eval_episodes=250,
)
test = dict(
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    ckpt_path="ckpt.0.pth",
    use_ckpt_cfg=True,
    num_eval_episodes=250,
    difficulty="easy",
    bounded=True,
)
