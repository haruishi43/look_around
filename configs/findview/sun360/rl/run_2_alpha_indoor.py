_base_ = [
    '../agents/ppo.py',
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
    end_type='inverse',
    end_type_param=10,
)
trainer = dict(
    run_id=2,
    num_envs=16,
    num_updates=15000,
    ckpt_interval=500,
    log_interval=10,
)
validator = dict(
    num_envs=16,
    num_eval_episodes=-1,
    ckpt_path="ckpt.best.pth",
    use_ckpt_cfg=True,
    difficulty="hard",
    bounded=False,
    remove_labels="others",
    num_episodes_per_img=1,
)
scheduler = dict(
    initial_difficulty='easy',
    update_interval=5000,
)
