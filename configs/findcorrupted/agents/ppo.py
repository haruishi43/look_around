_base_ = [
    '../_base_/benchmark.py',
]
policy = dict(
    action_distribution_type="categorical",
    use_log_std=False,
    use_softplus=False,
    min_std=1e-6,
    max_std=1,
    min_log_std=-5,
    max_log_std=2,
    action_activation="tanh",
)
ppo = dict(
    clip_param=0.2,
    ppo_epoch=4,
    num_mini_batch=1,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    lr=2.5e-4,
    eps=1e-5,
    max_grad_norm=0.5,
    num_steps=128,
    use_gae=True,
    use_linear_lr_decay=True,
    use_linear_clip_decay=True,
    gamma=0.99,
    tau=0.95,
    reward_window_size=50,
    use_normalized_advantage=False,
    hidden_size=512,
    use_double_buffered_sampler=False,
)
benchmark = dict(
    device=0,
    dtype="torch.float32",
)
