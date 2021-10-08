benchmark = dict(
    device=0,
    dtype="torch.float32",
    vec_type="threaded",
    ckpt_path="ckpt.best.pth",
    use_ckpt_cfg=True,
    num_eval_episodes=250,
    difficulty="easy",
    bounded=True,
)
