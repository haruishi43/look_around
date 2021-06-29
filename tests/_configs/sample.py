basic = dict(
    batch_size=2,
    num_episodes=2,
)
dataset = dict(
    name="SUN360_sample",
    data_root="tests/_data",
    data_path="SUN360_sample",
    dataset_root="tests/_dataset",
    dataset_path="SUN360_sample_indoor",
)
simulator = dict(
    name="SingleFrame",
    equi2pers=dict(
        w_pers=640,
        h_pers=480,
        fov_x=90,
        skew=0.0,
        sampling_method="default",
        mode="bilinear",
        z_down=True,
    ),
)
task = dict(
    name="FindView",
)
