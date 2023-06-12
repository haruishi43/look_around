_base_ = ["../dataset.py"]
dataset = dict(
    name="sun360",
    version="alpha",
    category="indoor",
    num_easy=30,
    num_medium=40,
    num_hard=30,
    fov=90.0,
    json_path=("{root}/" "{name}/" "{version}/" "{category}/" "{split}.json"),
)
