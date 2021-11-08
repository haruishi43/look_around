_base_ = [
    '../_base_/benchmark.py',
]
fm = dict(
    feature_type="ORB",
    matcher_type="BF",
    knn_matching=True,
    num_features=500,
    num_matches=10,
    distance_threshold=30,
    stop_threshold=1,
    num_track_actions=50,
    num_threads=4,
)
benchmark = dict(
    device='cpu',
    dtype="np.float32",
)

