# Setup Notes

Setting up `data` directory:

```Bash
ln -s ~/data2/360_datasets/pano1024x512 sun360
ln -s ~/data2/360_datasets/360-indoor wacv360indoor
```

Setting up `dataset` directory:

```Bash
python tools/make_findview_sun360_alpha_dataset.py --config configs/_base_/datasets/sun360_alpha_indoor.py
```

The directory should be organized like this:

```
dataset
|-- sun360
|   |-- alpha
|   |   |-- indoor
...
```

Setting up `results` directory:

```Bash
ln -s ~/data2/look_around_results/checkpoints checkpoints
ln -s ~/data2/look_around_results/videos videos
ln -s ~/data2/look_around_results/benchmarks benchmarks
```
