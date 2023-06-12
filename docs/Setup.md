# Setup Notes

## Dependencies

- Use pytorch 1.10.1 (for cuda 11.1)
- Pin protobuf to 3.20.1
- Don't install the latest tensorflow2


```Bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -U protobuf==3.20.1
pip install tensorboard==2.8.0
```


## Project

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
