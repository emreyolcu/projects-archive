# Deep Convolutional GAN (DCGAN)

This is a PyTorch implementation of DCGAN ([Radford et al.,
2015](http://arxiv.org/abs/1511.06434)) for MNIST.

## Running

The file [`dcgan.py`](dcgan.py) accepts several arguments. For instance, to
train a model with the default configuration on GPU 0 while using 8
subprocesses for data loading, you may run the command:

    python dcgan.py --config_path default.yaml --gpu 0 --num_workers 8

When the `--gpu` argument is not used training is performed on the CPU, and
when `--num_workers` is not used the data is loaded in the main process.

During training, images are generated periodically and saved in the `results`
directory.

## Results

![Results](https://user-images.githubusercontent.com/8561348/50507867-43eee500-0a4d-11e9-9303-f5a76e52269f.png)
