# Vanilla GAN

This is a PyTorch implementation of the Vanilla GAN ([Goodfellow et al.,
2014](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)) for
MNIST. A few tricks are used to improve training stability:

- Pixel values are normalized to be in the interval [-1, 1], so the activation
  function at the output layer of the generator is tanh.
- Generator is trained to maximize `log(D(G(z)))` rather than minimize
  `log(1 - D(G(z)))`.
- Adam is used with beta1=0.5 instead of the default 0.9.

## Running

The file [`gan.py`](gan.py) accepts several arguments. For instance, to train a
model with the default configuration on GPU 0 while using 8 subprocesses for
data loading, you may run the command:

    python gan.py --config_path default.yaml --gpu 0 --num_workers 8

When the `--gpu` argument is not used training is performed on the CPU, and
when `--num_workers` is not used the data is loaded in the main process.

During training, images are generated periodically and saved in the `results`
directory.

## Results

![Results](https://user-images.githubusercontent.com/8561348/50394418-57d3d780-072b-11e9-918e-6619356831b9.png)
