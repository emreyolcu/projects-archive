# Attentional Neural Machine Translation in PyTorch

This is an implementation of attentional neural machine translation, mostly
following the model by [Luong et al., 2015](https://arxiv.org/abs/1508.04025).

## Requirements

In order to install the required libraries you can run

    conda env create -f environment.yaml

## Running

To train a model, run the command

    python nmt.py --config_path configs/default.yaml --mode train

and to decode

    python nmt.py --config_path configs/default.yaml --mode decode --decode_set <test|dev>

This writes the output to the file `experiments/default/decode.txt`.
