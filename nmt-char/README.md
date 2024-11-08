# Fully Character-level Attentional Neural Machine Translation in PyTorch

This is an implementation of fully character-level attentional neural machine
translation, mostly following the models by [Lee et al.,
2017](https://arxiv.org/abs/1610.03017) and [Luong et al.,
2015](https://arxiv.org/abs/1508.04025).

## Requirements

In order to install the required libraries you can run

    conda env create -f environment.yaml

## Running

To train a model, run the command

    python nmt.py --config_path configs/char.yaml --mode train

and to decode

    python nmt.py --config_path configs/char.yaml --mode decode --decode_set test

This writes the output to the file `experiments/char/test_decode_beam.txt`.

## Computing BLEU

For the output of the character-level model we compute BLEU a little
differently:

    perl multi-bleu.perl data/test.de-en.en < <(scripts/tokenizer.perl -l en < experiments/char/test_decode_beam.txt)

This is because we detokenize the dataset for this model, and the output is
then also detokenized. For the BLEU to be comparable to that of a word-level
model we tokenize the output of this model and compute BLEU against a tokenized
reference.
