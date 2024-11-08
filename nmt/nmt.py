import argparse
import logging
import os
import pdb
import sys
import time

import numpy as np
import torch
import torch.optim as optim

import utils
import yaml
from nltk.translate.bleu_score import corpus_bleu
from seq2seq import Seq2seq
from tqdm import tqdm
from utils import batch_iter, read_corpus
from vocab import Vocab
from beam_search import beam_search

logger = logging.getLogger(__name__)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--decode_set', type=str, default='test')
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.load(f)

    os.makedirs(config['exp_dir'], exist_ok=True)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(config['exp_dir'], args.mode + '.log')),
            logging.StreamHandler()],
        level=getattr(logging, config['log_level'].upper())
    )
    logger.info('\n' + yaml.dump(config, default_flow_style=False))

    utils.init_device(config, args.gpu)
    logger.info('Using device: ' + str(utils.DEVICE))

    np.random.seed(config['seed'] * 13 // 7)
    torch.manual_seed(config['seed'])

    return config, args.mode, args.decode_set


def load_data(config):
    train_data_src = read_corpus(config['train_src'], source='src')
    train_data_tgt = read_corpus(config['train_tgt'], source='tgt')
    dev_data_src = read_corpus(config['dev_src'], source='src')
    dev_data_tgt = read_corpus(config['dev_tgt'], source='tgt')
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    vocab = Vocab(train_data, config['vocab_size'], config['vocab_freq_cutoff'])

    return train_data, dev_data, vocab


def train(config):
    train_data, dev_data, vocab = load_data(config)

    model = Seq2seq(vocab, config).to(utils.DEVICE)
    for p in model.parameters():
        p.data.uniform_(-config['uniform_init'], config['uniform_init'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    best_dev_ppl = float('inf')
    model_save_path = os.path.join(config['exp_dir'], 'model.pth')

    for epoch in range(1, config['max_epoch']):
        batch_count = 0
        report_loss = report_words = 0

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=config['batch_size'], shuffle=True):
            loss = model(src_sents, tgt_sents)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            optimizer.step()
            batch_count += 1

            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_loss += loss.item() * tgt_word_num_to_predict
            report_words += tgt_word_num_to_predict

            if batch_count % config['log_every'] == 0:
                avg_loss = report_loss / report_words
                logger.info('Epoch: {:3d}, Batch: {:5d}, Loss: {:.4f}, Perplexity: {:.4f}'.format(
                    epoch, batch_count, avg_loss, np.exp(avg_loss)))
                report_loss = report_words = 0

        model.eval()

        dev_ppl, dev_avg_loss = model.evaluate_ppl(dev_data, batch_size=128)
        logger.info('Validation perplexity: {:.4f}'.format(dev_ppl))

        model.train()

        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            logger.info('Saving best model')
            model.save(model_save_path)


def decode(config, decode_set):
    decode_src = config['decode_' + decode_set + '_src']
    test_data_src = read_corpus(decode_src, source='src')
    model = Seq2seq.load(os.path.join(config['exp_dir'], config['model_file'])).to(utils.DEVICE)
    model.eval()

    with torch.no_grad():
        hypotheses = []
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            h, _ = beam_search(model, src_sent, config['beam_size'], config['max_decoding_time_step'])
            hypotheses.append(h)

    with open(os.path.join(config['exp_dir'], decode_set + config['output_file']), 'w') as f:
        for h in hypotheses:
            f.write(' '.join(h[0][1:-1]) + '\n')


def main():
    config, mode, decode_set = setup()

    if mode == 'train':
        train(config)
    elif mode == 'decode':
        decode(config, decode_set)


if __name__ == '__main__':
    main()
