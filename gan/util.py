import argparse
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, default=None)
    parser.add_argument('-n', '--num_workers', type=int, default=0)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config_data = f.read()
    config = yaml.load(config_data)

    logging.basicConfig(
        handlers=[logging.FileHandler(config['log_file'], mode='w'), logging.StreamHandler()],
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    logger.setLevel(getattr(logging, config['log_level'].upper()))

    logger.info('Configuration:\n' + config_data)

    if config['seed']:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    use_gpu = args.gpu is not None and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_gpu else 'cpu')
    logger.info(f'Device: {device}')

    config['num_workers'] = args.num_workers

    os.makedirs('results', exist_ok=True)

    if config['show_results']:
        plt.ion()

    return config, device


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
