import itertools
import logging
import pdb
import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import util
from torchvision import transforms

logger = logging.getLogger(__name__)


def mlp(input_size, output_size, hidden_sizes, activation, output_activation):
    sizes = [input_size, *hidden_sizes, output_size]
    n = len(sizes)
    layers = (
        (nn.Linear(sizes[i - 1], sizes[i]), activation if i < n - 1 else output_activation)
        for i in range(1, n)
    )
    return nn.Sequential(*itertools.chain(*layers))


def data_sampler(batch_size, num_workers):
    dataset = torchvision.datasets.MNIST(
        root='../data/mnist',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )

    data_iter = util.cycle(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.RandomSampler(dataset, replacement=True),
            num_workers=num_workers,
            drop_last=True,
        )
    )

    return lambda: next(data_iter)[0]


def noise_sampler(batch_size, z_size):
    return lambda: torch.randn((batch_size, z_size))


def main():
    config, device = util.setup()
    logger.setLevel(getattr(logging, config['log_level'].upper()))

    sample_data = data_sampler(config['batch_size'], config['num_workers'])
    sample_noise = noise_sampler(config['batch_size'], config['z_size'])

    x_size = 784

    G = mlp(
        input_size=config['z_size'],
        output_size=x_size,
        hidden_sizes=config['g_hidden_sizes'],
        activation=nn.ReLU(True),
        output_activation=nn.Tanh(),
    ).to(device)

    D = mlp(
        input_size=x_size,
        output_size=1,
        hidden_sizes=config['d_hidden_sizes'],
        activation=nn.LeakyReLU(inplace=True),
        output_activation=nn.Sigmoid(),
    ).to(device)

    g_optim = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    d_optim = optim.Adam(D.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    g_loss_total = d_loss_total = 0

    for i in range(1, config['iterations'] + 1):
        x = sample_data().view(config['batch_size'], x_size).to(device)
        z = sample_noise().to(device)
        d_loss = -torch.mean(torch.log(D(x)) + torch.log(1 - D(G(z))))
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
        d_loss_total += d_loss.item()

        z = sample_noise().to(device)
        g_loss = -torch.mean(torch.log(D(G(z))))
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()
        g_loss_total += g_loss.item()

        if i % config['report_interval'] == 0:
            g_loss_avg = g_loss_total / config['report_interval']
            d_loss_avg = d_loss_total / config['report_interval']
            logger.info(f'Iter: {i:6d}, g_loss: {g_loss_avg:.4f}, d_loss: {d_loss_avg:.4f}')
            g_loss_total = d_loss_total = 0

        if i % config['generate_interval'] == 0:
            with torch.no_grad():
                r, c = config['grid_shape']
                z = torch.randn((r * c, config['z_size'])).to(device)
                img = G(z).cpu().numpy() * 0.5 + 0.5
                img = img.reshape((r, c, 28, 28)).transpose((0, 2, 1, 3)).reshape((r * 28, c * 28))
                k = i // config['generate_interval']
                plt.imsave(f'results/{k}.png', img, vmin=0, vmax=1, cmap='bone')
                if config['show_results']:
                    plt.axis('off')
                    plt.imshow(img, vmin=0, vmax=1, cmap='bone')
                    plt.pause(0.001)


if __name__ == '__main__':
    main()
