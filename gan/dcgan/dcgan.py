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


class Generator(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_size, 1024, 3, 1, 0, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 1, 4, 2, 1, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, 3, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(x.size(0))


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
    return lambda: 0.02 * torch.randn((batch_size, z_size, 1, 1))


def main():
    config, device = util.setup()
    logger.setLevel(getattr(logging, config['log_level'].upper()))

    sample_data = data_sampler(config['batch_size'], config['num_workers'])
    sample_noise = noise_sampler(config['batch_size'], config['z_size'])

    G = Generator(config['z_size']).to(device)
    D = Discriminator().to(device)

    g_optim = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    d_optim = optim.Adam(D.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    g_loss_total = d_loss_total = 0

    for i in range(1, config['iterations'] + 1):
        x = sample_data().to(device)
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
                z = torch.randn((r * c, config['z_size'], 1, 1)).to(device)
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
