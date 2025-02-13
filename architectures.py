

'''
      eurosat ds
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

class FC_Encoder(nn.Module):
    def __init__(self, latent_dim, input_size=784):
        super(FC_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc_mu = nn.Linear(512, latent_dim)  # Mean of the latent space
        self.fc_logvar = nn.Linear(512, latent_dim)  # Log variance of the latent space

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)  # Mean
        logvar = self.fc_logvar(h1)  # Log variance
        return mu, logvar

class FC_Decoder(nn.Module):
    def __init__(self, latent_dim, output_size=784):
        super(FC_Decoder, self).__init__()
        self.fc2 = nn.Linear(latent_dim, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, z):
        h2 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h2))  # Output in range [0, 1]

class CNN_Encoder(nn.Module):
    def __init__(self, latent_dim, input_size=(3, 64, 64)):
        super(CNN_Encoder, self).__init__()
        self.input_size = input_size
        self.channel_mult = 16
        in_channels = input_size[0]

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, self.channel_mult, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult, self.channel_mult*2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flat_fts = self.get_flat_fts(self.conv)
        self.fc_mu = nn.Linear(self.flat_fts, latent_dim)  # Mean of the latent space
        self.fc_logvar = nn.Linear(self.flat_fts, latent_dim)  # Log variance of the latent space

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flat_fts)
        mu = self.fc_mu(x)  # Mean
        logvar = self.fc_logvar(x)  # Log variance
        return mu, logvar

class CNN_Decoder(nn.Module):
    def __init__(self, latent_dim, output_size=(3, 64, 64)):
        super(CNN_Decoder, self).__init__()
        self.output_size = output_size
        self.channel_mult = 16
        out_channels = output_size[0]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, self.channel_mult*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_mult*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*8, self.channel_mult*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult, out_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 1, 1)
        x = self.deconv(x)
        return x

