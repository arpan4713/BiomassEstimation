# import argparse, os, sys
# import numpy as np
# import imageio
# from scipy import ndimage

# import torch
# from torchvision.utils import save_image

# from models.VAE import VAE
# from models.AE import AE

# from utils import get_interpolations

# parser = argparse.ArgumentParser(
#         description='Main function to call training for different AutoEncoders')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=42, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--results_path', type=str, default='results/', metavar='N',
#                     help='Where to store images')
# parser.add_argument('--model', type=str, default='AE', metavar='N',
#                     help='Which architecture to use')
# parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
#                     help='Which dataset to use')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)

# vae = VAE(args)
# ae = AE(args)
# architectures = {'AE':  ae,
#                  'VAE': vae}

# print(args.model)
# if __name__ == "__main__":
#     try:
#         os.stat(args.results_path)
#     except :
#         os.mkdir(args.results_path)

#     try:
#         autoenc = architectures[args.model]
#     except KeyError:
#         print('---------------------------------------------------------')
#         print('Model architecture not supported. ', end='')
#         print('Maybe you can implement it?')
#         print('---------------------------------------------------------')
#         sys.exit()

#     try:
#         for epoch in range(1, args.epochs + 1):
#             autoenc.train(epoch)
#             autoenc.test(epoch)
#     except (KeyboardInterrupt, SystemExit):
#         print("Manual Interruption")

#     with torch.no_grad():
#         images, _ = next(iter(autoenc.test_loader))
#         images = images.to(autoenc.device)
#         images_per_row = 16
#         interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)

#         sample = torch.randn(64, args.embedding_size).to(autoenc.device)
#         sample = autoenc.model.decode(sample).cpu()
#         save_image(sample.view(64, 1, 28, 28),
#                 '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
#         save_image(interpolations.view(-1, 1, 28, 28),
#                 '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset),  nrow=images_per_row)
#         interpolations = interpolations.cpu()
#         interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))
#         interpolations = ndimage.zoom(interpolations, 5, order=1)
#         interpolations *= 256
#         imageio.mimsave('{}/animation_{}_{}.gif'.format(args.results_path, args.model, args.dataset), interpolations.astype(np.uint8))


'''
original is aboveeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
'''

'''

EUROSAT EXP ds

'''

# import argparse
# import os
# import sys
# import numpy as np
# import imageio
# from scipy import ndimage

# import torch
# from torchvision.utils import save_image

# from models.VAE import VAE
# from models.AE import AE

# from utils import get_interpolations

# parser = argparse.ArgumentParser(
#     description='Main function to call training for different AutoEncoders')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=42, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--results_path', type=str, default='results/', metavar='N',
#                     help='Where to store images')
# parser.add_argument('--model', type=str, default='AE', metavar='N',
#                     help='Which architecture to use')
# parser.add_argument('--dataset', type=str, default='EuroSAT', metavar='N',  # Default dataset set to EuroSAT
#                     help='Which dataset to use (MNIST, EMNIST, FashionMNIST, EuroSAT)')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)

# vae = VAE(args)
# ae = AE(args)
# architectures = {'AE': ae,
#                  'VAE': vae}

# print(args.model)
# if __name__ == "__main__":
#     try:
#         os.stat(args.results_path)
#     except:
#         os.mkdir(args.results_path)

#     try:
#         autoenc = architectures[args.model]
#     except KeyError:
#         print('---------------------------------------------------------')
#         print('Model architecture not supported. ', end='')
#         print('Maybe you can implement it?')
#         print('---------------------------------------------------------')
#         sys.exit()

#     try:
#         for epoch in range(1, args.epochs + 1):
#             autoenc.train(epoch)
#             autoenc.test(epoch)
#     except (KeyboardInterrupt, SystemExit):
#         print("Manual Interruption")

#     with torch.no_grad():
#         images, _ = next(iter(autoenc.test_loader))
#         images = images.to(autoenc.device)
#         images_per_row = 16
#         interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)

#         sample = torch.randn(64, args.embedding_size).to(autoenc.device)
#         sample = autoenc.model.decode(sample).cpu()

#         # Adjust image dimensions for EuroSAT (64x64 RGB images)
#         if args.dataset == 'EuroSAT':
#             save_image(sample.view(64, 3, 64, 64),  # 3 channels, 64x64 resolution
#                        '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
#             save_image(interpolations.view(-1, 3, 64, 64),
#                        '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset), nrow=images_per_row)
#         else:
#             save_image(sample.view(64, 1, 28, 28),
#                        '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
#             save_image(interpolations.view(-1, 1, 28, 28),
#                        '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset), nrow=images_per_row)

#         # Create animation
#         interpolations = interpolations.cpu()
#         if args.dataset == 'EuroSAT':
#              interpolations = np.reshape(interpolations.data.numpy(), (-1, 64, 64, 3))  # Reshape for RGB
# else:
#     interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))

# interpolations = ndimage.zoom(interpolations, (1, 5, 5, 1) if args.dataset == 'EuroSAT' else 5, order=1)
# interpolations *= 256

# imageio.mimsave('{}/animation_{}_{}.gif'.format(args.results_path, args.model, args.dataset), interpolations.astype(np.uint8))

       

'''
eurosat cp
'''




# import argparse
# import os
# import sys
# import numpy as np
# import imageio
# from scipy import ndimage

# import torch
# from torchvision.utils import save_image

# from models.VAE import VAE
# from models.AE import AE

# from utils import get_interpolations

# parser = argparse.ArgumentParser(
#     description='Main function to call training for different AutoEncoders')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=42, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--results_path', type=str, default='results/', metavar='N',
#                     help='Where to store images')
# parser.add_argument('--model', type=str, default='AE', metavar='N',
#                     help='Which architecture to use')
# parser.add_argument('--dataset', type=str, default='EuroSAT', metavar='N',  # Default dataset set to EuroSAT
#                     help='Which dataset to use (MNIST, EMNIST, FashionMNIST, EuroSAT)')

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)

# vae = VAE(args)
# ae = AE(args)
# architectures = {'AE': ae,
#                  'VAE': vae}

# print(args.model)
# if __name__ == "__main__":
#     try:
#         os.stat(args.results_path)
#     except:
#         os.mkdir(args.results_path)

#     try:
#         autoenc = architectures[args.model]
#     except KeyError:
#         print('---------------------------------------------------------')
#         print('Model architecture not supported. ', end='')
#         print('Maybe you can implement it?')
#         print('---------------------------------------------------------')
#         sys.exit()

#     try:
#         for epoch in range(1, args.epochs + 1):
#             autoenc.train(epoch)
#             autoenc.test(epoch)
#     except (KeyboardInterrupt, SystemExit):
#         print("Manual Interruption")




# # Ensure correct indentation in conditional statements
# with torch.no_grad():
#     images, _ = next(iter(autoenc.test_loader))
#     images = images.to(autoenc.device)
#     images_per_row = 16
#     interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)

#     sample = torch.randn(64, args.embedding_size).to(autoenc.device)
#     sample = autoenc.model.decode(sample).cpu()

#     # Adjust image dimensions for EuroSAT (64x64 RGB images)
#     if args.dataset == 'EuroSAT':
#         save_image(sample.view(64, 3, 64, 64),  # 3 channels, 64x64 resolution
#                    '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
#         save_image(interpolations.view(-1, 3, 64, 64),
#                    '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset), nrow=images_per_row)
#         interpolations = np.reshape(interpolations.data.numpy(), (-1, 64, 64, 3))  # Reshape for RGB
#     else:
#         save_image(sample.view(64, 1, 28, 28),
#                    '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
#         save_image(interpolations.view(-1, 1, 28, 28),
#                    '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset), nrow=images_per_row)
#         interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))

#     # Proper zoom operation for animations
#     zoom_factor = (1, 5, 5, 1) if args.dataset == 'EuroSAT' else (5, 5)
#     interpolations = ndimage.zoom(interpolations, zoom_factor, order=1)
#     interpolations *= 256

#     imageio.mimsave('{}/animation_{}_{}.gif'.format(args.results_path, args.model, args.dataset),
#                     interpolations.astype(np.uint8))


'''
eurosat new cp

'''

# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.autograd import Variable
# import numpy as np

# def weights_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0.01)

# class FC_Encoder(nn.Module):
#     def __init__(self, latent_dim, input_size=784):
#         super(FC_Encoder, self).__init__()
#         self.fc1 = nn.Linear(input_size, 512)
#         self.fc_mu = nn.Linear(512, latent_dim)
#         self.fc_logvar = nn.Linear(512, latent_dim)
#         self.apply(weights_init)

#     def forward(self, x):
#         h1 = F.relu(self.fc1(x))
#         mu = self.fc_mu(h1)
#         logvar = self.fc_logvar(h1)
#         return mu, logvar

# class FC_Decoder(nn.Module):
#     def __init__(self, latent_dim, output_size=784):
#         super(FC_Decoder, self).__init__()
#         self.fc2 = nn.Linear(latent_dim, 512)
#         self.fc3 = nn.Linear(512, output_size)
#         self.apply(weights_init)

#     def forward(self, z):
#         h2 = F.relu(self.fc2(z))
#         return torch.sigmoid(self.fc3(h2))

# class CNN_Encoder(nn.Module):
#     def __init__(self, latent_dim, input_size=(3, 64, 64)):
#         super(CNN_Encoder, self).__init__()
#         self.input_size = input_size
#         self.channel_mult = 16
#         in_channels = input_size[0]

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, self.channel_mult, 4, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.channel_mult, self.channel_mult*2, 4, 2, 1),
#             nn.BatchNorm2d(self.channel_mult*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
#             nn.BatchNorm2d(self.channel_mult*4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
#             nn.BatchNorm2d(self.channel_mult*8),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#         self.flat_fts = self.get_flat_fts(self.conv)
#         self.fc_mu = nn.Linear(self.flat_fts, latent_dim)
#         self.fc_logvar = nn.Linear(self.flat_fts, latent_dim)
#         self.apply(weights_init)

#     def get_flat_fts(self, fts):
#         with torch.no_grad():
#             f = fts(torch.ones(1, *self.input_size))
#         return int(np.prod(f.size()[1:]))

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(-1, self.flat_fts)
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar

# class CNN_Decoder(nn.Module):
#     def __init__(self, latent_dim, output_size=(3, 64, 64)):
#         super(CNN_Decoder, self).__init__()
#         self.output_size = output_size
#         self.channel_mult = 16
#         out_channels = output_size[0]

#         self.fc = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(True)
#         )

#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(512, self.channel_mult*8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(self.channel_mult*8),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(self.channel_mult*8, self.channel_mult*4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(self.channel_mult*4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(self.channel_mult*2),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(self.channel_mult),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(self.channel_mult, out_channels, 4, 2, 1, bias=False),
#             nn.Sigmoid()
#         )
#         self.apply(weights_init)

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, 512, 1, 1)
#         x = self.deconv(x)
#         return x


'''
eurosat newer cp

'''

import argparse
import os
import sys
import numpy as np
import imageio
from scipy import ndimage

import torch
from torchvision.utils import save_image

from models.VAE import VAE
from models.AE import AE
from utils import get_interpolations

parser = argparse.ArgumentParser(
    description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='embedding size for latent space')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='EuroSAT', metavar='N',
                    help='Which dataset to use (MNIST, EMNIST, FashionMNIST, EuroSAT)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

vae = VAE(args)
ae = AE(args)
architectures = {'AE': ae, 'VAE': vae}

if __name__ == "__main__":
    os.makedirs(args.results_path, exist_ok=True)
    
    if args.model not in architectures:
        print(f'Error: Model {args.model} not supported!')
        sys.exit(1)
    
    autoenc = architectures[args.model]
    
    try:
        for epoch in range(1, args.epochs + 1):
            autoenc.train(epoch)
            autoenc.test(epoch)
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

    # Perform interpolation and save results
    with torch.no_grad():
        images, _ = next(iter(autoenc.test_loader))
        images = images.to(device)
        images_per_row = 16
        interpolations = get_interpolations(args, autoenc.model, device, images, images_per_row)

        sample = torch.randn(64, args.embedding_size).to(device)
        sample = autoenc.model.decode(sample).cpu()

        if args.dataset == 'EuroSAT':
            save_image(sample.view(64, 3, 64, 64), f"{args.results_path}/sample_{args.model}_{args.dataset}.png")
            save_image(interpolations.view(-1, 3, 64, 64), f"{args.results_path}/interpolations_{args.model}_{args.dataset}.png", nrow=images_per_row)
        else:
            save_image(sample.view(64, 1, 28, 28), f"{args.results_path}/sample_{args.model}_{args.dataset}.png")
            save_image(interpolations.view(-1, 1, 28, 28), f"{args.results_path}/interpolations_{args.model}_{args.dataset}.png", nrow=images_per_row)

        interpolations = interpolations.cpu().numpy().astype(np.float32)
        print("Interpolations shape before zoom:", interpolations.shape)
        
        zoom_factor = (1, 1, 5, 5) if args.dataset == 'EuroSAT' else (1, 5, 5, 1)
        interpolations = ndimage.zoom(interpolations, zoom_factor, order=1) * 256

        interpolations = np.clip(interpolations, 0, 255).astype(np.uint8)
        
        if args.dataset == 'EuroSAT':
            interpolations = interpolations.reshape(-1, 64, 64, 3)
        else:
            interpolations = interpolations.reshape(-1, 28, 28)

        print("Final Interpolations shape before saving:", interpolations.shape)
        imageio.mimsave(f"{args.results_path}/animation_{args.model}_{args.dataset}.gif", interpolations)