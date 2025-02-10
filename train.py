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
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='EuroSAT', metavar='N',  # Default dataset set to EuroSAT
                    help='Which dataset to use (MNIST, EMNIST, FashionMNIST, EuroSAT)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

vae = VAE(args)
ae = AE(args)
architectures = {'AE': ae,
                 'VAE': vae}

print(args.model)
if __name__ == "__main__":
    try:
        os.stat(args.results_path)
    except:
        os.mkdir(args.results_path)

    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('Maybe you can implement it?')
        print('---------------------------------------------------------')
        sys.exit()

    try:
        for epoch in range(1, args.epochs + 1):
            autoenc.train(epoch)
            autoenc.test(epoch)
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")




# Ensure correct indentation in conditional statements
with torch.no_grad():
    images, _ = next(iter(autoenc.test_loader))
    images = images.to(autoenc.device)
    images_per_row = 16
    interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)

    sample = torch.randn(64, args.embedding_size).to(autoenc.device)
    sample = autoenc.model.decode(sample).cpu()

    # Adjust image dimensions for EuroSAT (64x64 RGB images)
    if args.dataset == 'EuroSAT':
        save_image(sample.view(64, 3, 64, 64),  # 3 channels, 64x64 resolution
                   '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
        save_image(interpolations.view(-1, 3, 64, 64),
                   '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset), nrow=images_per_row)
        interpolations = np.reshape(interpolations.data.numpy(), (-1, 64, 64, 3))  # Reshape for RGB
    else:
        save_image(sample.view(64, 1, 28, 28),
                   '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))
        save_image(interpolations.view(-1, 1, 28, 28),
                   '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset), nrow=images_per_row)
        interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))

    # Proper zoom operation for animations
    zoom_factor = (1, 5, 5, 1) if args.dataset == 'EuroSAT' else (5, 5)
    interpolations = ndimage.zoom(interpolations, zoom_factor, order=1)
    interpolations *= 256

    imageio.mimsave('{}/animation_{}_{}.gif'.format(args.results_path, args.model, args.dataset),
                    interpolations.astype(np.uint8))
