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