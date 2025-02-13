
import numpy as np
import torch

def get_interpolations(args, model, device, images, images_per_row=20):
    model.eval()
    with torch.no_grad():
        def interpolate(t1, t2, num_interps):
            alpha = np.linspace(0, 1, num_interps + 2)
            interps = [(1 - a) * t1 + a * t2 for a in alpha]
            return torch.stack(interps, 0)

        images = images.to(device)
        
        if args.model == 'VAE':
            mu, logvar = model.encode(images)
            embeddings = model.reparameterize(mu, logvar).cpu()
        elif args.model == 'AE':
            embeddings = model.encode(images).cpu()
        else:
            raise ValueError(f"Unsupported model type: {args.model}")
        
        interps = []
        num_samples = embeddings.size(0)

        for i in range(num_samples - 1):
            interp = interpolate(embeddings[i], embeddings[i + 1], images_per_row - 4).to(device)
            interp_dec = model.decode(interp)
            line = torch.cat([images[i].unsqueeze(0), interp_dec, images[i + 1].unsqueeze(0)])
            interps.append(line)
        
        interp = interpolate(embeddings[-1], embeddings[0], images_per_row - 4).to(device)
        interp_dec = model.decode(interp)
        line = torch.cat([images[-1].unsqueeze(0), interp_dec, images[0].unsqueeze(0)])
        interps.append(line)
        
        interps = torch.cat(interps, 0).to(device)
    return interps


'''
python train.py --model VAE --dataset EuroSAT --batch-size 128 --epochs 10 --log-interval 10


'''