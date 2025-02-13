

'''
cp eurosat
'''

# import torch
# import torch.utils.data
# from torch import nn, optim
# from torch.nn import functional as F

# import sys
# sys.path.append('../')
# from architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder
# from datasets import get_dataloader

# class Network(nn.Module):
#     def __init__(self, args):
#         super(Network, self).__init__()
#         output_size = 512
#         self.encoder = CNN_Encoder(output_size)
#         self.var = nn.Linear(output_size, args.embedding_size)
#         self.mu = nn.Linear(output_size, args.embedding_size)
#         self.decoder = CNN_Decoder(args.embedding_size)

#     def encode(self, x):
#         x = self.encoder(x)
#         return self.mu(x), self.var(x)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# class VAE(object):
#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device("cuda" if args.cuda else "cpu")
#         self.train_loader = get_dataloader(self.args.dataset, self.args.batch_size)

#         self.model = Network(args).to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

#     def loss_function(self, recon_x, x, mu, logvar):
#         BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         return BCE + KLD

#     def train(self, epoch):
#         self.model.train()
#         train_loss = 0
#         for batch_idx, (data, _) in enumerate(self.train_loader):
#             data = data.to(self.device)
#             self.optimizer.zero_grad()
#             recon_batch, mu, logvar = self.model(data)
#             loss = self.loss_function(recon_batch, data, mu, logvar)
#             loss.backward()
#             train_loss += loss.item()
#             self.optimizer.step()
#             if batch_idx % self.args.log_interval == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(self.train_loader.dataset),
#                     100. * batch_idx / len(self.train_loader),
#                     loss.item() / len(data)))

#         print('====> Epoch: {} Average loss: {:.4f}'.format(
#               epoch, train_loss / len(self.train_loader.dataset)))

#     def test(self, epoch):
#         self.model.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for i, (data, _) in enumerate(self.train_loader):
#                 data = data.to(self.device)
#                 recon_batch, mu, logvar = self.model(data)
#                 test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

#         test_loss /= len(self.train_loader.dataset)
#         print('====> Test set loss: {:.4f}'.format(test_loss))

'''
ds eurosat

'''
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import sys
sys.path.append('../')
from architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder
from datasets import get_dataloader

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.encoder = CNN_Encoder(args.embedding_size)
        self.decoder = CNN_Decoder(args.embedding_size)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        # Load data
        self.train_loader = get_dataloader(self.args.dataset, self.args.batch_size, train=True)
        self.test_loader = get_dataloader(self.args.dataset, self.args.batch_size, train=False)

        # Initialize model and optimizer
        self.model = Network(args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (Binary Cross-Entropy for normalized data)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        return BCE + KLD

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)  # Data is already normalized in get_dataloader
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        avg_loss = train_loss / len(self.train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)  # Data is already normalized in get_dataloader
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

'''
new eurosat cp

'''

# import torch
# import torch.utils.data
# from torch import nn, optim
# from torch.nn import functional as F

# import sys
# sys.path.append('../')
# from architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder
# from datasets import get_dataloader

# class Network(nn.Module):
#     def __init__(self, args):
#         super(Network, self).__init__()
#         self.encoder = CNN_Encoder(args.embedding_size)
#         self.decoder = CNN_Decoder(args.embedding_size)

#     def encode(self, x):
#         mu, logvar = self.encoder(x)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# class VAE(object):
#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device("cuda" if args.cuda else "cpu")
        
#         # Load data
#         self.train_loader = get_dataloader(self.args.dataset, self.args.batch_size, train=True)
#         self.test_loader = get_dataloader(self.args.dataset, self.args.batch_size, train=False)

#         # Initialize model and optimizer
#         self.model = Network(args).to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)  # Reduced LR for stability

#     def loss_function(self, recon_x, x, mu, logvar):
#         # Use MSE loss for continuous data
#         MSE = F.mse_loss(recon_x, x, reduction='sum')
        
#         # KL divergence
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
#         # KL annealing for stability
#         beta = min(1, self.args.epoch / 10)  # Slowly increase weight over epochs
        
#         return MSE + beta * KLD

#     def train(self, epoch):
#         self.model.train()
#         train_loss = 0
#         for batch_idx, (data, _) in enumerate(self.train_loader):
#             data = data.to(self.device)
#             self.optimizer.zero_grad()
#             recon_batch, mu, logvar = self.model(data)
#             loss = self.loss_function(recon_batch, data, mu, logvar)
#             loss.backward()
#             train_loss += loss.item()
#             self.optimizer.step()

#             if batch_idx % self.args.log_interval == 0:
#                 print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}'
#                       f' ({100. * batch_idx / len(self.train_loader):.0f}%)]	Loss: {loss.item() / len(data):.6f}')

#         avg_loss = train_loss / len(self.train_loader.dataset)
#         print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

#     def test(self, epoch):
#         self.model.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for data, _ in self.test_loader:
#                 data = data.to(self.device)
#                 recon_batch, mu, logvar = self.model(data)
#                 test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

#         test_loss /= len(self.test_loader.dataset)
#         print(f'====> Test set loss: {test_loss:.4f}')
