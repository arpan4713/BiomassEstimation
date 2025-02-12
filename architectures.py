# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.autograd import Variable

# import numpy as  np

# class FC_Encoder(nn.Module):
#     def __init__(self, output_size):
#         super(FC_Encoder, self).__init__()
#         self.fc1 = nn.Linear(784, output_size)

#     def forward(self, x):
#         h1 = F.relu(self.fc1(x))
#         return h1

# class FC_Decoder(nn.Module):
#     def __init__(self, embedding_size):
#         super(FC_Decoder, self).__init__()
#         self.fc3 = nn.Linear(embedding_size, 1024)
#         self.fc4 = nn.Linear(1024, 784)

#     def forward(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

# class CNN_Encoder(nn.Module):
#     def __init__(self, output_size, input_size=(1, 28, 28)):
#         super(CNN_Encoder, self).__init__()

#         self.input_size = input_size
#         self.channel_mult = 16

#         #convolutions
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=1,
#                      out_channels=self.channel_mult*1,
#                      kernel_size=4,
#                      stride=1,
#                      padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1),
#             nn.BatchNorm2d(self.channel_mult*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1),
#             nn.BatchNorm2d(self.channel_mult*4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1),
#             nn.BatchNorm2d(self.channel_mult*8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1),
#             nn.BatchNorm2d(self.channel_mult*16),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#         self.flat_fts = self.get_flat_fts(self.conv)

#         self.linear = nn.Sequential(
#             nn.Linear(self.flat_fts, output_size),
#             nn.BatchNorm1d(output_size),
#             nn.LeakyReLU(0.2),
#         )

#     def get_flat_fts(self, fts):
#         f = fts(Variable(torch.ones(1, *self.input_size)))
#         return int(np.prod(f.size()[1:]))

#     def forward(self, x):
#         x = self.conv(x.view(-1, *self.input_size))
#         x = x.view(-1, self.flat_fts)
#         return self.linear(x)

# class CNN_Decoder(nn.Module):
#     def __init__(self, embedding_size, input_size=(1, 28, 28)):
#         super(CNN_Decoder, self).__init__()
#         self.input_height = 28
#         self.input_width = 28
#         self.input_dim = embedding_size
#         self.channel_mult = 16
#         self.output_channels = 1
#         self.fc_output_dim = 512

#         self.fc = nn.Sequential(
#             nn.Linear(self.input_dim, self.fc_output_dim),
#             nn.BatchNorm1d(self.fc_output_dim),
#             nn.ReLU(True)
#         )

#         self.deconv = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4,
#                                 4, 1, 0, bias=False),
#             nn.BatchNorm2d(self.channel_mult*4),
#             nn.ReLU(True),
#             # state size. self.channel_mult*32 x 4 x 4
#             nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
#                                 3, 2, 1, bias=False),
#             nn.BatchNorm2d(self.channel_mult*2),
#             nn.ReLU(True),
#             # state size. self.channel_mult*16 x 7 x 7
#             nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1,
#                                 4, 2, 1, bias=False),
#             nn.BatchNorm2d(self.channel_mult*1),
#             nn.ReLU(True),
#             # state size. self.channel_mult*8 x 14 x 14
#             nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False),
#             nn.Sigmoid()
#             # state size. self.output_channels x 28 x 28
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, self.fc_output_dim, 1, 1)
#         x = self.deconv(x)
#         return x.view(-1, self.input_width*self.input_height)


'''
above is the original
'''

'''
eurosat cp
'''
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.autograd import Variable
# import numpy as np

# class FC_Encoder(nn.Module):
#     def __init__(self, output_size, input_size=784):
#         super(FC_Encoder, self).__init__()
#         self.fc1 = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         h1 = F.relu(self.fc1(x))
#         return h1

# class FC_Decoder(nn.Module):
#     def __init__(self, embedding_size, output_size=784):
#         super(FC_Decoder, self).__init__()
#         self.fc3 = nn.Linear(embedding_size, 1024)
#         self.fc4 = nn.Linear(1024, output_size)

#     def forward(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

# class CNN_Encoder(nn.Module):
#     def __init__(self, output_size, input_size=(3, 64, 64)):
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
#         self.linear = nn.Sequential(
#             nn.Linear(self.flat_fts, output_size),
#             nn.BatchNorm1d(output_size),
#             nn.LeakyReLU(0.2)
#         )

#     def get_flat_fts(self, fts):
#         f = fts(Variable(torch.ones(1, *self.input_size)))
#         return int(np.prod(f.size()[1:]))

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(-1, self.flat_fts)
#         return self.linear(x)

# class CNN_Decoder(nn.Module):
#     def __init__(self, embedding_size, output_size=(3, 64, 64)):
#         super(CNN_Decoder, self).__init__()
#         self.output_size = output_size
#         self.channel_mult = 16
#         out_channels = output_size[0]

#         self.fc = nn.Sequential(
#             nn.Linear(embedding_size, 512),
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

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, 512, 1, 1)
#         x = self.deconv(x)
#         return x
    

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



'''
 updated

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
#         f = fts(Variable(torch.ones(1, *self.input_size)))
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