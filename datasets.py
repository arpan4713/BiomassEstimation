# import torch
# from torchvision import datasets, transforms

# class MNIST(object):
#     def __init__(self, args):
#         kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#         self.train_loader = torch.utils.data.DataLoader(
#             datasets.MNIST('data/mnist', train=True, download=True,
#                            transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)
#         self.test_loader = torch.utils.data.DataLoader(
#             datasets.MNIST('data/mnist', train=False, transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)

# class EMNIST(object):
#     def __init__(self, args):
#         kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#         self.train_loader = torch.utils.data.DataLoader(
#             datasets.EMNIST('data/emnist', train=True, download=True, split='byclass',
#                            transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)
#         self.test_loader = torch.utils.data.DataLoader(
#             datasets.EMNIST('data/emnist', train=False, split='byclass',
#             transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)

# class FashionMNIST(object):
#     def __init__(self, args):
#         kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#         self.train_loader = torch.utils.data.DataLoader(
#             datasets.FashionMNIST('data/fmnist', train=True, download=True,
#                            transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)
#         self.test_loader = torch.utils.data.DataLoader(
#             datasets.FashionMNIST('data/fmnist', train=False, transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)


'''
original is aboveeeeeeeeeeeeeeeeeeee

'''

'''
EUROSAT MODIFIED

'''

# import torch
# from torchvision import datasets, transforms

# class MNIST(object):
#     def __init__(self, args):
#         kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#         self.train_loader = torch.utils.data.DataLoader(
#             datasets.MNIST('data/mnist', train=True, download=True,
#                            transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)
#         self.test_loader = torch.utils.data.DataLoader(
#             datasets.MNIST('data/mnist', train=False, transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)

# class EMNIST(object):
#     def __init__(self, args):
#         kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#         self.train_loader = torch.utils.data.DataLoader(
#             datasets.EMNIST('data/emnist', train=True, download=True, split='byclass',
#                            transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)
#         self.test_loader = torch.utils.data.DataLoader(
#             datasets.EMNIST('data/emnist', train=False, split='byclass',
#             transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)

# class FashionMNIST(object):
#     def __init__(self, args):
#         kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#         self.train_loader = torch.utils.data.DataLoader(
#             datasets.FashionMNIST('data/fmnist', train=True, download=True,
#                            transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)
#         self.test_loader = torch.utils.data.DataLoader(
#             datasets.FashionMNIST('data/fmnist', train=False, transform=transforms.ToTensor()),
#             batch_size=args.batch_size, shuffle=True, **kwargs)

# class EuroSAT(object):
#     def __init__(self, args):
#         kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#         # Define transformations for EuroSAT
#         transform = transforms.Compose([
#             transforms.Resize((64, 64)),  # Resize images to 64x64
#             transforms.ToTensor(),       # Convert to tensor
#             transforms.Normalize(        # Normalize with ImageNet stats
#                 mean=[0.485, 0.456, 0.406],  # RGB mean
#                 std=[0.229, 0.224, 0.225]    # RGB std
#             )
#         ])

#         # Load EuroSAT dataset
#         self.train_loader = torch.utils.data.DataLoader(
#             datasets.EuroSAT(root='data/eurosat', train=True, download=True, transform=transform),
#             batch_size=args.batch_size, shuffle=True, **kwargs
#         )
#         self.test_loader = torch.utils.data.DataLoader(
#             datasets.EuroSAT(root='data/eurosat', train=False, download=True, transform=transform),
#             batch_size=args.batch_size, shuffle=True, **kwargs
#         )





'''
cp eurosat
'''


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os

def get_dataloader(dataset_name, batch_size, num_workers=4):
    """
    Returns a DataLoader for the specified dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Ensure images are 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])
    
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    elif dataset_name == 'EMNIST':
        dataset = datasets.EMNIST(root="data", split='letters', train=True, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    elif dataset_name == 'EuroSAT':
        dataset_path = "data/EuroSAT"
        if not os.path.exists(dataset_path):
            print("Downloading EuroSAT dataset...")
            dataset = datasets.EuroSAT(root="data", download=True, transform=transform)
        else:
            dataset = ImageFolder(root=dataset_path, transform=transform)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
