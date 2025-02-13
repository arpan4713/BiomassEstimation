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
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

def get_dataloader(dataset_name, batch_size, train=True, num_workers=4):
    # Define transformations with correct normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    if train:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize  # Normalize to [-1, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize  # Normalize to [-1, 1]
        ])
    
    # Load the dataset
    if dataset_name == 'EuroSAT':
        dataset_path = os.path.join("data", "EuroSAT")
        if not os.path.exists(dataset_path):
            print("Downloading EuroSAT dataset...")
            dataset = datasets.EuroSAT(root="data", download=True, transform=transform)
        else:
            dataset = ImageFolder(root=dataset_path, transform=transform)
        
        # Split dataset into training and testing sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        dataset = train_dataset if train else test_dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # Shuffle only for training
        num_workers=num_workers
    )
    return dataloader
