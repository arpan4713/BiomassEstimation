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

def get_dataloader(dataset_name, batch_size, train=True, num_workers=4):
    """
    Returns a DataLoader for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'EuroSAT').
        batch_size (int): Batch size for the DataLoader.
        train (bool): If True, loads the training set. Otherwise, loads the test set.
        num_workers (int): Number of workers for data loading.

    Returns:
        DataLoader: A DataLoader for the specified dataset.
    """
    # Define transformations
    if train:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images to 64x64
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.RandomRotation(10),  # Data augmentation
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize images to 64x64
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

    # Load the dataset
    if dataset_name == 'EuroSAT':
        dataset_path = os.path.join("data", "EuroSAT")
        if not os.path.exists(dataset_path):
            print("Downloading EuroSAT dataset...")
            # Download the dataset if it doesn't exist
            dataset = datasets.EuroSAT(root="data", download=True, transform=transform)
        else:
            # Load the dataset from the specified path
            dataset = ImageFolder(root=dataset_path, transform=transform)
        
        # Split the dataset into training and testing sets
        train_size = int(0.8 * len(dataset))  # 80% for training
        test_size = len(dataset) - train_size  # 20% for testing
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # Select the appropriate dataset based on the 'train' argument
        dataset = train_dataset if train else test_dataset
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if train else False,  # Shuffle only for training
        num_workers=num_workers
    )
    return dataloader