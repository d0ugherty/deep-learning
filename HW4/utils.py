import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder


class DatasetUtils:
    def __init__(self, data_loader):
        """
        Initialize the DatasetStatistics class with a DataLoader object.
        
        Parameters:
            data_loader (DataLoader): PyTorch DataLoader object containing the dataset.
        """
        self.data_loader = data_loader
        self.N = len(self.data_loader.dataset)

    """ Just a helper functions for internal use"""
    def _create_transforms(is_test, mean, std):
        if(is_test):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=mean, std=std)
            ])
        return transform


    def calculate_mean(self):
        """
        Calculate the mean of the dataset across each channel.
        
        Returns:
            mean (torch.Tensor): Mean of the dataset across each channel.
        """
        mean = 0.0
        for images, _ in self.data_loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean /= self.N
        return mean

    def calculate_std(self, mean):
        """
        Calculate the standard deviation of the dataset using precomputed mean.
        
        Parameters:
            mean (torch.Tensor): Precomputed mean of the dataset across each channel.
            
        Returns:
            std (torch.Tensor): Standard deviation of the dataset across each channel.
        """
        std = 0.0
        for images, _ in self.data_loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            std += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        std = torch.sqrt(std / (self.N * images.size(2)))
        return std
    

    @staticmethod
    def create_loader_and_transform(root_path, loader_func, extensions,
                                     batch_size=32,shuffle=True,is_test=True):
        
        # Step 1: Create initial DataLoader with ToTensor() transform for the entire dataset
        initial_dataset = DatasetFolder(root=root_path, loader=loader_func,
                                        extensions=extensions, transform=transforms.ToTensor())
        
        initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=shuffle)
    
        if(is_test):
            return DatasetUtils.create_test_loader(initial_dataset, initial_loader)
        else:    
            # Step 2: Randomly split the dataset into training and validation sets
            train_set, val_set = random_split(initial_dataset, [1200, 600])
            
            # Step 3: Create DataLoaders for training and validation sets
            
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)  # Usually, we don't shuffle the validation set

            # Step 4: Calculate statistics for the training set
            stats_calc = DatasetUtils(train_loader)
            mean = stats_calc.calculate_mean()
            std = stats_calc.calculate_std(mean)

            # Step 5: Create DataLoaders with full transform pipeline for both training and validation sets
            final_transform = DatasetUtils._create_transforms(is_test, mean, std)
            train_set.dataset.transform = final_transform
            val_set.dataset.transform = final_transform

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader, mean, std
        
    
    @staticmethod
    def create_test_loader_and_transform(init_dset, initial_loader):

        stats_calc = DatasetUtils(initial_loader)
        mean = stats_calc.calculate_mean()
        std = stats_calc.calculate_std(mean)

        final_transform = DatasetUtils.create_transforms(True, mean, std)
        
        final_dataset = DatasetFolder(root=init_dset.root_path, loader=init_dset.loader_func,
                                        extensions=init_dset.extensions, transform=final_transform)
        
        final_loader = DataLoader(final_dataset, batch_size=init_dset.batch_size, shuffle=False)

        return final_loader, mean, std
   