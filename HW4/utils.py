import torch

class DatasetStatistics:
    def __init__(self, data_loader):
        """
        Initialize the DatasetStatistics class with a DataLoader object.
        
        Parameters:
            data_loader (DataLoader): PyTorch DataLoader object containing the dataset.
        """
        self.data_loader = data_loader
        self.N = len(self.data_loader.dataset)

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