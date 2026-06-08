from torchvision.datasets import Flowers102
from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten, _base_normalization

import torch
import torchvision.transforms as transforms


class Flowers102_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size=64, **kwargs):
        super().__init__(batch_size=batch_size)
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Reescalado a 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR10 normalization

        ])
    
    def load_dataloader(self, download=True, split=None, **kwargs):
        dataset = Flowers102(
            './data',
            transform=self.transform,
            split=split,
            download=download  
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
