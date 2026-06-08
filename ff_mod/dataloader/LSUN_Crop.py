from torchvision.datasets import LSUN
from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten, _base_normalization

import torch
import torchvision.transforms as transforms


class LSUN_Crop_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size=64, **kwargs):
        super().__init__(batch_size=batch_size)
        
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR10 normalization

        ])
    
    def load_dataloader(self, download=True, split=None, **kwargs):
        dataset = LSUN(
            './data/lsun',
            transform=self.transform,
            classes=split
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
