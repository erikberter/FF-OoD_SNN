from torchvision.datasets import Caltech101
from ff_mod.dataloader.base import DataLoaderExtractor, _ttemp_flatten, _base_normalization

import torch
import torchvision.transforms as transforms


#Change Grayscale Image to RGB for the shape
class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':
            img = img.convert("RGB")
        return img

class Caltech101_Dataloader(DataLoaderExtractor):
    
    def __init__(self, batch_size=64, **kwargs):
        super().__init__(batch_size=batch_size)
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Reescalado a 32x32
            GrayscaleToRGB(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR10 normalization

        ])
    
    def load_dataloader(self, download=True, split=None, **kwargs):
        dataset = Caltech101(
            './data',
            transform=self.transform,
            download=download  
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
