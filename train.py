import argparse
import json
import os
from datetime import datetime


from ff_mod.networks.spiking_network import SpikingNetwork
from ff_mod.networks.base_network import AnalogNetwork

from ff_mod.dataloader.factory import DataloaderFactory

from ff_mod.overlay import AppendToEndOverlay
from ff_mod.loss.loss import VectorBCELoss

from ff_mod.trainer import Trainer

from ff_mod.callbacks.accuracy_writer import AccuracyWriter

import torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter

from ff_mod.utils import save_experiment_info, create_network

EXPERIMENTAL_FOLDER = "experiments/train"

CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")


import torch
from torch import nn


class ConvNetSequential(nn.Module):
    def __init__(self):
        super(ConvNetSequential, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    

# EXtract until last conv layer
class FeatureExtractor(torch.nn.Module):
    def __init__(self, feature_extractor, is_not_resnet = False, output_size = 512):
        super(FeatureExtractor, self).__init__()
        
        self.features = torch.nn.Sequential(
            *list(feature_extractor.children())[:-2]
        )
        if is_not_resnet:
            self.features = torch.nn.Sequential(
                *list(feature_extractor.model)[:-2]
            )
        
        self.flatt = torch.nn.Flatten()
        self.bn = torch.nn.BatchNorm1d(output_size)
        
        print(self.features)

    def forward(self, x):
        x_1 =  self.features(x)
        x_2 = self.flatt(x_1)
        x_2 = self.bn(x_2)
        
        return x_2

def get_resnet18(old_save = None):
    feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    if old_save is not None:
        feature_extractor.load_state_dict(torch.load(old_save))
    
    feature_extractor.eval()
    
    feature_extractor_final = FeatureExtractor(feature_extractor)
    feature_extractor_final.cuda()
    
    return feature_extractor_final

def get_small_resnet18(dataset):
    feature_extractor = ConvNetSequential()
    # Load features/resnet_svhn_small.pth
    feature_extractor.load_state_dict(torch.load(f"features/resnet_{dataset}_small.pth"))
    
    feature_extractor.eval()
    
    feature_extractor_final = FeatureExtractor(feature_extractor, is_not_resnet=True)
    feature_extractor_final.cuda()
    
    return feature_extractor_final

def train(network, args):
    if args['feature_extractor'] == "ResNet":
        feature_extractor = get_resnet18()
    elif args['feature_extractor'] == "ResNetSVHN":
        print("Using SVHN")
        feature_extractor = get_resnet18(old_save="features/resnet_svhn.pth")
    elif args['feature_extractor'] == "SmallResNetSVHN":
        feature_extractor = get_small_resnet18("svhn")
    elif args['feature_extractor'] == "SmallResNetCIFAR":
        feature_extractor = get_small_resnet18("cifar10")

    trainer = Trainer(device = 'cuda:0', greedy_goodness = args["greedy_goodness"])
    trainer.load_data_loaders(args["dataset"], batch_size = args["batch_size"], test_batch_size = args["batch_size"], resize=(args["dataset_resize"], args["dataset_resize"]))
    trainer.set_network(network)
    trainer.set_feature_extractors(feature_extractor)

    experiment_name = f"{args['dataset']}_{'SNN' if args['use_snn'] else 'ANN'}_({args['neurons_per_layer']})_{CURRENT_TIMESTAMP}/"
    if args["old_save_path"] is not None:
        experiment_name = args["old_save_path"].split("/")[-1]
    trainer.set_dynamic_save(f"./{EXPERIMENTAL_FOLDER}/{experiment_name}/model")
    
    writer = SummaryWriter(f"{EXPERIMENTAL_FOLDER}/{experiment_name}/summary/" )
    
    trainer.add_callback(AccuracyWriter(tensorboard=writer))
    trainer.train(epochs=args["epochs"], verbose=1)

def main():
    parser = argparse.ArgumentParser(description="Save experimental information to a JSON file")
    
    # Training Arguments
    parser.add_argument("--dataset", default="mnist", help="Dataset to use")
    parser.add_argument("--dataset_resize", type=int, default=28, help="Dataset to use")
    
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--greedy_goodness",default=False, action=argparse.BooleanOptionalAction, help="Whether to use greedy goodness")
    
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--use_snn", default=False, action=argparse.BooleanOptionalAction, help="Use spiking networks")
    
    # Overlay Arguments
    parser.add_argument("--pattern_size", type=int, default=100, help="Pattern size")
    parser.add_argument("--num_vectors", type=int, default=1, help="Number of vectors per class")
    parser.add_argument("--p", type=float, default=0.1, help="Percentaje of ones in the vectors")
    
    # Network Argument
    parser.add_argument("--neurons_per_layer", type=int, default=1000, help="Number of neurons per layer")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of steps for the spiking network")
    parser.add_argument("--internal_epoch", type=int, default=10, help="Number of epochs for the spiking network")
    parser.add_argument("--save_activity", default=False, action=argparse.BooleanOptionalAction, help="Save activity of the network")
    parser.add_argument("--input_size", type=int, default=784, help="Input size of the network")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layer of the network")
    parser.add_argument("--bounded_goodness", default=False, action=argparse.BooleanOptionalAction, help="Activate bounded goodness in SNN.")
    
    # Loss Arguments
    parser.add_argument("--loss", default="VectorBCELoss", help="Loss function to use")
    parser.add_argument("--threshold", type=float, default=6, help="Threshold for the loss function")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha for the loss function")
    parser.add_argument("--beta", type=float, default=1, help="Beta for the loss function")
    parser.add_argument("--negative_threshold", type=float, default=2, help="Negative threshold for the loss function")
    
    parser.add_argument("--feature_extractor", default=None, help="Feature extractor to use")   
    parser.add_argument("--is_symmetric", default=False, action=argparse.BooleanOptionalAction, help="Is symmetric")
    
    parser.add_argument("--old_save_path", default=None, help="Old save path")
    
    args = vars(parser.parse_args())

    if args["old_save_path"] is None:
        save_experiment_info(args, EXPERIMENTAL_FOLDER, CURRENT_TIMESTAMP)
    
    network = create_network(args)
    
    if args["old_save_path"] is not None:
        network.load_network(args["old_save_path"]+"/model")
        
        with open(f"{args['old_save_path']}/modeltrain_info.json") as file:
            train_info = json.load(file)
            
            
        args['epochs'] = args['epochs'] - train_info['epoch']
        print(f"Continuing training from epoch {args['epochs']}")
    
    train(network, args)

if __name__ == "__main__":
    main()
