from abc import abstractmethod
from ff_mod.algorithms.base import BaseOODAlgorithm

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet 

import torch

import logging as log

import os
import json

from tqdm import tqdm

from ff_mod.algorithms.distances import *

class PatternAlgorithm(BaseOODAlgorithm):
    def __init__(self):
        super().__init__()
        self.latent_spaces = None
    
    @torch.no_grad()
    @abstractmethod
    def initial_setup(self, net, in_loader, sample_size: int = 512):
        pass
    
    def get_name(self):
        pass
    
    def save_config(self, path, params):        
        if os.path.isfile(path + "/" + self.__class__.__name__ + ".json"):
            return
        
        with open(path + "/" + self.__class__.__name__ + ".json", 'w') as f:
            json.dump(params, f)

    @abstractmethod
    def get_scores_batch(self, data, net : AbstractForwardForwardNet, **kwargs):
        pass

    @torch.no_grad()
    def get_scores(self, ood_loader, net: AbstractForwardForwardNet, early_stop : int = -1, **kwargs):
        if self.latent_spaces is None:
            raise Exception("Initial patterns were not set. Please call initial_setup first.")
            
        scores = [] 
        for step, (x,y) in tqdm(enumerate(iter(ood_loader)), disable=self.verbose<1, total=len(ood_loader)):
            x, y = x.to(self.device), y.to(self.device)

            scores += self.get_scores_batch(x, net, **kwargs)
            
            if early_stop > 0 and step > early_stop:
                break
            
        return scores 


class PatternOODv3_Geo(PatternAlgorithm):
    
    # TODO Refactor so that dim is automatically inferred from the network
    def __init__(
            self,
            latent_depth: int = 1,
            p = 2,
            inverse_base = 10e4,
            zero_scale = 1.0,
            num_classes = 10,
            feature_extractor = None,
            distance = "manhattan",
            normalize = False,
            bounds = 0.2,
            top_k = 0,
            device = "cpu",
            verbose = 1
        ):
        
        self.latent_depth = latent_depth

        self.feature_extractor = feature_extractor
        
        self.distance = distance
        
        self.p = p
        
        self.device = device
        
        self.verbose = verbose
        
        self.normalize = normalize
        
        self.latent_spaces = None
        self.num_classes = num_classes
        
        self.inverse_base = inverse_base
        self.zero_scale = zero_scale
        self.bounds = bounds
        self.top_k = top_k
        
    
    def __get_latent(self, x, fake_lab, net : AbstractForwardForwardNet, overlay_factor = 1, **kwargs):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
            
        latents = net.get_latent(x, fake_lab, self.latent_depth, overlay_factor = overlay_factor)
        
        if self.normalize:
            latents = (latents + 0.001 * torch.randn_like(latents)) / (latents.norm(dim=1, keepdim=True) + 0.0001)
        
        if len(latents.shape) == 3:
            latents = latents.mean(axis=1)
            
        return latents

    def __get_goodness_scores(self, x, net: AbstractForwardForwardNet):
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)

        goodness_scores = net.predict(x, self.num_classes, return_prev=True)
        return goodness_scores
    
    @torch.no_grad()
    def initial_setup(self, net, in_loader, sample_size: int = 1000):
        dim = net.layers[self.latent_depth].weight.shape[0]

        self.latent_spaces = [[[] for _ in range(self.num_classes)] for _ in range(self.num_classes)] 
        
        for step, (x, y) in  tqdm(enumerate(iter(in_loader)), disable=self.verbose<1):
            x, y = x.to(self.device), y.to(self.device)
            
            for j in range(0,self.num_classes):
                fake_lab = (y+j)%self.num_classes
                
                for i, latent in enumerate(self.__get_latent(x, fake_lab, net)):
                    self.latent_spaces[y[i]][fake_lab[i]] += [latent]
                
            if step * x.shape[0] > sample_size:
                break
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                
                latent = torch.stack(self.latent_spaces[i][j])
            
                if i==j:
                    latent = latent[torch.norm(latent, p=self.p, dim=1).sort()[1][int(latent.shape[0]*self.bounds):]]
                else:
                    latent = latent[torch.norm(latent, p=self.p, dim=1).sort()[1][:int(latent.shape[0]*(1-self.bounds))]]
                    
                self.latent_spaces[i][j] = latent
            
    
    def get_name(self):
        if self.top_k is not None and self.top_k > 0:
            return f"PatternOODv3_Geo_{self.distance}_topk{self.top_k}"
        return f"PatternOODv3_Geo_{self.distance}"

    def save_config(self, path):
        params = {
            'latent_depth': self.latent_depth,
            'distance': self.distance,
            'top_k': self.top_k,
        }
        super().save_config(path, params)
    
    def get_minimal_distance(self, x, group, power_sum = False):
        
        if self.distance == "manhattan":
            dist_temp = torch.cdist(x, group, p=1).min(axis=1)[0]
        elif self.distance == "euclidean":
            dist_temp = torch.cdist(x, group, p=2).min(axis=1)[0]
        elif self.distance == "cosine":
            A_norm = x / (x.norm(dim=1, keepdim=True)+0.0001)
            B_norm = group / (group.norm(dim=1, keepdim=True)+0.0001)

            dist_temp = 1 - torch.mm(A_norm, B_norm.t()).min(axis=1)[0]
        elif self.distance == "kernel":
            distances = torch.cdist(x, group, p=2)
            dist_temp = torch.exp(-distances).mean(axis=1)
        
        elif "knn" in self.distance:
            #Get X the knn_X
            k = int(self.distance.split("_")[1])
            dist_temp = torch.cdist(x, group, p=2).topk(k, largest=False, dim=1).values.mean(axis=1)
            
        return dist_temp
    
    @torch.no_grad()
    def get_scores_batch(self, data, net : AbstractForwardForwardNet, power_sum = False, returl_latent_dist = False, **kwargs):
        
        dist_per_label = torch.zeros((data.shape[0], self.num_classes, self.num_classes)).to(self.device)
        zero_per_label = torch.zeros((data.shape[0], self.num_classes)).to(self.device)
                
        for label in range(self.num_classes):
            latent_vec = self.__get_latent(data, torch.full((data.shape[0],), label, dtype=torch.long), net, **kwargs)
                
            for pos_class in range(self.num_classes):
                dist_per_label[:, pos_class, label] = self.get_minimal_distance(latent_vec, self.latent_spaces[pos_class][label], power_sum=power_sum)
                
            zero_per_label[:, label] = self.get_minimal_distance(latent_vec, torch.zeros_like(latent_vec), power_sum=power_sum)

        top_k = self.num_classes if self.top_k is None else min(max(int(self.top_k), 0), self.num_classes)

        if top_k > 0 and top_k < self.num_classes:
            goodness_scores = self.__get_goodness_scores(data, net)
            topk_idx = goodness_scores.topk(top_k, dim=1).indices

            label_mask = torch.zeros((data.shape[0], self.num_classes), device=self.device)
            label_mask.scatter_(1, topk_idx, 1.0)

            latent_dist = (dist_per_label * label_mask.unsqueeze(1)).sum(dim=2)
            zero_dist = (zero_per_label * label_mask).sum(dim=1)
        else:
            latent_dist = dist_per_label.sum(dim=2)
            zero_dist = zero_per_label.sum(dim=1)
        
        zero_greatest = (self.zero_scale * zero_dist < latent_dist.min(axis=1)[0]).float()
        
        score_init = latent_dist.min(axis=1)[0]
        
        if returl_latent_dist:
            return ( score_init + (self.inverse_base - 2 * score_init)*zero_greatest), latent_dist
        
        return ( score_init + (self.inverse_base - 2 * score_init)*zero_greatest).tolist()
        #return score_init.tolist()
    