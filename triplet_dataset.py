import numpy as np
import pickle
from PIL import Image
import os
import random
from torch.utils.data import Dataset
import torch

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    # Distances in embedding space is calculated in euclidean
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class TripletData(Dataset):
    def __init__(self, path, transforms, dataset_name = 'rparis6k', split="train"):
        self.path = path
        self.split = split    # train or valid
        self.transforms = transforms
        self.metadata = self.load_metadata(path, dataset_name)
        self.classes = self.classes_loader()
        self.cats = len(self.classes)       # number of categories
        if split == 'train':
            self.samples = self.metadata['imlist']
        else:
            self.samples = self.metadata['qimlist']
        
    @staticmethod
    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def classes_loader(self):
        a = [i['easy'] + i['hard'] for i in self.metadata['gnd']]    
        b = []
        previous = []
        for i in a:
            if i[0] not in previous:
                b.append(i)
            previous.append(i[0])
        return b
    @staticmethod
    def load_metadata(root, dataset_name):
        with open(os.path.join(root, f'gnd_{dataset_name}.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        return metadata        

    def __getitem__(self, idx):
        # our positive class for the triplet
        idx = idx%self.cats

        # choosing our pair of positive images (im1, im2)
        im1, im2 = random.sample(self.classes[idx], 2)
        im1 = self.pil_loader(os.path.join(self.path, 'jpg', f'{self.samples[im1]}.jpg'))
        im2 = self.pil_loader(os.path.join(self.path, 'jpg', f'{self.samples[im2]}.jpg'))

        # choosing a negative class and negative image (im3)
        negative_cats = list(range(self.cats))
        negative_cats.remove(idx)
        negative_cat = random.choice(negative_cats)
        im3 = random.choice(self.classes[negative_cat])
        im3 = self.pil_loader(os.path.join(self.path, 'jpg', f'{self.samples[im3]}.jpg'))

        im1 = self.transforms(im1)
        im2 = self.transforms(im2)
        im3 = self.transforms(im3)

        return [im1, im2, im3]
        
    # we'll put some value that we want since there can be far too many triplets possible
    # multiples of the number of images/ number of categories is a good choice
    def __len__(self):
        return len(self.samples)