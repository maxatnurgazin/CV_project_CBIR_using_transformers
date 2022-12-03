from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import os
import pickle

class RevisitedDataset(Dataset):
    """
    Args:
        root (string):  Root directory path.
        phase (string): 'database' or 'query'
        setup (string): Default None; if 'query' -> 'easy', 'medium' or 'hard'
        dataset_name (string): rparis6k (default) or roxford5k
        transform (callable, optional): A function/transform that  takes in an PIL image 
                                        and returns a transformed version.
    
     Attributes:
        metadata
    """
    def __init__(self, root, phase, setup=None, dataset_name = 'rparis6k', transform=None) -> None:
        self.root = root
        self.phase = phase
        self.setup = setup
        self.transform = transform
        self.dataset_name = dataset_name
        self.metadata = self.load_metadata(root, dataset_name)

        if phase == 'database':
            self.samples = self.metadata['imlist']
        else:
            self.samples = self.metadata['qimlist']

            split = {}
            if self.setup == 'easy':
                split['ok'] = ['easy']
                split['junk'] = ['junk', 'hard']
            elif self.setup == 'medium':
                split['ok'] = ['easy', 'hard']
                split['junk'] = ['junk']
            else:
                split['ok'] = ['hard']
                split['junk'] = ['easy', 'junk']
            self.split = split

    @staticmethod
    def load_metadata(root, dataset_name):
        with open(os.path.join(root, f'gnd_{dataset_name}.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        return metadata

    @staticmethod
    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        # to avoid crashing for truncated (corrupted images)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        img_path = os.path.join(self.root, 'jpg', f'{self.samples[index]}.jpg')

        img = self.pil_loader(img_path)
        
        if self.phase == 'database':
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            g ={} 
            g['ok']   = np.concatenate([self.metadata['gnd'][index][i] for i in self.split['ok']])
            g['junk'] = np.concatenate([self.metadata['gnd'][index][i] for i in self.split['junk']])
            img = img.crop(self.metadata['gnd'][index]['bbx'])
            
            if self.transform is not None:
                img = self.transform(img)

            return img, g

    def __len__(self) -> int:
        return len(self.samples)