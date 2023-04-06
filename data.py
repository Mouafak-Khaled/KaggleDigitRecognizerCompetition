import cv2
import os
import seaborn as sns
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import adjust_contrast, adjust_sharpness
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from preprocessing import *

from torchvision.datasets import MNIST


class DigitMnistDataset(Dataset):
    
    
    def __init__(self, path, mode, transform=None):
        
        self.path = path       
        self.transform = transform
        self.dataset = MNIST(self.path,
                             download=False,
                             train=(mode.lower() == 'train'),
                             transform=None)

        
    def labels(self):
        return self.dataset.targets.numpy()
    
    
    def plot_label_distribution(self):
        
        figure = plt.figure(figsize=(8, 6))
        sns.histplot(data=self.labels())
        plt.xticks(np.unique(self.labels()))
        plt.ylabel("Frequency")
        plt.xlabel("Labels")
        plt.show()
        
    
    def display_sample_images(self):
        
        indecis = np.random.randint(low=0, high=len(self.dataset), size=2)
        figure, ax = plt.subplots(1, 2)
        ax[0].imshow((self.transform(self.dataset[indecis[0]][0])).permute(1, 2, 0))
        ax[1].imshow((self.transform(self.dataset[indecis[1]][0])).permute(1, 2, 0))
        plt.show()
     
    
    def __len__(self):
        return len(self.dataset) 
    
    
    def __getitem__(self, index):
        img , target= self.dataset[index] 
        # denoised_img = cv2.medianBlur(img.astype('float32'), 3)
        img = adjust_contrast(img, contrast_factor=2)
        img = adjust_sharpness(img, sharpness_factor=2)
        
        if self.transform:
            img = self.transform(img)
        
        
       
        return img, target
        
        
    