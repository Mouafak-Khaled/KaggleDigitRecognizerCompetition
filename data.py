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

class DigitMnistDataset(Dataset):
    
    DEFAULT_PATH = os.path.join(os.getcwd(), 'dataset')
    
    def __init__(self, mode, path=None, transforms=None):
        
        self.mode = mode.lower()
        self.check_mode()
        self.path = self.data_path(path)
        self.transforms = transforms
        if self.mode == 'train':
            self.images, self.labels = self._data()
        else:
            self.images = self._data()
            
        self._img_shape = (28, 28)
        
        
    def check_mode(self):
        if self.mode not in ['train', 'test']:
            raise ValueError("mode value is invalid!")
    
    
    def plot_label_distribution(self):
        
        if self.mode == 'train':
            figure = plt.figure(figsize=(8, 6))
            sns.histplot(data=self.labels)
            plt.xticks(np.unique(self.labels))
            plt.ylabel("Frequency")
            plt.xlabel("Labels")
            plt.show()
        
    
    def display_sample_images(self):
        
        indecis = np.random.randint(low=0, high=len(self.images), size=2)
        figure, ax = plt.subplots(1, 2)
        ax[0].imshow(self.images[indecis[0]].reshape(self._img_shape))
        ax[1].imshow(self.images[indecis[1]].reshape(self._img_shape))
        plt.show()

    
    def classes(self):
        if self.mode == 'train':
            return np.unique(self.labels)
        
    
    
    def data_path(self, path):
        if path == None:
            path = os.path.join(DigitMnistDataset.DEFAULT_PATH,
                                self.mode + '.csv')
        else:
            path = os.path.join(path, self.mode + '.csv')
            
        return path
    
       
    def _data(self):
        if self.mode == 'train':
            data = pd.read_csv(self.path)
            labels = data.label.to_numpy()
            images = data.drop(columns=['label']).to_numpy()
            return images, labels

        else:
            images = pd.read_csv(self.path).to_numpy()
            return images
         
    
    def __len__(self):
        return len(self.images) 
    
    
    def __getitem__(self, index):
        img = self.images[index].reshape(self._img_shape) 
        # denoised_img = cv2.medianBlur(img.astype('float32'), 3)

        
        if self.transforms:
            img = self.transforms(img)
        
        img = adjust_contrast(img, contrast_factor=2)
        img = adjust_sharpness(img, sharpness_factor=2)
        
        if self.mode == 'train':
            label = self.labels[index]
            return img, label
        else:
            return img
        
        
    