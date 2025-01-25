import csv
import os
import numpy as np
import torch
import tifffile as tiff
from torchvision import transforms
from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(self, csv_path):
        self.csv_file = csv_path
        with open(self.csv_file, 'r') as file:
            self.data = list(csv.reader(file))
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def preprocess_image(self, image_path):
        full_path = os.path.join(self.current_dir, 'datasets', image_path)
        image = tiff.imread(full_path)  
        image = image.astype(np.float32) / 65535.0

        if image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        
  
        image = torch.from_numpy(image)
    
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),  
            transforms.RandomRotation(degrees=15),  
            transforms.ColorJitter(brightness=0.2),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        image = transform(image)  
        
        return image
    
    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = self.preprocess_image(image_path)
        return image, int(label), image_path

    def __len__(self):
        return len(self.data)
    



