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
        full_path = os.path.join(self.current_dir, 'datasets_cutoff', image_path)
        image = tiff.imread(full_path)  # 使用tifffile读取图像
        # 确保图像为 float32 类型，同时归一化到 0-1 范围
        image = image.astype(np.float32) / 65535.0

        # 重排维度以适应 PyTorch：从 (H, W, C) 到 (C, H, W)
        if image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        
        # 创建Tensor
        image = torch.from_numpy(image)
        
        #应用transforms，确保所有操作都在PyTorch上进行
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整图像大小
            #transforms.RandomHorizontalFlip(0.5),#水平翻转
            #transforms.RandomVerticalFlip(0.5),  # 垂直翻转
            #transforms.RandomRotation(degrees=15),  # 随机旋转
            #transforms.ColorJitter(brightness=0.2),  # 亮度调整
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])
        image = transform(image)  # 应用转换
        
        return image
    
    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = self.preprocess_image(image_path)
        return image, int(label), image_path

    def __len__(self):
        return len(self.data)
    



