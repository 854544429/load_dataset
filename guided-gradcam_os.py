import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import cv2
from torchvision import models
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model = models.resnet50(weights=None)  
in_features = best_model.fc.in_features

best_model.fc = nn.Linear(in_features, 1) 
best_model = best_model.to(device)

state_dict = torch.load('')
del state_dict['fc.weight']
del state_dict['fc.bias']

best_model.load_state_dict(state_dict, strict=False)


best_model.load_state_dict(torch.load(''))
best_model.eval() 
img_path = 'F:/'
combined_image = tifffile.imread(img_path)

print("Original image shape:", combined_image.shape) 

combined_image = combined_image.astype(np.float32) / 65535.0  

img_pil = Image.fromarray(np.uint8(combined_image.transpose(1, 2, 0) * 255))  

preprocess = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
print("Input tensor shape:", input_tensor.shape)

best_model.eval()  
output = best_model(input_tensor)
print("Model output:", output)


class_idx = 0  

target_layers = [best_model.layer4[2]]  
grad_cam = GradCAM(model=best_model, target_layers=target_layers)

targets = [ClassifierOutputTarget(class_idx)]
grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)[0]  

grayscale_cam_resized = cv2.resize(grayscale_cam, (combined_image.shape[2], combined_image.shape[1]))
threshold = 0.2 
grayscale_cam_resized[grayscale_cam_resized < threshold] = 0 
cam_image = show_cam_on_image(combined_image.transpose(1, 2, 0), grayscale_cam_resized, use_rgb=True)
#cam_image = cv2.applyColorMap(np.uint8(255 * grayscale_cam_resized), cv2.COLORMAP_JET)

plt.figure(figsize=(12, 6))
plt.imshow(cam_image)
plt.title('Grad-CAM')
plt.axis('off')
plt.show()








