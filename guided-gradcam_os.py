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

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型，加载预训练模型并修改最后的全连接层
best_model = models.resnet50(weights=None)  # 不加载预训练权重
in_features = best_model.fc.in_features
# 修改全连接层
best_model.fc = nn.Linear(in_features, 1)  # 1类输出，按照您的训练模型输出来设置
best_model = best_model.to(device)
# 加载模型权重并删除 fc 层的权重
state_dict = torch.load('')
del state_dict['fc.weight']
del state_dict['fc.bias']

# 加载其余的权重到模型中
best_model.load_state_dict(state_dict, strict=False)


# 加载您训练好的模型权重
best_model.load_state_dict(torch.load(''))
best_model.eval()  # 切换到评估模式
# 1. 加载并处理16-bit图像
img_path = 'F:/'
combined_image = tifffile.imread(img_path)

# 检查图像形状是否为 3*150*150
print("Original image shape:", combined_image.shape)  # 确认形状为 (3, 150, 150)

# 将 16-bit 转换为 float32，并归一化到 [0, 1] 范围
combined_image = combined_image.astype(np.float32) / 65535.0  # 将 16-bit 数据归一化到 [0, 1]

# 转换为 PIL 格式，以便继续处理
img_pil = Image.fromarray(np.uint8(combined_image.transpose(1, 2, 0) * 255))  # 转换为 8-bit 图像，并调整通道顺序

# 2. 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小以适应模型输入
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 对图像进行预处理，添加 batch 维度
input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
print("Input tensor shape:", input_tensor.shape)

# 3. 输入模型并检查输出
best_model.eval()  # 切换到评估模式
output = best_model(input_tensor)
print("Model output:", output)

# 假设我们要查看类别 0 的 Grad-CAM
class_idx = 0  # 根据模型输出选择类别索引

# 4. 生成 Grad-CAM
target_layers = [best_model.layer4[2]]  # 选择卷积层作为目标层
grad_cam = GradCAM(model=best_model, target_layers=target_layers)

# 获取 Grad-CAM 热力图
targets = [ClassifierOutputTarget(class_idx)]
grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)[0]  # 选择第一个 batch 的结果

# 将热力图叠加在原始图像上
grayscale_cam_resized = cv2.resize(grayscale_cam, (combined_image.shape[2], combined_image.shape[1]))
# 4. 手动调整背景区域
threshold = 0.2 # 设置阈值
grayscale_cam_resized[grayscale_cam_resized < threshold] = 0  # 将低于阈值的区域设置为 0
cam_image = show_cam_on_image(combined_image.transpose(1, 2, 0), grayscale_cam_resized, use_rgb=True)
#cam_image = cv2.applyColorMap(np.uint8(255 * grayscale_cam_resized), cv2.COLORMAP_JET)
# 5. 可视化 Grad-CAM
plt.figure(figsize=(12, 6))
plt.imshow(cam_image)
plt.title('Grad-CAM')
plt.axis('off')
plt.show()








