import torch
import torchvision
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import swanlab
from torch.utils.data import DataLoader
from load_datasets2 import DatasetLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import os

# 定义训练函数
def train(model, device, train_dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    all_labels = []  # 收集所有标签用于AUC计算
    all_outputs = []  # 收集所有输出概率用于AUC计算
    for iter, (inputs, labels, filenames) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算训练集准确度
        total_loss += loss.item()
        predicted = outputs > 0.5  # 使用0.5作为分类阈值
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 保存标签和输出用于AUC计算
        all_labels.append(labels.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    train_accuracy = 100 * correct / total
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    auc_score = roc_auc_score(all_labels, all_outputs)  # 计算AUC

    print(f'Epoch [{epoch}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Loss: {total_loss/len(train_dataloader):.4f}, AUC: {auc_score:.4f}')
    swanlab.log({"train_loss": total_loss/len(train_dataloader), "train_accuracy": train_accuracy, "train_AUC": auc_score})
    return train_accuracy, auc_score, all_labels, all_outputs

# 定义测试函数
def test(model, device, test_dataloader, epoch):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_filenames = []
    with torch.no_grad():
        for iter, (inputs, labels, filenames) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)

            predicted = outputs > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 收集所有标签和输出概率
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            all_filenames.extend(filenames)
    
    test_accuracy = correct / total * 100
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    auc_score = roc_auc_score(all_labels, all_predictions)
    
    print(f'Epoch [{epoch}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%, AUC: {auc_score:.4f}')
    swanlab.log({"test_accuracy": test_accuracy, "AUC": auc_score})
    return test_accuracy, auc_score, all_labels, all_predictions, all_filenames

if __name__ == "__main__":
    num_epochs = 200
    lr = 1e-4  # 学习率
    lambda_l2 = 1e-6  # L2正则化强度
    batch_size = 64
    num_classes = 1  # 二分类任务

    # 设置device
    try:
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
    elif use_mps:
        device = "mps"
    else:
        device = "cpu"

    # 初始化swanlab
    swanlab.init(
        experiment_name="ResNet50",
        description="Train ResNet50 for Tumor Budding classification.",
        config={
            "model": "resnet50",
            "optim": "Adam",
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_class": num_classes,
            "device": device,
            "lambda_l2": lambda_l2
        },
        logdir="./logs",
    )
    TrainDataset = DatasetLoader("datasets_cutoff/train2.csv")
    ValDataset = DatasetLoader("datasets_cutoff/val2.csv")
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=64, shuffle=False)

    # 载入ResNet50模型
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # 将全连接层替换为2分类
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # 早停机制
    early_stopping_rounds = 50  # 若连续50个epoch评估指标不提升，则停止训练
    early_stopping_counter = 0
    #初始化最佳AUC
    best_auc = 0.0

    # 开始训练
    best_accuracy = 0.0  # 最高训练集准确度
    test_accuracies = []
    all_labels = []
    all_predictions = []
    for epoch in range(1, num_epochs + 1):
        train_accuracy, train_auc, train_labels, train_outputs = train(model, device, TrainDataLoader, optimizer, criterion, epoch)  # Train for one epoch
        test_accuracy, auc_score, labels, predictions, filenames = test(model, device, ValDataLoader, epoch)
        scheduler.step()
        test_accuracies.append(test_accuracy)
        all_labels.extend(labels)
        all_predictions.extend(predictions)

        # 保存权重文件
        if auc_score > best_auc:
            best_auc = auc_score
            early_stopping_counter = 0
            # Save the best model
            best_model_path = 'checkpoint/best_model.pth'
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved new best model with best auc: {best_auc:.4f}')
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= early_stopping_rounds:
            print(f'Early stopping at epoch {epoch}')
            break

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    final_auc_score = roc_auc_score(all_labels, all_predictions)
    print(f'Final AUC score: {final_auc_score:.4f}')
    swanlab.log({"final_AUC": final_auc_score})
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {final_auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    swanlab.log({"example": swanlab.Image(plt)})
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o', linestyle='-', color='r')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    swanlab.log({"example": swanlab.Image(plt)})
    print("Training complete")