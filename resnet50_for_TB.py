import torch
import torchvision
from torchvision.models import ResNet50_Weights
import torchvision.models as models
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


def train(model, device, train_dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    all_labels = []  
    all_outputs = []  
    for iter, (inputs, labels, filenames) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        
        total_loss += loss.item()
        predicted = outputs > 0.5  
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.append(labels.detach().cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    train_accuracy = 100 * correct / total
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    auc_score = roc_auc_score(all_labels, all_outputs)  

    print(f'Epoch [{epoch}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Loss: {total_loss/len(train_dataloader):.4f}, AUC: {auc_score:.4f}')
    return train_accuracy, auc_score, all_labels, all_outputs


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
            
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            all_filenames.extend(filenames)
    
    test_accuracy = correct / total * 100
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    auc_score = roc_auc_score(all_labels, all_predictions)
    
    print(f'Epoch [{epoch}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%, AUC: {auc_score:.4f}')
    return test_accuracy, auc_score, all_labels, all_predictions, all_filenames

if __name__ == "__main__":
    num_epochs = 200
    lr = 1e-4  
    lambda_l2 = 1e-6  
    batch_size = 64
    num_classes = 1  

   
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

   
    TrainDataset = DatasetLoader("datasets")
    ValDataset = DatasetLoader("datasets")
    TrainDataLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True)
    ValDataLoader = DataLoader(ValDataset, batch_size=64, shuffle=False)

    
    model = torchvision.models.resnet50(weights=ResNet50_Weights)
    
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    model.to(device)

   
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
  
    early_stopping_rounds = 50  
    early_stopping_counter = 0
   
    best_auc = 0.0


    best_accuracy = 0.0  
    test_accuracies = []
    all_labels = []
    all_predictions = []
    for epoch in range(1, num_epochs + 1):
        train_accuracy, train_auc, train_labels, train_outputs = train(model, device, TrainDataLoader, optimizer, criterion, epoch) 
        test_accuracy, auc_score, labels, predictions, filenames = test(model, device, ValDataLoader, epoch)
        scheduler.step()
        test_accuracies.append(test_accuracy)
        all_labels.extend(labels)
        all_predictions.extend(predictions)

    
        if auc_score > best_auc:
            best_auc = auc_score
            early_stopping_counter = 0
     
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
