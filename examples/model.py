import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models


class MobileNetV2(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, num_class)
        )
    def forward(self, x):
        x = self.base_model(x)
        return x


class ViT16(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.base_model.heads[0].in_features
        self.base_model.heads = nn.Sequential(
            nn.Linear(in_features, num_class)
        )
    def forward(self, x):
        x = self.base_model(x)
        return x

class ModelTrainer:
    def train(self, model, train_loader, valid_loader, model_save_directory, num_epochs=100, lr=1e-5, warmup_epochs=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        metrics = {
            "train": {"loss": [], "accuracy": [], "f1": []},
            "valid": {"loss": [], "accuracy": [], "f1": []}
        }

        best_val_loss = None
        best_val_file = None

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # Warm-up scheduler: Linearly increase the learning rate
        def get_lr(epoch):
            if epoch < warmup_epochs:
                return lr * (epoch + 1) / warmup_epochs
            else:
                return lr * (1 - (epoch + 1 - warmup_epochs) / (num_epochs - warmup_epochs))

        for epoch in range(num_epochs):
            model.train()
            running_loss, all_preds, all_labels = 0.0, [], []

            current_lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            
            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            all_preds = np.argmax(all_preds, axis=1)
            metrics["train"]["loss"].append(running_loss / len(train_loader.dataset))
            metrics["train"]["accuracy"].append(accuracy_score(all_labels, all_preds))
            metrics["train"]["f1"].append(f1_score(all_labels, all_preds, average="macro"))
            
            model.eval()
            val_loss, val_preds, val_labels = 0.0, [], []
    
            with torch.no_grad():
                for inputs, labels, _ in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
    
            if best_val_loss is None or best_val_loss > val_loss:
                best_val_loss = val_loss
                best_val_file = f"{model_save_directory}model_{epoch}.pt"
                torch.save(model.state_dict(), best_val_file)

            val_preds = np.argmax(val_preds, axis=1)
            metrics["valid"]["loss"].append(val_loss / len(valid_loader.dataset))
            metrics["valid"]["accuracy"].append(accuracy_score(val_labels, val_preds))
            metrics["valid"]["f1"].append(f1_score(val_labels, val_preds, average="macro"))
    
            print(f'Epoch: {epoch} | Validation Accuracy: {metrics["valid"]["accuracy"][-1]:.4f} | Loss: {metrics["valid"]["loss"][-1]:.4f} | F1: {metrics["valid"]["f1"][-1]:.4f}')

        torch.save(model.state_dict(), Path(model_save_directory) / f"model_last.pt")
        
        plt.figure(figsize=(15, 3))
        epochs = range(num_epochs)
        fontsize=9

        for i, metric in enumerate(["loss", "accuracy", "f1"]):
            plt.subplot(1, 3, i + 1)
            plt.plot(epochs, metrics["train"][metric], label=f'Train {metric.capitalize()}')
            plt.plot(epochs, metrics["valid"][metric], label=f'Valid {metric.capitalize()}')
            plt.xlabel('Epoch', fontsize=fontsize)
            plt.ylabel(metric.capitalize(), fontsize=fontsize)
            plt.title(metric.capitalize(), fontsize=fontsize)
            plt.legend(fontsize=fontsize)
        
        plt.tight_layout()
        plt.show()
        
        print(best_val_file)
        return best_val_file
    