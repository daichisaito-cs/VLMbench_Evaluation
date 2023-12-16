import torch
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np


def train_model(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    num_zeros, num_ones = 0, 0
    true_predicted, false_predicted = 0, 0

    total_train_loss = 0
    correct, total = 0, 0
    TP, FP, FN = 0, 0, 0
    for batch_idx, (images, text, ada, image_paths, target) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
    ):
        images, ada, target = images.to(device), ada.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(text, images, ada)
        _, predicted = torch.max(output, 1)
        true_predicted += (predicted == 1).sum().item()
        false_predicted += (predicted == 0).sum().item()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total += target.size(0)
        correct += (predicted == target).sum().item()
        TP += ((predicted == 1) & (target == 1)).sum().item()
        FP += ((predicted == 1) & (target == 0)).sum().item()
        FN += ((predicted == 0) & (target == 1)).sum().item()

        if (epoch == 0 and batch_idx >= 100) or epoch > 0:
            wandb.log({"train_loss": loss.item()})

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = 100 * correct / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"--train-- Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f}")
    print(f"--train-- Epoch {epoch}: Train Accuracy: {train_acc:.2f}%")
    print(f"--train-- Epoch {epoch}: F1 Score: {F1:.4f}")
    print(
        f"--train-- Epoch {epoch}: True Predicted: {true_predicted}, False Predicted: {false_predicted}"
    )
    print(
        f"--train-- Epoch {epoch}: Number of Zeros: {num_zeros}, Number of Ones: {num_ones}"
    )
