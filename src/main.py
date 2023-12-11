from calendar import c
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from torch.utils.data import TensorDataset, DataLoader, random_split
import wandb
from model import *
from torch.utils.data import DataLoader, random_split, ConcatDataset
from utils.data_loader import CustomDataset, create_data_loaders
from utils.utils import torch_fix_seed, save_checkpoint, load_checkpoint, create_checkpoint_dir, find_trainable_layers, init_weights_he_normal, init_weights_he_normal, text_to_ids, FocalLoss
from train_model import train_model
from validate_model import validate_model
from test_model import test_model
import torchvision

def main():
    with open("configs/config.json") as config_file:
        config = json.load(config_file)
    torch_fix_seed(config["seed"])
    lr = config["lr"]
    max_epoch = config["max_epoch"]
    patience = config["patience"]
    batch_size = config["batch_size"]
    NUM_IMAGES = config["input_image_num"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = CustomDataset(config["train_data_path"], NUM_IMAGES=NUM_IMAGES)
    valid_set = CustomDataset(config["valid_data_path"], NUM_IMAGES=NUM_IMAGES)
    test_set = CustomDataset(config["test_data_path"], NUM_IMAGES=NUM_IMAGES)

    train_loader, valid_loader, test_loader = create_data_loaders(train_set, valid_set, test_set, batch_size=batch_size)

    wandb.init(project="flamingo_for_vlmbench", name="run_example")
    model = VLMbenchEvaluator(NUM_IMAGES=NUM_IMAGES)
    # model = SceneNarrativeEvaluator(NUM_IMAGES=NUM_IMAGES)
    model.to(device)
    # model.apply(init_weights_he_normal)
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"Total Trainable Params: {params}")
    wandb.watch(model, log_freq=100)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=config["focal_loss_gamma"], alpha=config["focal_loss_alpha"])

    wandb.config.update({
        "optimizer": "Adam",
        "lr": lr,
        "batch_size": batch_size,
        "epoch": max_epoch,
        "patience": patience
    })

    counter = 0  # 改善しないエポック数をカウント
    best_valid_loss = float('inf')  # 最小の検証損失を保存
    best_acc = 0
    checkpoint_dir = create_checkpoint_dir()
    best_checkpoint_path = ""
    freeze_after_epoch = 8
    for epoch in range(max_epoch):
        # freeze_after_epochを超えたら、特定のパラメータを凍結
        if epoch == freeze_after_epoch:
            for name, param in model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False
            # オプティマイザーを更新（凍結したパラメータを除外）
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
        train_model(model, train_loader, optimizer, criterion, device, epoch)
        # if epoch < 5:
        #     scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_model.pth")
        save_checkpoint(model, checkpoint_path)
        valid_acc, avg_valid_loss = validate_model(model, valid_loader, criterion, device, epoch)
        print(f"Valid Accuracy: {valid_acc}%")

        # EarlyStopping
        # if valid_acc > best_acc:
        #     best_acc = valid_acc
        #     best_checkpoint_path = checkpoint_path
        #     counter = 0
        # else:
        #     counter += 1

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_checkpoint_path = checkpoint_path
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("EarlyStopping!")
            break

    # テスト
    print(best_checkpoint_path)
    load_checkpoint(model, best_checkpoint_path)
    test_acc = test_model(model, test_loader, device, checkpoint_path)
    print(f"Test Accuracy: {test_acc}")
    wandb.finish()

if __name__ == "__main__":
    main()
