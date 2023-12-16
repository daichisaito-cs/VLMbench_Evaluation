import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from torch.utils.data import TensorDataset, DataLoader, random_split
import wandb
from src.model import Flamingo
from torch.utils.data import DataLoader, random_split, ConcatDataset
from utils.data_loader import CustomDataset, create_data_loaders
from utils import torch_fix_seed
from tqdm import tqdm

def main():
    with open("config.json") as config_file:
        config = json.load(config_file)
    torch_fix_seed(config["seed"])
    lr = config["lr"]
    max_epoch = config["max_epoch"]
    patience = config["patience"]
    train = config["train_data_path"]
    valid = config["valid_data_path"]
    test = config["test_data_path"]
    batch_size = config["batch_size"]
    data_dirs = [train, valid, test]
    MAX_LENGTH = 64
    NUM_IMAGES = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = CustomDataset(train, NUM_IMAGES=NUM_IMAGES)
    valid_set = CustomDataset(valid, NUM_IMAGES=NUM_IMAGES)
    test_set = CustomDataset(test, NUM_IMAGES=NUM_IMAGES)

    # combined_set = ConcatDataset([train_set, valid_set, test_set])
    # total_size = len(combined_set)
    # train_size = int(0.80 * total_size)
    # valid_size = int(0.1 * total_size)
    # test_size = total_size - train_size - valid_size
    # train_set, valid_set, test_set = random_split(combined_set, [train_size, valid_size, test_size])

    train_loader, valid_loader, test_loader = create_data_loaders(train_set, valid_set, test_set, batch_size=batch_size)

    for batch_idx, (images, text, ada, image_paths, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        print(f"images: {images}")
        print(f"text: {text}")
        print(f"ada: {ada}")
        print(f"image_paths: {image_paths}")
        print(f"target: {target}")
        for i, image_path in enumerate(image_paths):
            print(f"image_path: {image_path}")
        break

if __name__ == "__main__":
    main()
