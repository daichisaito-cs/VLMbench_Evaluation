import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from torch.utils.data import TensorDataset, DataLoader, random_split
import wandb
from torch.utils.data import DataLoader, random_split, ConcatDataset
from src.utils.data_loader import CustomDataset, create_data_loaders
from src.utils.utils import torch_fix_seed
from tqdm import tqdm

from PIL import Image
import base64
import io
import base64
import requests
import time

# OpenAI API Key
api_key = "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"

# Function to encode the image using Image.open
def encode_image_with_pil(image_path):
    # Open the image file
    with Image.open(image_path) as image:
        # Convert the image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        # Encode the image to base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def main():
    with open("configs/config.json") as config_file:
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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_set = CustomDataset(train, NUM_IMAGES=NUM_IMAGES)
    # valid_set = CustomDataset(valid, NUM_IMAGES=NUM_IMAGES)
    test_set = CustomDataset(test, NUM_IMAGES=NUM_IMAGES)

    # combined_set = ConcatDataset([train_set, valid_set, test_set])
    # total_size = len(combined_set)
    # train_size = int(0.80 * total_size)
    # valid_size = int(0.1 * total_size)
    # test_size = total_size - train_size - valid_size
    # train_set, valid_set, test_set = random_split(combined_set, [train_size, valid_size, test_size])

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    for batch_idx, (images, texts, ada, image_paths, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        for i, text in enumerate(texts[0]):
            # print(f"text: {texts[0][i]}")
            # print(f"image_path1: {image_paths[0][i]}, image_path2: {image_paths[1][i]}")
            # prompt = f"These are images taken from a single viewpoint, showing a robot performing a task with the instruction '{texts[0][i]}' Do you think it has succeeded? Answer with yes or no."
            # prompt = f"You will now be assessing the success of a task. These images are a sequence captured from a single viewpoint camera, showing a robot performing a task following the instruction '{texts[0][i]}'. Based on these images and the instruction, please determine whether the robot has successfully completed the task and answer with 'true' or 'false'. Espacially, you should focus on the color, shape, size, and position of the objects in the images."
            prompt = f"You will now be assessing the success of a task. These images are a sequence captured from a single viewpoint camera, showing a robot performing a task following the instruction '{texts[0][i]}'. Based on these images and the instruction, please determine whether the robot has successfully completed the task and answer with 'true' or 'false'. In addition, give a reason for your answer."
            # prompt = "Describe these two images."
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": prompt
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_with_pil(image_paths[0][i])}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image_with_pil(image_paths[1][i])}"
                        }
                        }
                    ]
                    }
                ],
            "max_tokens": 300
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            print(f"GT: {target[i].item()}, Pred: {response.json()['choices'][0]['message']['content']}")
            time.sleep(0.1)
        # print(f"images: {images}")
        # print(f"text: {text}")
        # print(f"ada: {ada}")
        # print(f"image_paths: {image_paths}")
        # print(f"target: {target}")
        # for i, image_path in enumerate(image_paths):
        #     print(f"image_path: {image_path}")
        break

if __name__ == "__main__":
    main()
