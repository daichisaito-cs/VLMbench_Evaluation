import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor
import clip

# データセットクラスの定義
class CustomDataset(Dataset):
    def __init__(self, data_dir, NUM_IMAGES=2):
        self.data_dir = data_dir
        self.num_images = NUM_IMAGES
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNetの平均と標準偏差で正規化
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, self.preprocessor = clip.load("RN101", device=self.device)
        # self.load_data_for_clip()
        self.load_data_scene_narrative_with_clip()

    def load_data(self):
        for task in tqdm(os.listdir(self.data_dir), total=len(os.listdir(self.data_dir))):
            with open(f"{self.data_dir}/{task}/new_evaluations.json") as f:
                json_file = json.load(f)
            episodes = [e for e in os.listdir(f"{self.data_dir}/{task}") if os.path.isdir(f"{self.data_dir}/{task}/{e}")]
            episodes = sorted(episodes, key=lambda x: int(x[7:]))
            for episode in episodes:
                if not episode in json_file:
                    continue
                angles = ["overhead", "left", "right", "wrist"]
                for angle in angles:
                    episode_images = []
                    image_paths = []
                    for img in sorted(os.listdir(f"{self.data_dir}/{task}/{episode}")):
                        if not angle in img:
                            continue
                        img_path = f"{self.data_dir}/{task}/{episode}/{img}"
                        image_paths.append(img_path)
                        image = Image.open(img_path)
                        episode_images.append(image)
                    if len(episode_images) != self.num_images:
                        # print(f"Error: {task}/{episode}/{angle} has {len(episode_images)} images")
                        continue
                    stacked_episode_image = [self.transform(episode_image) for episode_image in episode_images]
                    stacked_episode_image = torch.stack(stacked_episode_image)
                    text = json_file[episode]['description']
                    ada = np.load(json_file[episode]['embedding_path'])
                    label = json_file[episode]["succeeded"]
                    self.data.append({
                        "image": stacked_episode_image,
                        "image_paths": image_paths,
                        "text": text,
                        "ada": ada,
                        "label": label
                    })

    def load_data_for_clip(self):
        for task in tqdm(os.listdir(self.data_dir), total=len(os.listdir(self.data_dir))):
            with open(f"{self.data_dir}/{task}/new_evaluations.json") as f:
                json_file = json.load(f)
            episodes = [e for e in os.listdir(f"{self.data_dir}/{task}") if os.path.isdir(f"{self.data_dir}/{task}/{e}")]
            episodes = sorted(episodes, key=lambda x: int(x[7:]))
            for episode in episodes:
                if not episode in json_file:
                    continue
                angles = ["overhead", "left", "right", "wrist"]
                for angle in angles:
                    episode_images = []
                    image_paths = []
                    for img in sorted(os.listdir(f"{self.data_dir}/{task}/{episode}")):
                        if not angle in img:
                            continue
                        img_path = f"{self.data_dir}/{task}/{episode}/{img}"
                        image_paths.append(img_path)
                        image = Image.open(img_path)
                        episode_images.append(image)
                    if len(episode_images) != self.num_images:
                        # print(f"Error: {task}/{episode}/{angle} has {len(episode_images)} images")
                        continue
                    stacked_episode_image = [self.preprocessor(episode_image) for episode_image in episode_images]
                    stacked_episode_image = torch.stack(stacked_episode_image)
                    text = json_file[episode]['description']
                    ada = np.load(json_file[episode]['embedding_path'])
                    label = json_file[episode]["succeeded"]
                    self.data.append({
                        "image": stacked_episode_image,
                        "image_paths": image_paths,
                        "text": text,
                        "ada": ada,
                        "label": label
                    })

    def load_data_scene_narrative_with_clip(self):
        # Include scene narrative embedding
        for task in tqdm(os.listdir(self.data_dir), total=len(os.listdir(self.data_dir))):
            if "pick_cube_shape" in task:
                continue
            with open(f"{self.data_dir}/{task}/new_evaluations.json") as f:
                json_file = json.load(f)
            with open(f"data/instruct_blip/{self.data_dir.split('/')[1]}/{task}_instruct_blip.json") as f:
                # print(f"data/instruct_blip/{self.data_dir.split('/')[1]}/{task}_instruct_blip.json")
                scene_narrative_json = json.load(f)
            episodes = [e for e in os.listdir(f"{self.data_dir}/{task}") if os.path.isdir(f"{self.data_dir}/{task}/{e}")]
            episodes = sorted(episodes, key=lambda x: int(x[7:]))
            for episode in episodes:
                if not episode in json_file:
                    continue
                angles = ["overhead", "left", "right", "wrist"]
                for angle in angles:
                    episode_images = []
                    image_paths = []
                    scene_narratives = []
                    for img in sorted(os.listdir(f"{self.data_dir}/{task}/{episode}")):
                        if not angle in img:
                            continue
                        img_path = f"{self.data_dir}/{task}/{episode}/{img}"
                        image_paths.append(img_path)
                        image = Image.open(img_path)
                        episode_images.append(image)
                        scene_narratives.append(scene_narrative_json[img_path[5:]])
                    if len(episode_images) != self.num_images:
                        # print(f"Error: {task}/{episode}/{angle} has {len(episode_images)} images")
                        continue
                    stacked_episode_image = [self.preprocessor(episode_image) for episode_image in episode_images]
                    stacked_episode_image = torch.stack(stacked_episode_image)
                    text = [json_file[episode]['description'], scene_narratives]
                    ada = np.load(json_file[episode]['embedding_path'])
                    label = json_file[episode]["succeeded"]
                    self.data.append({
                        "image": stacked_episode_image,
                        "image_paths": image_paths,
                        "text": text,
                        "ada": ada,
                        "label": label
                    })

    def __getitem__(self, index):
        sample = self.data[index]
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        return sample["image"], sample["text"], sample["ada"], sample["image_paths"], label_tensor

    def __len__(self):
        return len(self.data)

def create_data_loaders(train_set, valid_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader
