import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import clip
from src.utils.utils import freeze_model, get_seed_worker
import h5py

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
        self.clip, self.clip_preprocessor = clip.load("RN101", device=self.device)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
        freeze_model(self.clip)
        freeze_model(self.bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.clip_layer_hook = self.clip.visual.layer3.register_forward_hook(
            self.get_intermediate_output
        )
        self.load_data()

    def load_data(self):
        # Include scene narrative embedding
        for task in tqdm(os.listdir(self.data_dir), total=len(os.listdir(self.data_dir))):
            # if "pick_cube_shape" in task:
            #     continue
            with open(f"{self.data_dir}/{task}/new_evaluations.json") as f:
                json_file = json.load(f)
            with open(f"{self.data_dir.split('/')[0]}/instruct_blip/{self.data_dir.split('/')[1]}/{task}_instruct_blip.json") as f:
                scene_narrative_json = json.load(f)
            episodes = [e for e in os.listdir(f"{self.data_dir}/{task}") if os.path.isdir(f"{self.data_dir}/{task}/{e}")]
            episodes = sorted(episodes, key=lambda x: int(x[7:]))
            for episode in episodes:
                if not episode in json_file:
                    continue
                angles = ["overhead", "left", "right", "wrist"]
                for angle in angles:
                    image_paths = [f"{self.data_dir}/{task}/{episode}/{img}" for img in sorted(os.listdir(f"{self.data_dir}/{task}/{episode}")) if angle in img]
                    if len(image_paths) != self.num_images:
                        # print(f"Error: {task}/{episode}/{angle} has {len(episode_images)} images")
                        continue

                    if os.path.exists(f"data/embeddings/images/clip/{task}/{episode}_{angle}.h5"):
                        clip_image = self.load_embedding_from_hdf5(f"data/embeddings/images/clip/{task}/{episode}_{angle}.h5")
                    else:
                        episode_images = [Image.open(img_path) for img_path in image_paths]
                        stacked_episode_image = torch.stack([self.clip_preprocessor(image) for image in episode_images]).to(self.device)
                        clip_image = self.clip.encode_image(stacked_episode_image).float()
                        self.save_embedding_to_hdf5(f"data/embeddings/images/clip/{task}/{episode}_{angle}.h5", clip_image)
                    
                    if os.path.exists(f"data/embeddings/images/clip2d/{task}/{episode}_{angle}.h5"):
                        clip2d = self.load_embedding_from_hdf5(f"data/embeddings/images/clip2d/{task}/{episode}_{angle}.h5")
                    else:
                        episode_images = [Image.open(img_path) for img_path in image_paths]
                        stacked_episode_image = torch.stack([self.clip_preprocessor(image) for image in episode_images]).to(self.device)
                        clip_image = self.clip.encode_image(stacked_episode_image).float()
                        clip2d = self.intermediate_output.float()
                        self.save_embedding_to_hdf5(f"data/embeddings/images/clip2d/{task}/{episode}_{angle}.h5", clip2d)

                    if os.path.exists(f"data/embeddings/scene_narrative/bert/{task}/{episode}_{angle}.h5"):
                        bert_scene_narratives = self.load_embedding_from_hdf5(f"data/embeddings/scene_narrative/bert/{task}/{episode}_{angle}.h5")
                    else:
                        scene_narratives = [scene_narrative_json[img_path[5:]] for img_path in image_paths]
                        bert_scene_narratives = []
                        for i, narrative in enumerate(scene_narratives):
                            scene_narrative_bert = self.get_bert_emb(narrative, max_length=128)
                            bert_scene_narratives.append(scene_narrative_bert)
                        bert_scene_narratives = torch.cat(bert_scene_narratives, dim=0)
                        self.save_embedding_to_hdf5(f"data/embeddings/scene_narrative/bert/{task}/{episode}_{angle}.h5", bert_scene_narratives)
                    
                    if os.path.exists(f"data/embeddings/scene_narrative/ada/{task}/{episode}_{angle}.h5"):
                        ada_scene_narratives = self.load_embedding_from_hdf5(f"data/embeddings/scene_narrative/ada/{task}/{episode}_{angle}.h5")
                    else:
                        # ada_scene_narrative0 =  np.load(f"temp/ada_scene_narrative/{task}/{episode}_0_{angle}.npy")
                        # ada_scene_narrative1 =  np.load(f"temp/ada_scene_narrative/{task}/{episode}_1_{angle}.npy")
                        # ada_scene_narratives = np.stack([ada_scene_narrative0, ada_scene_narrative1], axis=0)
                        # ada_scene_narratives = torch.tensor(ada_scene_narratives, dtype=torch.float32).to(self.device)
                        # self.save_embedding_to_hdf5(f"data/embeddings/scene_narrative/ada/{task}/{episode}_{angle}.h5", ada_scene_narratives)
                        print(f"Error: ada embedding does not exist")
                        continue

                    images = {
                        "clip2d_images": clip2d,
                        "clip_images": clip_image,
                        "bert_scene_narratives": bert_scene_narratives,
                        "ada_scene_narratives": ada_scene_narratives
                    }
                    # texts
                    inst = json_file[episode]['description']
                    if os.path.exists(f"data/embeddings/instruction/bert/{task}/{episode}.h5"):
                        bert_inst = self.load_embedding_from_hdf5(f"data/embeddings/instruction/bert/{task}/{episode}.h5")
                    else:
                        bert_inst = self.get_bert_emb(inst, 16).squeeze(0)
                        self.save_embedding_to_hdf5(f"data/embeddings/instruction/bert/{task}/{episode}.h5", bert_inst)
                    if os.path.exists(f"data/embeddings/instruction/clip/{task}/{episode}.h5"):
                        clip_inst = self.load_embedding_from_hdf5(f"data/embeddings/instruction/clip/{task}/{episode}.h5")
                    else:
                        clip_inst = self.clip.encode_text(clip.tokenize(inst).to(self.device)).squeeze(0).float()
                        self.save_embedding_to_hdf5(f"data/embeddings/instruction/clip/{task}/{episode}.h5", clip_inst)
                    if os.path.exists(f"data/embeddings/instruction/ada/{task}/{episode}.h5"):
                        ada_inst = self.load_embedding_from_hdf5(f"data/embeddings/instruction/ada/{task}/{episode}.h5")
                    else:
                        # ada_inst = np.load(f"data/ada_embeddings/{task}/{episode}.npy")
                        # ada_inst = torch.tensor(ada_inst, dtype=torch.float32)
                        # self.save_embedding_to_hdf5(f"data/embeddings/instruction/ada/{task}/{episode}.h5", ada_inst)
                        print(f"Error: ada embedding does not exist")
                        continue
                    texts = {
                        "bert": bert_inst,
                        "clip": clip_inst,
                        "ada": ada_inst
                    }
                    label = json_file[episode]["succeeded"]
                    self.data.append({
                        "images": images,
                        "image_paths": image_paths,
                        "texts": texts,
                        "label": label
                    })
        
        # del self.clip_layer_hook, self.clip, self.clip_preprocessor, self.bert_model, self.bert_tokenizer
    
    def get_intermediate_output(self, module, input, output):
        # 中間層の出力を取得
        self.intermediate_output = (
            output  # torch.Size([batch_size*num_images, 1024, 14, 14])
        )
    
    def get_bert_emb(self, text, max_length):
        inputs = self.bert_tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(self.device)
        return self.bert_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).pooler_output

    def save_embedding_to_hdf5(self, file_path, embedding):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with h5py.File(file_path, 'w') as h5file:
            h5file.create_dataset('embedding', data=embedding.cpu().numpy())

    def load_embedding_from_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as h5file:
            embedding = torch.tensor(h5file['embedding'][()])
        return embedding

    def __getitem__(self, index):
        sample = self.data[index]
        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        return sample["images"], sample["texts"], sample["image_paths"], label_tensor

    def __len__(self):
        return len(self.data)

def create_data_loaders(train_set, valid_set, test_set, batch_size, seed=42):
    seed_worker = get_seed_worker()
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
    return train_loader, valid_loader, test_loader

class GPTDataset(Dataset):
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
            # if "pick_cube_shape" in task:
            #     continue
            with open(f"{self.data_dir}/{task}/new_evaluations.json") as f:
                json_file = json.load(f)
            with open(f"{self.data_dir.split('/')[0]}/instruct_blip/{self.data_dir.split('/')[1]}/{task}_instruct_blip.json") as f:
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
                    # ada = np.load(json_file[episode]['embedding_path'])
                    ada = np.load(f"data/ada_embeddings/{task}/{episode}.npy")
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