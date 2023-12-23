from email.mime import image
from math import comb
from matplotlib.transforms import Transform
from sympy import im
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
import torchvision.models as models
from torchvision import transforms
# from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch.nn.functional as Fn
from einops import rearrange
import timm
import clip
import torch.nn.functional as F
import numpy as np
from transformer_encoder import TransformerEncoder
# import loralib as lora

class SceneNarrativeEvaluator(nn.Module):
    def __init__(self, NUM_IMAGES=2, MAX_LENGTH=64):
        super(SceneNarrativeEvaluator, self).__init__()
        input_dim = 768 * 3 + 512 * NUM_IMAGES
        self.num_images = NUM_IMAGES
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transformer = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )

        self.bert_scene_narrative = nn.Linear(768, 512)
        self.ada_scene_narrative = nn.Linear(1536, 512)
        self.bert_inst = nn.Linear(768, 512)
        self.ada_linear = nn.Linear(1536, 512)
        self.text_linear = nn.Linear(768+512, 512)
        self.fc1 = nn.Linear(512+768, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.conv = nn.Conv2d(1024, 512, kernel_size=1)

        self.attention_aggregator = AttentionAggregator(398, 512)
        self.pos_enc = AddPositionalEncoding(512, 1000)

    def forward(self, images, texts):
        inst_bert = texts["bert"].to(self.device) # torch.Size([32, 768])
        clip_inst = texts["clip"].to(self.device) # torch.Size([32, 512])
        ada_inst = texts["ada"].to(self.device) # torch.Size([32, 1536])
        bert_scene_narrative = images["bert_scene_narratives"].to(self.device) # torch.Size([32, 2, 768])
        ada_scene_narrative = images["ada_scene_narratives"].to(self.device) # torch.Size([32, 2, 1536])
        clip2d_images = images["clip2d_images"].to(self.device) # torch.Size([32, 2, 1024, 14, 14])
        clip_images = images["clip_images"].to(self.device) # torch.Size([32, 2, 512])
        task_emb = texts["task"].to(self.device) # torch.Size([32, 768])

        B, N, _ = clip_images.shape

        # Scene narrative embedding
        bert_scene_narrative = self.bert_scene_narrative(bert_scene_narrative) # torch.Size([32, 2, 512])
        ada_scene_narrative = self.ada_scene_narrative(ada_scene_narrative) # torch.Size([32, 2, 512])

        # BERT instruction embedding
        inst_bert = inst_bert.unsqueeze(1)  # [batch, 1, 768]
        inst_bert = self.bert_inst(inst_bert)  # [batch, 1, 512]

        #CLIP
        clip_inst = clip_inst.unsqueeze(1) # torch.Size([32, 1, 512])

        # clip2d_image = self.intermediate_output.float()  # [batch_size*num_images, 1024, 14, 14]
        clip2d_image = clip2d_images.view(-1, 1024, 14, 14)  # [batch_size*num_images, 1024, 14, 14]
        clip2d_image = self.conv(clip2d_image) # [batch_size*num_images, 512, 14, 14]
        clip2d_image = clip2d_image.flatten(2) # [batch_size*num_images, 512, 196]
        clip2d_image = clip2d_image.permute(0, 2, 1) # [batch_size*num_images, 196, 512]
        clip2d_image = clip2d_image.reshape(B, self.num_images*196, 512) # [batch_size, num_images*196, 512]

        # Process ada instruction
        ada_inst = ada_inst.unsqueeze(1) # torch.Size([32, 1, 1536])
        ada_inst = self.ada_linear(ada_inst.float())  # [batch, 1, 512]

        # text features
        text_features = torch.cat([clip_inst, ada_inst, inst_bert], dim=1) # [batch_size, 4, 512]
        
        # scene_narrativesとclip2d_imageを結合
        image_features = torch.cat([clip_images, clip2d_image, ada_scene_narrative, bert_scene_narrative], dim=1) # [batch_size, num_images*196+6, 512]

        combined_features = self.transformer(image_features, text_features) # [batch_size, num_images*196+6, 512]
        
        attn_weights = self.attention_aggregator(combined_features)
        x = (combined_features * attn_weights) # [batch_size, 512]
        # max pooling
        x = x.max(dim=1)[0] # [batch_size, 512]

        x = torch.cat([x, task_emb], dim=1) # [batch_size, 512+768]

        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class AttentionAggregator(nn.Module):
    def __init__(self, seq_len, d_model):
        super(AttentionAggregator, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x):
        # xの形状: (B, seq_len, d_model), ここでB=64, seq_len=396, d_model=512

        # 各要素に対するスコアを計算
        scores = self.attention_weights(x).squeeze(-1)  # 形状: (B, seq_len)
        attention_weights = F.softmax(scores, dim=1)  # 形状: (B, seq_len)

        # 加重平均を計算
        weighted_average = torch.bmm(attention_weights.unsqueeze(1), x)
        # 形状: (B, 1, d_model)

        return weighted_average

class AddPositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, max_len: int, device: torch.device = torch.device("cpu")
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        positional_encoding_weight: torch.Tensor = self._initialize_weight().to(device,non_blocking=True)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(0)

    def _get_positional_encoding(self, pos: int, i: int) -> float:
        w = pos / (10000 ** (((2 * i) // 2) / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()

class ResnetEvaluator(nn.Module):
    def __init__(self, NUM_IMAGES=2, MAX_LENGTH=64):
        super(ResnetEvaluator, self).__init__()
        # input_dim = 768 + 2048 * NUM_IMAGES
        input_dim = 512 * NUM_IMAGES
        self.fc = nn.Linear(input_dim, 2)
        # self.resnet = models.resnet101(pretrained=True).cuda()
        self.resnet = models.resnet18(pretrained=True).cuda()
        self.resnet.fc = nn.Identity()

    def forward(self, text, images, ada):
        # Reshape images for ResNet or CLIP
        reshaped_images = images.view(-1, 3, 224, 224)  # 新しい形状: [16, 3, 224, 224]
        # ResNet
        res_images = self.resnet(reshaped_images)  # 出力形状: [16, 2048, 1, 1]
        res_images = res_images.view(len(text), -1)  # 新しい形状: [4, 8192] (2048*4 = 8192)

        x = self.fc(res_images)

        return x

class ViTEvaluator(nn.Module):
    def __init__(self, NUM_IMAGES=2, MAX_LENGTH=64):
        super(ViTEvaluator, self).__init__()
        # self.resnet = models.resnet101(pretrained=True).cuda()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
        self.fc = torch.nn.Linear(1000*2, 2)  # 出力層の変更: ViTからの出力を2倍して2クラス分類

    def forward(self, text, images, ada):
        # Reshape images for ResNet or CLIP
        reshaped_images = images.view(-1, 3, 224, 224)  # 新しい形状: [16, 3, 224, 224]
        # ResNet
        vit_images = self.vit(reshaped_images)  # 出力形状: [16, 2048, 1, 1]
        vit_images = vit_images.view(len(text), -1)  # 新しい形状: [4, 8192] (2048*4 = 8192)

        x = self.fc(vit_images)

        return x
