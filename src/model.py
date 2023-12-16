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
import loralib as lora

class VLMbenchEvaluator(nn.Module):
    def __init__(self, NUM_IMAGES=2, MAX_LENGTH=64):
        super(VLMbenchEvaluator, self).__init__()
        self.num_images = NUM_IMAGES
        # input_dim = 768 + 2048 * NUM_IMAGES
        self.input_dim = 512 + 512 * NUM_IMAGES

        self.image_feature_dim = 512
        self.text_feature_dim = 512
        self.combined_dim = 512  # 特徴量の次元を統一

        self.layers = nn.ModuleList([
            SublayerUnit(self.image_feature_dim, self.combined_dim)
            for _ in range(24)
        ])
        self.fc = nn.Linear(self.combined_dim, 2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip, self.preprocessor = clip.load("RN101", device=self.device)
        # make clip not trainable
        for param in self.clip.parameters():
            param.requires_grad = False

        self.hook = self.clip.visual.layer3.register_forward_hook(self.get_intermediate_output)
        self.conv = nn.Conv2d(1024, 512, kernel_size=1)


    def forward(self, text, images, ada):
        # images : torch.Size([32, 2, 3, 224, 224])
        # BERT
        # inputs = self.bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=self.MAX_LENGTH)
        # inputs['input_ids'] = inputs['input_ids'].cuda()
        # inputs['attention_mask'] = inputs['attention_mask'].cuda()
        # outputs = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # bert_emb = outputs.pooler_output.cuda() # torch.Size([16, 768])

        # Reshape images for ResNet or CLIP
        reshaped_images = images.view(-1, 3, 224, 224)  # 新しい形状: [batch_size*2, 3, 224, 224]

        #CLIP
        processed_text = clip.tokenize(text).to(self.device)
        # logits_per_image, logits_per_text = self.clip(reshaped_images, processed_text)
        image_features = self.clip.encode_image(reshaped_images).float()
        text_features = self.clip.encode_text(processed_text).float()

        image_features = image_features.view(-1, self.num_images, self.image_feature_dim) # torch.Size([32, 2, 512])
        text_features = text_features.unsqueeze(1).expand(-1, self.num_images, -1)

        position_features = self.conv(self.intermediate_output.float())
        position_features = torch.nn.functional.adaptive_avg_pool2d(position_features, (1, 1))
        position_features = position_features.view(-1, self.num_images, 512)  # [batch_size, num_images, 512]

        combined_features = image_features + position_features
        for layer in self.layers:
            combined_features = layer(combined_features, text_features) + position_features # torch.Size([batch_size, 2, 512])
        # TODO 二枚の画像の合わせ方を変更する
        combined_features = combined_features.mean(dim=1) # torch.Size([batch_size, 512])

        output = self.fc(combined_features)

        return output

    def get_intermediate_output(self, module, input, output):
        # 中間層の出力を取得
        self.intermediate_output = output # torch.Size([batch_size*num_images, 1024, 14, 14])

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, output_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(query_dim, output_dim)
        self.key_proj = nn.Linear(key_dim, output_dim)
        self.value_proj = nn.Linear(value_dim, output_dim)
        self.scale = output_dim ** -0.5

    def forward(self, query, key, value):
        query_proj = self.query_proj(query)
        key_proj = self.key_proj(key)
        value_proj = self.value_proj(value)

        attention_scores = torch.matmul(query_proj, key_proj.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value_proj)
        return context

class SublayerUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SublayerUnit, self).__init__()
        self.cross_attn = CrossAttention(input_dim, input_dim, input_dim, output_dim)
        self.linear1 = nn.Linear(output_dim, output_dim*4)
        self.linear2 = nn.Linear(output_dim*4, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, text_features):
        # クロスアテンション層
        attention_output = self.cross_attn(x, text_features, text_features) # torch.Size([batch_size, 2, 512])
        # 2層の線形層
        linear_output = F.relu(self.linear1(attention_output))
        linear_output = self.linear2(linear_output)
        # 残差接続とレイヤー正規化
        return self.norm(linear_output)

class SceneNarrativeEvaluator(nn.Module):
    def __init__(self, NUM_IMAGES=2, MAX_LENGTH=64):
        super(SceneNarrativeEvaluator, self).__init__()
        input_dim = 768 * 3 + 512 * NUM_IMAGES
        self.num_images = NUM_IMAGES
        # input_dim = 512 + 512 * NUM_IMAGES

        self.bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # make bert_model not trainable
        for param in self.bert_model.parameters():
            param.requires_grad = False
        # self.resnet = models.resnet101(pretrained=True).cuda()
        # self.resnet = models.resnet18(pretrained=True).cuda()
        # self.resnet.fc = nn.Identity()
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

        # self.self_attn = nn.MultiheadAttention(512, 8, batch_first=True)

        # self.transformer_encoder = self.transformer.encoder

        self.clip, self.preprocessor = clip.load("RN101", device=self.device)
        # make clip not trainable
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # self.integrate_lora_to_specific_layers(self.clip)
        # lora.mark_only_lora_as_trainable(self.clip)
        # for name, param in self.clip.named_parameters():
        #     print(name, param.requires_grad)

        self.scene_narrative1 = nn.Linear(768, 512)
        self.scene_narrative2 = nn.Linear(768, 512)
        self.text_linear = nn.Linear(768+512, 512)
        self.fc1 = nn.Linear(512, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.conv = nn.Conv2d(1024, 512, kernel_size=1)

        # self.positional_encoding = nn.Parameter(torch.randn(1, self.num_images, 1024, 14*14))

        self.hook = self.clip.visual.layer3.register_forward_hook(self.get_intermediate_output)

    def forward(self, texts, images, ada):
        instruction = texts[0]
        scene_narratives = texts[1]
        batch_size = len(instruction)

        # Scene narrative embedding
        scene_narrative_berts = []
        layers = [self.scene_narrative1, self.scene_narrative2]  # 異なる層のリスト
        for i, narrative in enumerate(scene_narratives):
            inputs = self.bert_tokenizer(narrative, padding=True, truncation=True, return_tensors="pt", max_length=256)
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            outputs = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            scene_narrative_bert = outputs.pooler_output.cuda().unsqueeze(1)  # [batch, 1, 768]
            processed_narrative = layers[i](scene_narrative_bert)  # [batch, 1, 512]
            scene_narrative_berts.append(processed_narrative)
        concatenated_scene_narrative_bert = torch.cat(scene_narrative_berts, dim=1)  # [batch, 2, 512]

        # Reshape images for ResNet or CLIP
        reshaped_images = images.view(-1, 3, 224, 224)  # 新しい形状: [batch*num_images, 3, 224, 224]

        #CLIP
        processed_text = clip.tokenize(instruction).to(self.device)
        clip_image = self.clip.encode_image(reshaped_images).float()
        clip_inst = self.clip.encode_text(processed_text).float().unsqueeze(1) # torch.Size([32, 1, 512])

        clip2d_image = self.intermediate_output.float()  # [batch_size*num_images, 1024, 14, 14]
        clip2d_image = self.conv(clip2d_image) # [batch_size*num_images, 512, 14, 14]
        clip2d_image = clip2d_image.flatten(2) # [batch_size*num_images, 512, 196]
        clip2d_image = clip2d_image.permute(0, 2, 1) # [batch_size*num_images, 196, 512]
        clip2d_image = clip2d_image.reshape(batch_size, self.num_images*196, 512) # [batch_size, num_images*196, 512]

        # scene_narrativesとclip2d_imageを結合
        combined_image_features = torch.cat([clip2d_image, concatenated_scene_narrative_bert], dim=1) # [batch_size, num_images*196+2, 512]

        combined_features = self.transformer(clip_inst, combined_image_features) # [batch_size, num_images*196+2, 512]
        # combined_featuresにself_attn(batch_first=True)を適用
        # self_attn_output, _ = self.self_attn(combined_features, combined_features, combined_features) # [batch_size, num_images*196+2, 512]
        
        x = combined_features.mean(dim=1) # [batch_size, 512]
        # x = combined_features[:, 0, :] # [batch_size, 512]

        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        # _, predicted = torch.max(x, 1)
        # print(predicted)

        return x

    def get_intermediate_output(self, module, input, output):
        # 中間層の出力を取得
        self.intermediate_output = output # torch.Size([batch_size*num_images, 1024, 14, 14])
    
    def integrate_lora_to_specific_layers(clip_resnet, rank=16):
        layer_names = [
            "transformer.resblocks.11.attn.in_proj_weight",
            "transformer.resblocks.11.attn.in_proj_bias",
            "transformer.resblocks.11.attn.out_proj.weight",
            "transformer.resblocks.11.attn.out_proj.bias",
            "transformer.resblocks.11.ln_1.weight",
            "transformer.resblocks.11.ln_1.bias",
            "transformer.resblocks.11.mlp.c_fc.weight",
            "transformer.resblocks.11.mlp.c_fc.bias",
            "transformer.resblocks.11.mlp.c_proj.weight",
            "transformer.resblocks.11.mlp.c_proj.bias",
            "transformer.resblocks.11.ln_2.weight",
            "transformer.resblocks.11.ln_2.bias"
        ]

        for name, module in clip_resnet.named_modules():
            if name in layer_names and isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                new_module = lora.Linear(in_features, out_features, r=rank)
                parent_name, child_name = name.rsplit('.', 1)
                parent = dict(clip_resnet.named_modules())[parent_name]
                setattr(parent, child_name, new_module)

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
