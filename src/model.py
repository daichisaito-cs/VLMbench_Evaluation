from email.mime import image
from matplotlib.transforms import Transform
from sympy import im
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
import torchvision.models as models
from torchvision import transforms
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch.nn.functional as Fn
from einops import rearrange
import timm
import clip
import torch.nn.functional as F
Transformer = nn.Transformer
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
        # input_dim = 512 + 512 * NUM_IMAGES
        self.fc1 = nn.Linear(input_dim, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # make bert_model not trainable
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.MAX_LENGTH = MAX_LENGTH
        # self.batch_norm_before_resnet = nn.BatchNorm2d(3)
        # self.resnet = models.resnet101(pretrained=True).cuda()
        self.resnet = models.resnet18(pretrained=True).cuda()
        self.resnet.fc = nn.Identity()

        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

    def forward(self, texts, images, ada):
        text = texts[0]
        # Get scene narrative embedding using BERT
        inputs = self.bert_tokenizer(texts[1][0], padding=True, truncation=True, return_tensors="pt", max_length=128)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        scene_narrative_bert1 = outputs.pooler_output.cuda() # torch.Size([16, 768])

        inputs = self.bert_tokenizer(texts[1][1], padding=True, truncation=True, return_tensors="pt", max_length=128)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        scene_narrative_bert2 = outputs.pooler_output.cuda() # torch.Size([16, 768])

        # BERT
        inputs = self.bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=self.MAX_LENGTH)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        bert_emb = outputs.pooler_output.cuda() # torch.Size([16, 768])

        # Reshape images for ResNet or CLIP
        reshaped_images = images.view(-1, 3, 224, 224)  # 新しい形状: [16, 3, 224, 224]

        # ResNet
        res_images = self.resnet(reshaped_images)  # 出力形状: [16, 2048, 1, 1]
        res_images = res_images.view(len(text), -1)  # 新しい形状: [4, 8192] (2048*4 = 8192)

        x = torch.cat([scene_narrative_bert1, scene_narrative_bert2, bert_emb, res_images], dim=1)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        # _, predicted = torch.max(x, 1)
        # print(predicted)

        return x

class FlamingoBasedEvaluator(nn.Module):
    def __init__(self, NUM_IMAGES=2, MAX_LENGTH=64):
        super(FlamingoBasedEvaluator, self).__init__()
        # input_dim = 768 + 2048 * NUM_IMAGES
        input_dim = 50280 + NUM_IMAGES * 64 * 1024
        self.fc1 = nn.Linear(input_dim, 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MAX_LENGTH = MAX_LENGTH

        self.flamingo, image_processor, self.flamingo_tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            cross_attn_every_n_layers=1
        )
        self.flamingo_tokenizer.padding_side = "left" # For generation padding tokens should be on the left

        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
        self.flamingo.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.flamingo = self.flamingo.to("cuda")

        # flamingoの学習可能なパラメータを出力
        # for name, param in self.flamingo.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        for param in self.flamingo.parameters():
            param.requires_grad = False
        for layer in self.flamingo.lang_encoder.transformer.blocks[-1:]:
            for param in layer.parameters():
                param.requires_grad = True
        # for block in self.flamingo.vision_encoder.transformer.resblocks[-2:]:
        #     for param in block.parameters():
        #         param.requires_grad = True
        for layer in self.flamingo.perceiver.layers:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, text, images, ada):
        # images: [batch_size, 2, 3, 224, 224]
        # [batch_size, 2, 3, 224, 224]から[batch_size, 1, 2, 3, 224, 224]に変換
        images = images.unsqueeze(2)
        # print("images: ", images.shape)

        # flamingo
        max_length = max([len(self.flamingo_tokenizer.encode(f"<image>, <image>, <image>, <image> are images from four different angles showing the robot performing a task with the instruction: {decoded_text} Has it succeeded? Answer:")) for decoded_text in text])
        questions = []
        for decoded_text in text:
            # question = f"<image><image><image><image>Question: Did it {decoded_text[:-1]}? Answer:"
            question = f"<image>, <image>, <image>, <image> are images from four different angles showing the robot performing a task with the instruction: {decoded_text} Has it succeeded? Answer:"
            # question = decoded_text
            padding_length = max_length - len(self.flamingo_tokenizer.encode(question))
            padded_question = "<PAD>" * padding_length + question
            questions.append(padded_question)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lang_x = self.flamingo_tokenizer(
            questions,
            return_tensors="pt",
        ).to(device)

        outputs = self.flamingo(
            vision_x=images,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"]
        )
        # print("outputs: ", outputs)
        logits = outputs.logits

        logits = logits.mean(dim=1)      # [batch_size, 35, 50280] -> [batch_size, 50280]
        # logits = Fn.relu(self.fc_logit(logits))
        logits = Fn.relu(logits)

        b, T, F = images.shape[:3]
        # # # # print("b, T, F: ", b, T, F)

        vision_x = rearrange(images, "b T F c h w -> (b T F) c h w")
        # print("vision_x2: ", vision_x.shape)
        vision_x = self.flamingo.vision_encoder(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.flamingo.perceiver(vision_x)     # [8, 4, 64, 1024]
        # print("vision_x2: ", vision_x.shape)
        vision_x = Fn.relu(vision_x.view(b, -1))   # [8, 4 * 64 * 1024]

        x = torch.cat([vision_x, logits], dim=1)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        # _, predicted = torch.max(x, 1)
        # print(predicted)

        return x

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
