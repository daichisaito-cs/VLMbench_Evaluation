import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
import torchvision.models as models
from torchvision import transforms
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch.nn.functional as Fn
from einops import rearrange

class VLMbenchEvaluator(nn.Module):
    def __init__(self, NUM_IMAGES=2, MAX_LENGTH=64):
        super(VLMbenchEvaluator, self).__init__()
        # input_dim = 768 + 2048 * NUM_IMAGES
        input_dim = 512 + 512 * NUM_IMAGES
        self.fc1 = nn.Linear(input_dim, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.MAX_LENGTH = MAX_LENGTH
        # self.batch_norm_before_resnet = nn.BatchNorm2d(3)
        # self.resnet = models.resnet101(pretrained=True).cuda()
        # self.resnet = models.resnet18(pretrained=True).cuda()
        # self.resnet.fc = nn.Identity()

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

    def forward(self, text, images, ada):
        # BERT
        # inputs = self.bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=self.MAX_LENGTH)
        # inputs['input_ids'] = inputs['input_ids'].cuda()
        # inputs['attention_mask'] = inputs['attention_mask'].cuda()
        # outputs = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # bert_emb = outputs.pooler_output.cuda() # torch.Size([16, 768])

        # Reshape images for ResNet or CLIP
        reshaped_images = images.view(-1, 3, 224, 224)  # 新しい形状: [16, 3, 224, 224]

        #CLIP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {key: val.to(device) for key, val in text_inputs.items()}
        outputs = self.clip(**text_inputs, pixel_values=reshaped_images)
        clip_text = outputs['text_embeds']
        clip_image = outputs['image_embeds']
        clip_image = clip_image.view(len(text), -1) # torch.Size([batch_size, 512])

        # # ResNet
        # reshaped_images = self.batch_norm_before_resnet(reshaped_images)
        # res_images = self.resnet(reshaped_images)  # 出力形状: [16, 2048, 1, 1]
        # res_images = res_images.view(len(text), -1)  # 新しい形状: [4, 8192] (2048*4 = 8192)

        x = torch.cat([clip_text, clip_image], dim=1)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        # _, predicted = torch.max(x, 1)
        # print(predicted)

        return x

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

        #CLIP
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        # text_inputs = {key: val.to(device) for key, val in text_inputs.items()}
        # outputs = self.clip(**text_inputs, pixel_values=reshaped_images)
        # clip_text = outputs['text_embeds']
        # clip_image = outputs['image_embeds']
        # clip_image = clip_image.view(len(text), -1) # torch.Size([batch_size, 512])

        # # ResNet
        # reshaped_images = self.batch_norm_before_resnet(reshaped_images)
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
