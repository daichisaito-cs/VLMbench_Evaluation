import torch
import numpy as np
import random
import os
from datetime import datetime
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.ops import sigmoid_focal_loss
import loralib as lora

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def save_checkpoint(model, filename, adopt_lora=False):
    if adopt_lora:
        torch.save(model.state_dict(), filename)
        torch.save(lora.lora_state_dict(model), filename.replace(".pth", "_lora.pth"))
    else:
        torch.save(model.state_dict(), filename)

def load_checkpoint(model, filename, adopt_lora=False):
    if adopt_lora:
        model.load_state_dict(torch.load(filename), strict=False)
        model.load_state_dict(filename.replace(".pth", "_lora.pth"), strict=False)
    else:
        model.load_state_dict(torch.load(filename))
    model.eval()

def create_checkpoint_dir(base_dir="checkpoints"):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(base_dir, current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def find_trainable_layers(model):
    trainable_layers = {}
    non_trainable_layers = {}

    for name, child in model.named_children():
        if any(param.requires_grad for param in child.parameters()):
            trainable_layers[name] = child
        else:
            non_trainable_layers[name] = child

    # 学習可能な層の名前を出力
    print("Trainable Layers:")
    for name in trainable_layers.keys():
        print(f" - {name}")

    # 学習不可能な層の名前を出力
    print("Non-Trainable Layers:")
    for name in non_trainable_layers.keys():
        print(f" - {name}")


def init_weights_he_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def text_to_ids(text, tokenizer, max_length):
        return tokenizer.encode(text, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True)

def find_trainable_layers(model):
    trainable_layers = {}
    non_trainable_layers = {}

    for name, child in model.named_children():
        if any(param.requires_grad for param in child.parameters()):
            trainable_layers[name] = child
        else:
            non_trainable_layers[name] = child

    # 学習可能な層の名前を出力
    print("Trainable Layers:")
    for name in trainable_layers.keys():
        print(f" - {name}")

    # 学習不可能な層の名前を出力
    print("Non-Trainable Layers:")
    for name in non_trainable_layers.keys():
        print(f" - {name}")

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def plot_confusion_matrices(task_names, task_metrics, output_path):
    rows = int(np.ceil(len(task_names) / 4))  # 4は1行あたりの混同行列の数
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))  # figsizeを適切に調整

    axes = axes.ravel()  # 2Dのaxesを1Dのリストに変換

    for ax, task_name in zip(axes, task_names):
        # 混同行列を作成
        TP, FP, FN, task_total = task_metrics[task_name]
        TN = task_total - (TP + FP + FN)

        confusion_matrix = np.array([[TN, FP], [FN, TP]])

        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 24})
        ax.set_title(f'Task: {task_name}')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')

    # タスクの数よりも多くのサブプロットがある場合、余分なサブプロットを非表示にする
    for i in range(len(task_names), rows * 4):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)

class FocalLoss(nn.Module):
    """
    alpha: データの不均衡によってlossの重み付けを変える。0.5が真ん中。負例の方が多い場合は0.25くらいにして負例のlossに大きく重み付けをする。
    gamma: 易しい例の重みを下げることで、予測が難しい例に焦点を当てた学習を行う。
    targetが1の場合alpha_t = alpha、targetが0の場合alpha_t = 1 - alpha。それをlossにかける。
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # targetsをone-hot形式に変換する
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.to(dtype=inputs.dtype, device=inputs.device)

        # sigmoid_focal_lossの呼び出し
        loss = sigmoid_focal_loss(inputs, targets_one_hot, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        return loss