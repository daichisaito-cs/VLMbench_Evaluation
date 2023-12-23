import torch
from tqdm import tqdm
from utils.utils import load_checkpoint
from utils.utils import plot_confusion_matrices
from src.model import SceneNarrativeEvaluator
from utils.data_loader import CustomDataset, create_data_loaders
from utils.utils import torch_fix_seed, save_checkpoint, load_checkpoint, create_checkpoint_dir, find_trainable_layers, init_weights_he_normal, init_weights_he_normal, text_to_ids, FocalLoss
import wandb
import json
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

def test_model(model, test_loader, device, checkpoint_path):
    task_correct = {}
    task_total = {}
    task_TP = {}
    task_FP = {}
    task_FN = {}

    with torch.no_grad():
        for images, texts, image_paths, target in tqdm(test_loader, total=len(test_loader)):
            # images, ada, target = images.to(device), ada.to(device), target.to(device)
            output = model(images, texts)
            _, predicted = torch.max(output, 1)
            if predicted.shape == torch.Size([]) or target.shape == torch.Size([]):
                continue
            for i, image_path in enumerate(image_paths[0]):
                # 画像ファイル名からタスク名を取得
                task_name = image_path.split('/')[-3]
                if task_name not in task_correct:
                    task_correct[task_name] = 0
                    task_total[task_name] = 0
                    task_TP[task_name] = 0
                    task_FP[task_name] = 0
                    task_FN[task_name] = 0

                # 正解数と総数を更新
                task_total[task_name] += 1
                if predicted[i] == target[i]:
                    task_correct[task_name] += 1
                task_TP[task_name] += ((predicted[i] == 1) & (target[i] == 1)).item()
                task_FP[task_name] += ((predicted[i] == 1) & (target[i] == 0)).item()
                task_FN[task_name] += ((predicted[i] == 0) & (target[i] == 1)).item()

    # 各タスクの正解率を表示
    total = 0
    correct = 0
    for task_name in task_correct:
        accuracy = task_correct[task_name] / task_total[task_name]
        total += task_total[task_name]
        correct += task_correct[task_name]
        print(f"Task: {task_name}, Correct: {task_correct[task_name]}, Total: {task_total[task_name]}, Accuracy: {accuracy:.2f}")

        if task_TP[task_name] + task_FP[task_name] == 0 or task_TP[task_name] + task_FN[task_name] == 0:
            precision = 0
            recall = 0
        else:
            precision = task_TP[task_name] / (task_TP[task_name] + task_FP[task_name])
            recall = task_TP[task_name] / (task_TP[task_name] + task_FN[task_name])
        if precision == 0 and recall == 0:
            F1 = 0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

    # 各タスクのメトリクスを計算し、視覚化
    task_names = list(task_correct.keys())
    task_metrics = {task_name: (task_TP[task_name], task_FP[task_name], task_FN[task_name], task_total[task_name]) for task_name in task_names}
    plot_confusion_matrices(task_names, task_metrics, 'task_confusion_matrices.png')

    # 総合的な正解率を計算
    total = sum(task_total.values())
    correct = sum(task_correct.values())
    test_acc = correct / total

    return test_acc

def main():
    with open("configs/config.json") as config_file:
        config = json.load(config_file)
    torch_fix_seed(config["seed"])
    train = config["train_data_path"]
    valid = config["valid_data_path"]
    test = config["test_data_path"]
    batch_size = config["batch_size"]
    NUM_IMAGES = config["input_image_num"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SceneNarrativeEvaluator(NUM_IMAGES=NUM_IMAGES)
    model.to(device)
    test_set = CustomDataset(test, NUM_IMAGES=NUM_IMAGES)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    checkpoint_path = "checkpoints/20231223-124427/epoch_81_model.pth"

    # テスト
    print(checkpoint_path)
    load_checkpoint(model, checkpoint_path)
    test_acc = test_model(model, test_loader, device, checkpoint_path)
    print(f"Test Accuracy: {test_acc}")

    # for i in range(50, 85):
    #     checkpoint_path = f"checkpoints/20231223-124427/epoch_{i}_model.pth"
    #     print(checkpoint_path)
    #     load_checkpoint(model, checkpoint_path)
    #     test_acc = test_model(model, test_loader, device, checkpoint_path)
    #     print(f"Epoch{i}, Test Accuracy: {test_acc}")

if __name__ == "__main__":
    main()
