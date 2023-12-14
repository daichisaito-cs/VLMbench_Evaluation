import torch
from tqdm import tqdm
from utils import load_checkpoint
from src.model import VLMbenchEvaluator
from utils.data_loader import CustomDataset, create_data_loaders
from utils import torch_fix_seed, save_checkpoint, load_checkpoint, create_checkpoint_dir, find_trainable_layers, init_weights_he_normal, init_weights_he_normal, text_to_ids, FocalLoss
import json
import os
import shutil

def main():
    with open("config.json") as config_file:
        config = json.load(config_file)
    torch_fix_seed(config["seed"])
    train = config["train_data_path"]
    valid = config["valid_data_path"]
    test = config["test_data_path"]
    batch_size = config["batch_size"]
    NUM_IMAGES = config["input_image_num"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = CustomDataset(train, NUM_IMAGES=NUM_IMAGES)
    valid_set = CustomDataset(valid, NUM_IMAGES=NUM_IMAGES)
    test_set = CustomDataset(test, NUM_IMAGES=NUM_IMAGES)

    train_loader, valid_loader, test_loader = create_data_loaders(train_set, valid_set, test_set, batch_size=batch_size)

    model = VLMbenchEvaluator(NUM_IMAGES=NUM_IMAGES)
    model.to(device)

    # checkpoint_path = "/home/initial/workspace/VLMbench_Evaluation/checkpoints/20231129-164317-train/epoch_18_model.pth"
    # valid
    # checkpoint_path = "/home/initial/workspace/VLMbench_Evaluation/checkpoints/20231130-051617-valid/epoch_14_model.pth"
    # test
    # checkpoint_path = "/home/initial/workspace/VLMbench_Evaluation/checkpoints/20231130-071237-test/epoch_16_model.pth"
    checkpoint_path = "/home/initial/workspace/VLMbench_Evaluation/checkpoints/20231130-082402/epoch_4_model.pth"
    load_checkpoint(model, checkpoint_path)

    eliminate_list = []

    with torch.no_grad():
        for images, text, ada, image_paths, target in tqdm(valid_loader, total=len(valid_loader)):
            images, ada, target = images.to(device), ada.to(device), target.to(device)
            output = model(text, images, ada)
            _, predicted = torch.max(output, 1)
            # predictedとtargetが一致しているか確認し、一致していないpathを出力
            # print(image_paths)
            for i in range(len(predicted)):
                if predicted[i].item() != target[i].item():
                    eliminate = {
                        "task": image_paths[0][i].split("/")[1],
                        "episode": image_paths[0][i].split("/")[2],
                        "image_path_1": image_paths[0][i],
                        "image_path_2": image_paths[1][i],
                        "output_diff": abs((output[i][0] - output[i][1]).item())
                    }
                    eliminate_list.append(eliminate)

    # eliminate_listをoutput_diffの値でソート
    eliminate_list = sorted(eliminate_list, key=lambda x: x["output_diff"], reverse=True)
    # for i in eliminate_list[:15]:
    #     print(i)
    
    # src_dirs = ["train_c_set", "valid_c_set", "test_c_set"]
    # new_dirs = ["train_c_set_v2", "valid_c_set_v2", "test_c_set_v2"]
    # new_dirsが存在すればその中身を空にする
    # for new_dir in new_dirs:
    #     if os.path.exists(new_dir):
    #         shutil.rmtree(new_dir)
    #     os.makedirs(new_dir)

    # # src_dirsの中身をnew_dirsにコピー
    # for src_dir, new_dir in zip(src_dirs, new_dirs):
    #     for task in tqdm(os.listdir(src_dir), total=len(os.listdir(src_dir))):
    #         # もしtaskのディレクトリが存在しなければ作成
    #         if not os.path.exists(f"{new_dir}/{task}"):
    #             os.makedirs(f"{new_dir}/{task}")
    #         shutil.copy(f"{src_dir}/{task}/new_evaluations.json", f"{new_dir}/{task}/new_evaluations.json")
    #         for episode in os.listdir(f"{src_dir}/{task}"):
    #             # episodeがjsonであれば
    #             if episode.endswith(".json"):
    #                 continue
    #             if not os.path.exists(f"{new_dir}/{task}/{episode}"):
    #                 os.makedirs(f"{new_dir}/{task}/{episode}")
    #             for img in os.listdir(f"{src_dir}/{task}/{episode}"):
    #                 shutil.copy(f"{src_dir}/{task}/{episode}/{img}", f"{new_dir}/{task}/{episode}/{img}")
    
    # eliminate_listを元にnew_dirsの中身を削除（ただし、それぞれのタスクで上限100まで）
    new_dir = "valid_c_set_v2"
    tasks = os.listdir(new_dir)
    task_count = {}
    for task in tasks:
        task_count[task] = 0
    for eliminate in eliminate_list[:183]:
        os.remove(f"{new_dir}/{eliminate['image_path_1'].split('/')[1]}/{eliminate['image_path_1'].split('/')[2]}/{eliminate['image_path_1'].split('/')[3]}")
        os.remove(f"{new_dir}/{eliminate['image_path_2'].split('/')[1]}/{eliminate['image_path_2'].split('/')[2]}/{eliminate['image_path_2'].split('/')[3]}")
        task_count[eliminate["task"]] += 1
        if task_count[eliminate["task"]] >= 12:
            continue

if __name__ == "__main__":
    main()