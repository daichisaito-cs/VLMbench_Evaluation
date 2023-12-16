import os
import json
# instruct_blip内のディレクトリをfor文で回す
search_dir = "/home/initial/workspace/VLMbench_Evaluation/data"
splits = ["train", "valid", "test"]

for split in splits:
    for task_dir in os.listdir(f"{search_dir}/{split}"):
        for file in os.listdir(f"{search_dir}/{split}/{task_dir}"):
            if file.endswith(".json"):
                with open(f"{search_dir}/{split}/{task_dir}/{file}") as config_file:
                    config = json.load(config_file)
                for key in config.keys():
                    config[key]["embedding_path"] = config[key]["embedding_path"].replace("VLMbench_Evaluation", "VLMbench_Evaluation/data")
                with open(f"{search_dir}/{split}/{task_dir}/{file}", "w") as config_file:
                    json.dump(config, config_file, indent=4)
