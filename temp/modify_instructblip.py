import os
import json
# instruct_blip内のディレクトリをfor文で回す
search_dir = "/home/initial/workspace/VLMbench_Evaluation/temp/instruct_blip"
for split in os.listdir(search_dir):
    for json_file in os.listdir(f"{search_dir}/{split}"):
        with open(f"{search_dir}/{split}/{json_file}", "r") as f:
            json_data = json.load(f)
            new_json_data = {}
            # jsonファイルのキーを変更
            for key in json_data.keys():
                new_key = key.replace("/home/initial/workspace/VLMbench_Evaluation/train_c_set_v2", "train").replace("/home/initial/workspace/VLMbench_Evaluation/valid_c_set_v2", "valid").replace("/home/initial/workspace/VLMbench_Evaluation/test_c_set_v2", "test")
                new_json_data[new_key] = json_data[key]
    
        with open(f"{search_dir}/{split}/{json_file}", "w") as f:
            json.dump(new_json_data, f, indent=4)