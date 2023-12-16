import os
import json

def count_episodes(directory):
    count = 0
    true_num = 0
    false_num = 0
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.startswith("episode"):
                count += len(os.listdir(os.path.join(root, dir))) / 2
            # もしjsonファイルがあれば、その中身を確認する
    return count

def count_rates(directory):
    count = 0
    true_num = 0
    false_num = 0
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            with open(os.path.join(root, dir, "new_evaluations.json")) as f:
                data = json.load(f)
                for i in data.values():
                    if i["succeeded"]:
                        true_num += 1
                    else:
                        false_num += 1
        break
    return true_num, false_num

train_dir = "train_c_set_v2"
valid_dir = "valid_c_set_v2"
test_dir = "test_c_set_v2"

train_episodes = count_episodes(train_dir)
valid_episodes = count_episodes(valid_dir)
test_episodes = count_episodes(test_dir)

train_true, train_false = count_rates(train_dir)

print("Train episodes:", train_episodes)
print("Valid episodes:", valid_episodes)
print("Test episodes:", test_episodes)
print("Total episodes:", train_episodes + valid_episodes + test_episodes)
print("Train true:", train_true)
print("Train false:", train_false)