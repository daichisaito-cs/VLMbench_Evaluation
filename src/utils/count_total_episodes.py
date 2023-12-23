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
    true_num = 0
    false_num = 0
    total = 0
    for task in os.listdir(directory):
        with open(os.path.join(directory, task, "new_evaluations.json")) as f:
            data = json.load(f)
        for episode in os.listdir(os.path.join(directory, task)):
            if episode.endswith(".json"):
                continue
            angles = ["overhead", "right", "left", "wrist"]
            for angle in angles:
                for file in os.listdir(os.path.join(directory, task, episode)):
                    if angle in file:
                        total += 1
                        if data[episode]["succeeded"]:
                            true_num += 1
                        else:
                            false_num += 1
                        break

    return true_num, false_num, total

def calculate_dataset_metrics(directories):
    total_words = 0
    vocab = set()
    sentence_lengths = []

    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for episode in data.values():
                            description = episode.get("description", "")
                            words = description.split()
                            total_words += len(words)
                            vocab.update(words)
                            sentence_lengths.append(len(words))

    average_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    vocab_size = len(vocab)
    total_word_count = total_words

    return average_sentence_length, vocab_size, total_word_count

train_dir = "data/train"
valid_dir = "data/valid"
test_dir = "data/test"

dirs = [train_dir, valid_dir, test_dir]
metrics = calculate_dataset_metrics([train_dir, valid_dir, test_dir])
print("Average Sentence Length:", metrics[0])
print("Vocabulary Size:", metrics[1])
print("Total Word Count:", metrics[2])

train_episodes = count_episodes(train_dir)
valid_episodes = count_episodes(valid_dir)
test_episodes = count_episodes(test_dir)

train_true, train_false, total = count_rates(train_dir)
test_true, test_false, total = count_rates(test_dir)

print("Train episodes:", train_episodes)
print("Valid episodes:", valid_episodes)
print("Test episodes:", test_episodes)
print("Total episodes:", train_episodes + valid_episodes + test_episodes)
print("\n")
print("Train true:", train_true)
print("Train false:", train_false)
print("\n")
print("Test true:", test_true)
print("Test false:", test_false)