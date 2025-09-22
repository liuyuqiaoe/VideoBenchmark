import os
from shutil import copyfile
import json
import random
import glob
import re
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def select_images(k, train_split_dir, output_path, label_filtering=None):
    action_to_images = {}
    img_extensions = {"jpg", "jpeg"}
    for root, dirs, files in os.walk(train_split_dir):
        for action in dirs:
            action_to_images[action] = action_to_images.get(action, [])
        for img in files:
            img_suffix = img.split(".")[-1].lower().strip()
            if img_suffix in img_extensions:
                img_path = os.path.join(root, img)
                action = os.path.basename(root)
                action_lst = action_to_images.get(action, [])
                action_to_images[action] = action_lst + [img_path]

    selected_images = {}

    for action, imgs in action_to_images.items():
        selected_images[action] = random.sample(imgs, k=k)

    if label_filtering:
        selected_images = {k: selected_images[k] for k in label_filtering}

    with open(output_path, "w") as f:
        json.dump(selected_images, f, indent=2)

    print(f"Dumped {len(action_to_images)} actions to path ")
    return action_to_images

# copied from others' work
def creating_dataset():
    dataset_dir = os.path.join(os.getcwd(), "Stanford40Actions")
    images_path = dataset_dir + "/JPEGImages"
    labels_path = dataset_dir + "/ImageSplits"
    new_dataset_path = dataset_dir + "/StanfordActionDataset"

    if not (os.path.exists(new_dataset_path)):
        os.mkdir(new_dataset_path)
        os.mkdir(new_dataset_path + '/' + 'train')
        os.mkdir(new_dataset_path + '/' + 'test')

    txts = os.listdir(labels_path)
    for txt in txts:
        idx = txt[0:-4].rfind('_')
        class_name = txt[0:idx]
        if class_name in ['actions.tx', 'test.tx', 'train.tx']:
            continue
        train_or_test = txt[idx + 1:-4]
        txt_contents = open(labels_path + '/' + txt)
        txt_contents = txt_contents.read()
        image_names = txt_contents.split('\n')
        num_aid_images_per_class = 1
        for image_name in image_names[0:-1]:
            if not (os.path.exists(new_dataset_path + '/' + train_or_test + '/' + class_name)):
                os.mkdir(new_dataset_path + '/' + train_or_test + '/' + class_name)
            copyfile(images_path + '/' + image_name,
                     new_dataset_path + '/' + train_or_test + '/' + class_name + '/' + image_name)

def create_image_batches(images_dir, output_path, batch_size=128):
    images_lst = []
    img_extensions = {"jpg", "jpeg"}
    
    for root, dirs, files in os.walk(images_dir):
        for img_file in files:
            if img_file.split(".")[-1].lower().strip() in img_extensions:
                img_path = os.path.join(root, img_file)
                images_lst.append(img_path)
    img_num = len(images_lst)
    batch_num = ((img_num - 1) // batch_size) + 1
    batch_metadata = {
        "total_images": img_num,
        "batch_size": batch_size,
        "total_batches": batch_num,
        "image_file_paths": images_lst
    }
    batches = []
    for batch_idx in range(batch_num):
        start = batch_idx * batch_size
        end = min(img_num - 1, start + batch_size - 1)
        batch = {
            "batch_index": batch_idx,
            "batch_size": end - start + 1,
            "start_index": start,
            "end_index": end,
            "image_paths": images_lst[start:end+1]
        }
        batches.append(batch)

    batch_data = {
        "metadata": batch_metadata,
        "batches": batches
    }

    with open(output_path, "w") as f:
        json.dump(batch_data, f, indent=2)
    
    return batch_data

def get_image_batches(batch_file_path, start_idx, end_idx):
    with open(batch_file_path, "r") as f:
        batch_data = json.load(f)
    batch_num = batch_data["metadata"]["total_batches"]
    if start_idx < 0 or start_idx > end_idx or end_idx >= batch_num:
        print(f"Invalid batch range {start_idx}-{end_idx}. Valid range 0-{batch_num-1}")
    batches = batch_data.get("batches", [])
    
    output = {}
    for batch_idx in range(start_idx, end_idx + 1):
        assert batches[batch_idx]["batch_index"] == batch_idx
        output[batch_idx] = batches[batch_idx]["image_paths"]
    
    return output

def extract_range_from_filename(filename):
    match = re.search(r'_(\d+)_(\d+)(?:_results)?\.json$', filename)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return (start, end)
    return (0, 0)  

def concat_json_files(dir_path=None, pattern="*return_all*results.json"):
    original_dir = os.getcwd()
    os.chdir(dir_path)
    
    files = glob.glob(pattern)
    files.sort(key=extract_range_from_filename)
    
    all_data = []
    
    for file_path in files:
        start, end = extract_range_from_filename(file_path)
        print(f"Processing: {file_path} (range: {start}-{end})")
    
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_data.extend(data)
                
    output_file = os.path.join(dir_path, "concatenated_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    
    os.chdir(original_dir)
    return output_file

def parse_phrases(response):
    phrases = []
    lines = response.split("\n")
    if len(lines) != 1:
        for line in lines:
            if ":" in line and "-" in line:
                phrase = line.split(":")[1].strip()
                phrases.append(phrase)
    
    return phrases

if __name__ == "__main__":
    # pattern_return_all = "*return_all_[0-9]*_[0-9]*.json"
    # output_file = concat_json_files(dir_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_30_experiments/results/e5v_s40a_single_image", pattern=pattern_return_all)
    # select_images(10, "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train", "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/selected_images.json")
    # create_image_batches(batch_size=128, images_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/JPEGImages", output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/s40a/image_batches.json")
    # output = get_image_batches("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/s40a/image_batches.json", start_idx=0, end_idx=2)
    
    sampled_label_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_16_experiments/results/label_split_basedon_query_nopatches/label_sampled5.json"
    output_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_16_experiments/results/label_split_basedon_query_nopatches/selected_images.json"
    with open(sampled_label_path, "r") as f:
        content = json.load(f)
    labels = []
    for k, v in content.items():
        labels += v
    select_images(k=40, train_split_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train", output_path=output_path, label_filtering=labels)
    
    
 