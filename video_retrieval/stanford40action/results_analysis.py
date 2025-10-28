import json
import os
import re
import numpy as np
from PIL import Image

from video_retrieval.stanford40action.utils import draw_border, concatenate_images_grid_with_highlight

def extract_range_from_filename(filename):
    match = re.search(r'_(\d+)_(\d+)(?:_results)?\.json$', filename)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return (start, end)
    return (0, 0) 

def concat_results(results_dir, output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_llava/origin/origin_results.json"):
    results_paths = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.find("results") != -1:
                results_paths.append(os.path.join(root, f))
    results_paths.sort(key=extract_range_from_filename)
    data = []
    for result in results_paths:
        # print(result) # /research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_e5v/origin/ans_file_8_28_e5v_multi_descriptions_cosine_max_mean_0_49_results.json
        with open(result, "r") as f:
            r_data = json.load(f)
        data += r_data
    breakpoint()
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def question_to_label(dataset_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/datasets/ucf101_vqa_multiple_descriptions.json"):
    with open(dataset_dir, "r") as f:
        data = json.load(f)
    qa_pairs = data["qa_pairs"]
    q_to_l = {}
    for qa_pair in qa_pairs:
        q_to_l[qa_pair["question"]] = qa_pair["action"]
    return q_to_l

def get_action_mam(output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_llava/origin/hr_results.json"):
    with open("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_llava/origin/origin_results.json", "r") as f:
        data = json.load(f)
    q_to_l = question_to_label()
    analysis = {}
    for q in data:
        action = q_to_l[q["Question"]]
        lst = analysis.get(action, [])
        lst.append(q)
        analysis[action] = lst
    results = []
    for action, r_lst in analysis.items():
        hit_rate_lst = [r["hit_rate"] for r in r_lst]
        content = {
            "action": action,
            "max_hr": max(hit_rate_lst),
            "min_hr": min(hit_rate_lst),
            "avg_hr": sum(hit_rate_lst) / len(hit_rate_lst),
            "hr_lst": hit_rate_lst
        }
        results.append(content)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

def concat_origin(results_dir, output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_e5v/origin/origin.json"):
    results_paths = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.find("results") == -1:
                results_paths.append(os.path.join(root, f))
    results_paths.sort(key=extract_range_from_filename)
    data = []
    for result in results_paths:
        # print(result) # /research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_e5v/origin/ans_file_8_28_e5v_multi_descriptions_cosine_max_mean_0_49_results.json
        with open(result, "r") as f:
            r_data = json.load(f)
        data += r_data
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def get_10_25_50(origin_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_e5v/origin/origin.json", output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_e5v/origin/hr102550.json"):
    with open(origin_path, "r") as f:
        data = json.load(f)
    l_to_q_hr = {}
    total_hr_10_25_50_100 = []
    for q in data:
        l_lst = l_to_q_hr.get(q["gt_action"], [])
        n10, n25, n50 = int(0.1*q["gt_num"]), int(0.25*q["gt_num"]), int(0.5*q["gt_num"])
        n100 = q["gt_num"]
        hit = [1 if item["action_category"] == q["gt_action"] else 0 for item in q["items"]]
        hr_10_25_50_100 = [sum(hit[:n10])/n10, sum(hit[:n25])/n25, sum(hit[:n50])/n50, sum(hit)/n100]
        total_hr_10_25_50_100.append(hr_10_25_50_100)
        content = {
            "question": q["question"],
            "n_10_25_50_100": [n10, n25, n50, n100],
            "hr_10_25_50_100": hr_10_25_50_100
        }
        l_lst.append(content)
        l_to_q_hr[q["gt_action"]] = l_lst
    total_hr_10_25_50_100_mat = np.array(total_hr_10_25_50_100)
    total_hr_10_25_50_100 = total_hr_10_25_50_100_mat.mean(axis=0).tolist()
    result = {
        "total_hr_10_25_50_100": total_hr_10_25_50_100,
        "content": l_to_q_hr
        }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

def get_hitrate_ks(origin_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/9_3_experiments/results/llava_s40a_multi_phrase_none_colbert_tml5/origin/llava_s40a_multi_phrase_none_colbert_tml5_origin.json", output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/9_3_experiments/results/llava_s40a_multi_phrase_none_colbert_tml5/origin/llava_s40a_multi_phrase_none_colbert_tml5_hr_ks.json", k_lst=[1, 2, 3, 5, 10, 20]):
    with open(origin_path, "r") as f:
        data = json.load(f)
    hr_lst = []
    for q in data:
        gt_v_id = q["gt_image_ids"]
        hit_rank = len(q["retrieved_image_ids"])
        for rank, v_id in enumerate(q["retrieved_image_ids"]):
            if v_id == gt_v_id[0]:
                hit_rank = rank
        hitrate_ks = {k: (1 if hit_rank <= (k-1) else 0) for k in k_lst}
        content = {
            "question": q["question"],
            "k_lst": k_lst,
            "hit_rank": hit_rank,
            "hitrate_ks": hitrate_ks,
            "gt_action": q["gt_action"]
        }
        hr_lst.append(content)

    with open(output_path, "w") as f:
        json.dump(hr_lst, f, indent=2)


def get_hitrate_ks_analysis(hitrate_ks_file, output_path):
    with open(hitrate_ks_file, "r") as f:
        data = json.load(f)

    final_result = {}
    group_by_action = {}
    total_hit_rank = []
    for q in data:
        gt_action = q["gt_action"]
        content = group_by_action.get(gt_action, {})
        total_hit_rank.append(q["hit_rank"])
        if not content:
            content = {
                "questions": [q["question"]],
                "hitrate_ks": [list(q["hitrate_ks"].values())],
                "k_lst": q["k_lst"],
                "hit_ranks": [q["hit_rank"]],
                "avg_hitrate_ks": [],
                "avg_hit_rank": "tobedone",
                "items": [q] 
            }
            group_by_action[gt_action] = content
        else:
            content["questions"].append(q["question"])
            content["hitrate_ks"].append(list(q["hitrate_ks"].values())),
            content["hit_ranks"].append(q["hit_rank"]),
            content["items"].append(q)
            group_by_action[gt_action] = content

    for action, content in group_by_action.items():
        content["avg_hitrate_ks"] = np.array(content["hitrate_ks"]).mean(axis=0).tolist()
        # content["avg_hit_rank"] = int(np.array(content["hit_ranks"]).mean())
    
    final_result["k_lst"] = list(group_by_action.values())[0]["k_lst"]
    total_hit_rank = np.array(total_hit_rank)
    # final_result["avg_hit_rank"] = int(total_hit_rank.mean())
    final_result["avg_hitrate_ks"] = [float((total_hit_rank <= (r-1)).astype(int).mean()) for r in final_result["k_lst"]]
    max_k = max(final_result["k_lst"])
    final_result["total_hit_rank"] = [r if r < max_k else None for r in total_hit_rank.tolist()]
    final_result["group_by_action"] = group_by_action

    with open(output_path, "w") as f:
        json.dump(final_result, f, indent=2)

def get_hitrate_ks_hybrid(origin_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_7_experiments/results/llave_s40a_hybrid_no_patches_colbert_maxsim_mean/origin.json", output_path=None, k_lst=[1, 2, 3, 5, 10, 20]):
    with open(origin_path, "r") as f:
        data = json.load(f)
    hr_lst = []
    for q in data:
        gt_v_id = q["gt_image_id"] + ".jpg"
        for i, item in enumerate(q["items"]):
            if item["image_id"] == gt_v_id:
                hit_rank = i
                similarity = item["similarity"]

        hitrate_ks = {k: (1 if hit_rank <= (k-1) else 0) for k in k_lst}
        content = {
            "question_text": q["question_text"],
            "question_image": os.path.basename(q["question_image"]),
            "k_lst": k_lst,
            "hit_rank": hit_rank,
            "similarity": similarity,
            "hitrate_ks": hitrate_ks,
            "gt_action": q["gt_action"]
        }
        hr_lst.append(content)

    with open(output_path, "w") as f:
        json.dump(hr_lst, f, indent=2)

def get_hitrate_ks_analysis_hybrid(hitrate_ks_file, output_path):
    with open(hitrate_ks_file, "r") as f:
        data = json.load(f)

    final_result = {}
    group_by_question_img = {}
    total_hit_rank = []
    for q in data:
        question_img = q["question_image"]
        question_text = q["question_text"]
        content = group_by_question_img.get(question_img, {})
        total_hit_rank.append(q["hit_rank"])
        if not content:
            content = {
                "question_texts": [question_text],
                "hitrate_ks": [list(q["hitrate_ks"].values())],
                "k_lst": q["k_lst"],
                "hit_ranks": [q["hit_rank"]],
                "avg_hitrate_ks": [],
                "avg_hit_rank": "tobedone",
                "items": [q] 
            }
            group_by_question_img[question_img] = content
        else:
            content["question_texts"].append(question_text)
            content["hitrate_ks"].append(list(q["hitrate_ks"].values())),
            content["hit_ranks"].append(q["hit_rank"]),
            content["items"].append(q)
            group_by_question_img[question_img] = content

    for question_img, content in group_by_question_img.items():
        content["avg_hitrate_ks"] = np.array(content["hitrate_ks"]).mean(axis=0).tolist()
        # content["avg_hit_rank"] = int(np.array(content["hit_ranks"]).mean())
    
    final_result["k_lst"] = list(group_by_question_img.values())[0]["k_lst"]
    total_hit_rank = np.array(total_hit_rank)
    # final_result["avg_hit_rank"] = int(total_hit_rank.mean())
    final_result["avg_hitrate_ks"] = [float((total_hit_rank <= (r-1)).astype(int).mean()) for r in final_result["k_lst"]]
    max_k = max(final_result["k_lst"])
    final_result["total_hit_rank"] = total_hit_rank.tolist()
    final_result["group_by_question_img"] = group_by_question_img

    with open(output_path, "w") as f:
        json.dump(final_result, f, indent=2)

def image_id_to_path(image_id, dataset_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset"):
    image_id = image_id.split(".")[0] if image_id.find(".") != -1 else image_id
    label = image_id.rsplit("_", 1)[0]
    train = os.path.join(dataset_dir, "train", label)
    test = os.path.join(dataset_dir, "test", label)
    image_path = ""
    for root, dirs, files in os.walk(train):
        for f in files:
            if f.split(".")[0] == image_id:
                image_path = os.path.join(root, f)

    for root, dirs, files in os.walk(test):
        for f in files:
            if f.split(".")[0] == image_id:
                image_path = os.path.join(root, f)
    return image_path
    
def get_big_image(original_file, output_dir):
    with open(original_file, "r") as f:
        data = json.load(f)      

    for item in data:
        img_name = f"{item['question_idx']}_{os.path.basename(item['question_image']).rsplit('.', 1)[0]}_{item['gt_image_id']}.jpg"  
        img_path = os.path.join(output_dir, img_name)
        image_ids = item["retrieved_image_ids"][:20]
        image_paths = [image_id_to_path(image_id) for image_id in image_ids]
        pil_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        gt_image_id = item["gt_image_id"] if isinstance(item["gt_image_id"], str) else item["gt_image_id"][0]
        rank = []
        for r in item["items"]:
            if r["image_id"].split(".")[0] == gt_image_id:
                rank.append(r["rank"])
  
        concatenate_images_grid_with_highlight(
            pil_images, 
            grid_rows=4, 
            grid_cols=5, 
            target_index=rank, 
            border_width=5, 
            border_color=(255, 0, 0), 
            save_path=img_path
        )

def get_big_image_i2i(original_file, output_dir):
    with open(original_file, "r") as f:
        data = json.load(f)      

    for item in data:
        img_name = f"{item['question_idx']}_{item['question'].rsplit('.', 1)[0]}.jpg"  
        img_path = os.path.join(output_dir, img_name)
        image_ids = item["retrieved_image_ids"][:20]
        image_paths = [image_id_to_path(image_id) for image_id in image_ids]
        pil_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        gt_image_ids = item["gt_image_ids"] 
        rank = []
        for r in item["items"][:20]:
            if r["image_id"] in gt_image_ids:
                rank.append(r["rank"])
  
        concatenate_images_grid_with_highlight(
            pil_images, 
            grid_rows=4, 
            grid_cols=5, 
            target_index=rank, 
            border_width=5, 
            border_color=(255, 0, 0), 
            save_path=img_path
        )

def get_results(origin_path):
    root = os.path.dirname(origin_path)
    get_hitrate_ks_hybrid(
        origin_path=origin_path, 
        output_path=os.path.join(root, "hr_ks.json"), 
        )
    get_hitrate_ks_analysis_hybrid(
        hitrate_ks_file=os.path.join(root, "hr_ks.json"), 
        output_path=os.path.join(root, "hr_ks_analysis.json")
        )

if __name__ == "__main__":
    # origin_paths = []
    # for root, ds, fs in os.walk("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results_2images/10_22_llave_s40a_imageText2image_no_patches_2imagesquery_colbert_weighted_query_token_sum_50_50"):
    #     for f in fs:
    #         if "ans_file" in f:
    #             origin_paths.append(os.path.join(root, f))
    # for origin_path in origin_paths:
    #     get_results(origin_path)
    
    get_big_image(
        original_file="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results_haolin2/results_haolin3/10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_40_60/ans_file_10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_40_60_0_18.json", 
        output_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results_haolin2/results_haolin3/10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_40_60/images"
        )
    get_big_image(
        original_file="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results_haolin2/results_haolin3/10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_50_50/ans_file_10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_50_50_0_18.json", 
        output_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results_haolin2/results_haolin3/10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_50_50/images"
        )
    get_big_image(
        original_file="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results_haolin2/results_haolin3/10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_60_40/ans_file_10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_60_40_0_18.json", 
        output_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results_haolin2/results_haolin3/10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum_60_40/images"
        )

    # origin_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results/10_22_llave_s40a_imageText2image_no_patches_colbert_weighted_query_token_sum_50_50/ans_file_10_22_llave_s40a_imageText2image_no_patches_colbert_weighted_query_token_sum_50_50_0_210.json"
    # get_results(origin_path)
    # concat_origin(
    #     results_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_7_experiments/results/llave_s40a_hybrid_no_patches_colbert_maxsim_mean/origin",
    #     output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_7_experiments/results/llave_s40a_hybrid_no_patches_colbert_maxsim_mean/origin.json"
    #     )
    # get_hitrate_ks_hybrid(
    #     origin_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results/10_22_llave_s40a_imageText2image_no_patches_1_9_colbert_weighted_query_token_sum/ans_file_10_22_llave_s40a_imageText2image_no_patches_1_9_colbert_weighted_query_token_sum_0_210.json", 
    #     output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_15_experiments/results/llave_s40a_imageText2image_no_patches_9_91_colbert_weighted_query_token_sum/9_91_hr_ks.json", 
    #     )
    # get_hitrate_ks_analysis_hybrid(
    #     hitrate_ks_file="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_15_experiments/results/llave_s40a_imageText2image_no_patches_9_91_colbert_weighted_query_token_sum/9_91_hr_ks.json", 
    #     output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_15_experiments/results/llave_s40a_imageText2image_no_patches_9_91_colbert_weighted_query_token_sum/9_91_hr_ks_analysis.json"
    #     )
    # 
    # get_big_image_i2i(
    #     original_file="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_22_experiments/results/9_22_llave_s40a_image2image_nopatches_colbert_maxsim_mean/ans_file_9_22_llave_s40a_image2image_nopatches_colbert_maxsim_mean_0_42.json", 
    #     output_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_22_experiments/results/9_22_llave_s40a_image2image_nopatches_colbert_maxsim_mean/images"
    #     )
    # get_big_image(
    #     original_file="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_8_experiments/results/llave_s40a_imageText2image_10_patches_colbert_maxsim_mean/ans_file_10_8_llave_s40a_imageText2image_10_patches_colbert_maxsim_mean_0_210.json", 
    #     output_dir="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_8_experiments/results/llave_s40a_imageText2image_10_patches_colbert_maxsim_mean/images"
    #     )

    # get_action_mam()
    # concat_origin(
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_11_experiments/results/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean/origin", 
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_11_experiments/results/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean/origin/origin.json"
    #     )
    # get_hitrate_ks(
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_11_experiments/results/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean/origin/origin.json",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_11_experiments/results/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean/origin/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean_hr_ks.json"
    # )
    # get_hitrate_ks_analysis(
    #     hitrate_ks_file="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_11_experiments/results/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean/origin/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean_hr_ks.json",
    #     output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_11_experiments/results/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean/origin/vlm2vec_s40a_merged_multi_query_template5_nopatches_colbert_maxsim_mean_final_results.json"
    #     )
    # get_10_25_50(origin_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_12_experiments/results/ans_file_9_12_llave_s40a_only_label_10patches_colbert_maxsim_mean_0_40.json", output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_12_experiments/results/ans_file_9_12_llave_s40a_only_label_10patches_colbert_maxsim_mean_hr102550.json")
    # with open("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_6_experiments/datasets/ImageText2Image4.json", "r") as f:
    #     data = json.load(f)
    # query_imgs = []
    # for item in data["qa_pairs"]:
    #     query_imgs.append(item["question_image"])
    # query_imgs = set(query_imgs)
    # output_dir = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_8_experiments/results/query_imgs"
    # for path in query_imgs:
    #     img = Image.open(path).convert("RGB")
    #     img_new_path = os.path.join(output_dir, os.path.basename(path))
    #     img.save(img_new_path)
