import json
import os
import re
import numpy as np

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

def split_labels_by_hr(hr_ks_analysis_file, output_path):
        
if __name__ == "__main__":
    # concat_results("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_28_experiments/results/ucf101_multi_desc_llava/origin")
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
    get_10_25_50(origin_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_12_experiments/results/ans_file_9_12_llave_s40a_only_label_10patches_colbert_maxsim_mean_0_40.json", output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_12_experiments/results/ans_file_9_12_llave_s40a_only_label_10patches_colbert_maxsim_mean_hr102550.json")