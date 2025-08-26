import json
import os
from typing import Dict, List, Any, Optional, Callable
from video_retrieval.encoders.intern_video2_encoder import InternVideo2Encoder
from video_retrieval.encoders.e5v_encoder import E5VVideoEncoder
from video_retrieval.encoders.llava_qwen_encoder import LLaVAQwenEncoder
from tqdm import tqdm
from video_retrieval.video_indexer import LanceDBVideoIndex, LanceDBVideoRetriever


class UFC101QADataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = self.load_ufc101_qa_dataset()
        qa_pairs = self.get_qa_pairs()
        if len(qa_pairs) > 0:
            self.qa_pairs = qa_pairs
        else:
            self.vqa_pairs = self.get_vqa_pairs()
        self.metadata = self.get_metadata()
    
    def load_ufc101_qa_dataset(self):
        if os.path.isfile(self.dataset_path):
            with open(self.dataset_path, "r") as f:
                dataset = json.load(f)
            return dataset
        else:
            print(f"Path does not exist: {self.dataset_path}")
            return {}
    
    def get_metadata(self):
        metadata = {
            "dataset_name": self.dataset.get("dataset_name", "None"),
            "description": self.dataset.get("description", "None"),
            "dataset_path": self.dataset.get("dataset_path", "None"),
            "total_questions": self.dataset.get("total_questions", -1),
            "total_videos": self.dataset.get("total_videos", -1)
        }
        print(f"Dataset Metadata: {metadata}")
        return metadata
        
    def get_qa_pairs(self):
        return self.dataset.get("qa_pairs", [])
    
    def get_vqa_pairs(self):
        return self.dataset.get("vqa_pairs", [])

    # qa
    def get_idx_qa_mapping(self):
        qas = []
        qa_to_idx = {}
       
        for idx, qa_pair in enumerate(self.qa_pairs):
            query = qa_pair["question"]
            label = qa_pair["action"]
            gt_num = qa_pair["gt_video_count"]
            qas.append((query, label))
            q_to_idx[query] = idx
        return qas, qa_to_idx

class DatasetTester:
    def __init__(self, db_path, encoder, similarity_type=None):
        self.db_path = db_path
        self.encoder = encoder
        self.similarity_type = similarity_type
        self.retriever = LanceDBVideoRetriever(encoder=self.encoder, db_path=self.db_path)

    def load_encoder(self):
        pass

    def test_dataset(
        self, 
        dataset: UFC101QADataset, 
        ans_path = " ", 
        return_all = False,
        where_clause = False,
        only_label = False
    ):  
        qa_pairs = getattr(dataset, "qa_pairs", getattr(dataset, 'vqa_pairs', None))

        for idx, qa_pair in enumerate(tqdm(qa_pairs, desc="Retrieving answers")):

            # only label
            if only_label:
                query = qa_pair["action"]
            else:
                query = qa_pair["question"]
            ans_lst = []
            gt_video_ids = qa_pair["video_ids"]
            gt_action = qa_pair["action"]
            gt_num = len(gt_video_ids)
            queries = [query]
            # where clause (to get gt similarities)
            if where_clause:
                results = self.retriever.search(
                    queries=queries, 
                    top_k=gt_num, 
                    where_clause=f"action_category = '{gt_action}'"
                )
            else:
                # return all
                results = self.retriever.search(
                    queries=queries, 
                    top_k=gt_num, 
                    return_all=return_all
                )
            results = list(results.values())[0]
            retrieved_video_ids = []
            items = []
            similarities = []
            hit = 0
            
            for i, result in enumerate(results):
                retrieved_video_ids.append(result[2]["video_name"])
                similarities.append(result[1])
                
                item = {
                    "video_id": result[2]["video_name"], 
                    "similarity": result[1],
                    "search_time": result[2]["search_time"],
                    "action_category": result[2]["action_category"],
                    "query": result[2]["query_text"],
                    "encoder_type": result[2]["encoder_type"]
                }
                items.append(item)
                
                if i < gt_num and result[2]["action_category"].strip().lower() == gt_action.strip().lower():
                    hit += 1
            
            ans = {
                "question_idx": idx,
                "question": query,
                "gt_video_ids": gt_video_ids,
                "gt_num": gt_num,
                "gt_action": gt_action,
                "retrieved_video_ids": retrieved_video_ids,
                "retrieved_num": len(retrieved_video_ids),
                "items": items,
                "avg_similarity": sum(similarities) / len(retrieved_video_ids) if retrieved_video_ids else 0,
                "max_similarity": max(similarities) if similarities else 0,
                "min_similarity": min(similarities) if similarities else 0,
                "hit": hit
            }
            
            ans_lst.append(ans)
        
        with open(ans_path, 'w', encoding='utf-8') as f:
            json.dump(ans_lst, f, indent=2, ensure_ascii=False)
        
        print(f"Results dumped to: {ans_path}")
        print(f"Total results: {len(ans_lst)}")
        
        return ans_path
    
    def test_dataset_batch_query(
        self, 
        dataset: UFC101QADataset, 
        ans_path = " ", 
        return_all = False,
        where_clause = False,
        only_label = False
    ):  
        qa_pairs = getattr(dataset, "qa_pairs", getattr(dataset, 'vqa_pairs', None))
        query_to_idx = {}
        qs  = []
        topk = 0
        topk_lst = []
        for idx, qa_pair in enumerate(qa_pairs):
            query_to_idx[qa_pair["question"]] = idx
            qs.append(qa_pair["question"])
            topk = max(topk, qa_pair["video_count"])
            topk_lst.append(qa_pair["video_count"])

        batch_size = 100
        total_batch = (len(qs) -1) // batch_size + 1
        ans = []
        ans_file = []
    
        for batch_idx in tqdm(range(total_batch)):
            start = batch_idx * batch_size
            end = min(start + batch_size - 1, len(qa_pairs))
            ans_path_batch = ans_path.split(".")[0] + f"_{start}_{end}.json"
            qs_batch = qs[start:end+1]
            topk_lst_batch = topk_lst[start:end+1]
            if self.similarity_type:
                results_multi_queries = self.retriever.search(
                    queries=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    similarity_type=self.similarity_type
                )
            else:
                results_multi_queries = self.retriever.search(
                    queries=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all
                )
            results_multi_queries = self.retriever.formated_results(results_multi_queries)    
            
            # self.retriever.dump_results(results_multi_queries,ans_path_batch)
            for query_idx, query_result in results_multi_queries.items():
                hit = 0
                similarities = []
                retrieved_video_ids = []
                items = []
                results = query_result["results"]
                question = query_result["query_text"]
                question_idx = query_to_idx[question]
                qa_pair = qa_pairs[question_idx]
                gt_num = qa_pair["video_count"]
                for i, result in enumerate(results):
                    retrieved_video_ids.append(result["metadata"]["video_name"])
                    item = {
                        "video_id": result["metadata"]["video_name"],
                        "similarity": result["similarity"],
                        "rank": result["rank"],
                        "search_time": result["metadata"]["search_time"],
                        "action_actegory": result["metadata"]["action_category"],
                        "video_index": result["metadata"]["video_index"],
                        "encoder_type": result["metadata"]["encoder_type"],
                        "embedding_dim": result["metadata"]["embedding_dim"],
                        "similarity_type": result["metadata"]["similarity_type"],
                        "query_embedding_shape": result["metadata"]["query_embedding_shape"],
                        "video_embedding_shape": result["metadata"]["video_embedding_shape"]
                    }
                    items.append(item)
                    similarities.append(result["similarity"])

                    if i < gt_num and result["metadata"]["action_category"].strip().lower() == qa_pair["action"].strip().lower():
                        hit += 1

                ans = {
                    "question_idx": question_idx,
                    "question": question,
                    "gt_video_ids": qa_pair["video_ids"],
                    "gt_num": qa_pair["video_count"],
                    "gt_action": qa_pair["action"],
                    "retrieved_video_ids": retrieved_video_ids,
                    "retrieved_num": len(results),
                    "items": items,
                    "avg_similarity": sum(similarities) / len(retrieved_video_ids) if retrieved_video_ids else 0,
                    "max_similarity": max(similarities) if similarities else 0,
                    "min_similarity": min(similarities) if similarities else 0,
                    "hit": hit
                }
                ans_file.append(ans)
            
            with open(ans_path, "w", encoding='utf-8') as f:
                json.dump(ans_file, f, indent=2, ensure_ascii=False)
        
        print(f"Results dumped to: {ans_path}")
        print(f"Total queries: {len(ans_file)}")

        return ans_path

    def get_score(self, ans_file = " "):
        with open(ans_file, "r") as f:
            answers = json.load(f)
        
        results = []

        for ans in answers:
            gt_num = ans["gt_num"]
            gt_video_ids = set(ans["gt_video_ids"])
            retrieved_video_ids = ans["retrieved_video_ids"]
            retrieved_num = ans["retrieved_num"]
            hit_by_id = 0
            
            for i, v in enumerate(retrieved_video_ids):
                if i < gt_num and (v in gt_video_ids):
                    hit_by_id += 1
                    gt_video_ids.remove(v)
            
            clean_result = {
                "Question": ans["question"],
                "gt_num": gt_num,
                "hit_rate": hit_by_id / gt_num if gt_num > 0 else 0,
                "avg_similarity": ans["avg_similarity"],
                "max_similarity": ans["max_similarity"],
                "min_similarity": ans["min_similarity"]
            }
            results.append(clean_result)
        
        ans_file_root = os.path.dirname(ans_file)
        result_file_name = os.path.basename(ans_file).split(".")[0] + "_results.json"
        results_path = os.path.join(ans_file_root, result_file_name)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results_path


def run_test_configurations(config, db_path, encoder, similarity_type):
    tester = DatasetTester(db_path, encoder, similarity_type)
    
    for key, item in config.items():
        print(f"\n{'='*50}")
        print(f"Testing: {key}")
        print(f"{'='*50}")
        
        dataset_path = item["dataset_path"]
        dataset = UFC101QADataset(dataset_path)
        ans_path = tester.test_dataset_batch_query(
            dataset=dataset,
            ans_path=item["ans_path"],
            return_all=item.get("return_all", False),
            where_clause=item.get("where_clause", False),
            only_label=item.get("only_label", False)
        )
        
        results_path = tester.get_score(ans_path)
        print(f"Results stored at: {results_path}")

def test_dataset(dataset: UFC101QADataset, db_path, ans_path, return_all = False):
    tester = DatasetTester(db_path)
    return tester.test_dataset(dataset, ans_path, return_all=return_all)


def test_dataset_action_label_query(dataset: UFC101QADataset, db_path, ans_path, return_all = False):
    tester = DatasetTester(db_path)
    return tester.test_dataset(
        dataset, ans_path, 
        return_all=return_all, 
        use_action_as_query=True
    )


def get_gt_similarity(dataset: UFC101QADataset, db_path, ans_path, only_label = False):
    tester = DatasetTester(db_path)
    return tester.test_dataset(
        dataset, ans_path,
        where_clause=True,
        only_label=only_label
    )


def get_score(ans_file):
    tester = DatasetTester("")  # db_path not needed for scoring
    return tester.get_score(ans_file)


if __name__ == "__main__":
    # You can change the paths
    # config = {
    #     "ans_file": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_real_qa_dataset.json",
    #         "ans_path": os.path.join(ans_file_root,"ans_file.json"),
    #         "return_all": False,
    #         "where_clause": False,
    #         "only_label": False
    #     },
    #     "ans_file_only_label": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_real_qa_dataset.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_only_label.json"),
    #         "return_all": False,
    #         "where_clause": False,
    #         "only_label": True
    #     },
    #     "ans_file_replaced": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_replaced_dataset.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_replaced.json"),
    #         "return_all": False,
    #         "where_clause": False,
    #         "only_label": False
    #     },
    #     "ans_file_gt": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_real_qa_dataset.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_gt.json"),
    #         "return_all": False,
    #         "where_clause": True,
    #         "only_label": False
    #     },
    #     "ans_file_only_label_gt": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_real_qa_dataset.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_only_label_gt.json"),
    #         "return_all": False,
    #         "where_clause": True,
    #         "only_label": True
    #     },
    #     "ans_file_replaced_gt": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_replaced_dataset.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_replaced_gt.json"),
    #         "return_all": False,
    #         "where_clause": True,
    #         "only_label": False
    #     },
    #     "ans_file_return_all": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_real_qa_dataset.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_return_all.json"),
    #         "return_all": True,
    #         "where_clause": False,
    #         "only_label": False
    #     },
    #     "ans_file_only_label_return_all": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_real_qa_dataset.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_only_label_return_all.json"),
    #         "return_all": True,
    #         "where_clause": False,
    #         "only_label": True
    #     },
    #     "ans_file_replaced_return_all": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_replaced_dataset.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_replaced_return_all.json"),
    #         "return_all": True,
    #         "where_clause": False,
    #         "only_label": False
    #     },
    #     "ans_file_only_action_description": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_replaced_dataset_copy.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_only_action_description.json"),
    #         "return_all": False,
    #         "where_clause": False,
    #         "only_label": False
    #     },
    #     "ans_file_only_action_description_gt": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_replaced_dataset_copy.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_only_action_description_gt.json"),
    #         "return_all": False,
    #         "where_clause": True,
    #         "only_label": False
    #     },
    #     "ans_file_only_action_description_return_all": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/ufc101_dataset/ucf101_replaced_dataset_copy.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_only_action_description_return_all.json"),
    #         "return_all": True,
    #         "where_clause": False,
    #         "only_label": False,
    #     },
    # }
    
    # config_new_only_action_description = {
    #     "ans_file_ucf101_vqa_dataset_with_descriptions": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiment_test/vqa_datasets/ucf101_vqa_dataset_with_descriptions.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_new_only_action_description.json"),
    #         "return_all": False,
    #         "where_clause": False,
    #         "only_label": False
    #     },
    #     "ans_file_ucf101_vqa_dataset_with_descriptions_gt": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiment_test/vqa_datasets/ucf101_vqa_dataset_with_descriptions.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_new_only_action_description_gt.json"),
    #         "return_all": False,
    #         "where_clause": True,
    #         "only_label": False
    #     },
    #     "ans_file_ucf101_vqa_dataset_with_descriptions_return_all": {
    #         "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiment_test/vqa_datasets/ucf101_vqa_dataset_with_descriptions.json",
    #         "ans_path": os.path.join(ans_file_root, "ans_file_new_only_action_description_return_all.json"),
    #         "return_all": True,
    #         "where_clause": False,
    #         "only_label": False
    #     }
    # }
    db_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/ufc101_db"
    db_path_iv2 = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/ufc101_iv2_db"
    db_path_llava = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/ufc101_llava_db"
    ans_file_root = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_26_experiments/results_description_llava"
    os.makedirs(ans_file_root, exist_ok=True)
    config_8_22_multi_descriptions = {
        "ans_file_ucf101_vqa_multiple_descriptions": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_multiple_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_multiple_descriptionsjson"),
            "return_all": False,
            "where_clause": False,
            "only_label": False
        },
        "ans_file_ucf101_vqa_multiple_descriptions_gt": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_multiple_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_multiple_descriptions_gt.json"),
            "return_all": False,
            "where_clause": True,
            "only_label": False
        },
        "ans_file_ucf101_vqa_multiple_descriptions_return_all": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_multiple_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_multiple_descriptions_return_all.json"),
            "return_all": True,
            "where_clause": False,
            "only_label": False
        }
    }
    config_8_22_ratio = {
        "ans_file_ucf101_vqa_retrieval_ratio": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_retrieval_ratio.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_retrieval_ratio.json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False
        },
        "ans_file_ucf101_vqa_retrieval_ratio_gt": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_retrieval_ratio.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_retrieval_ratio_gt.json"),
            "return_all": False,
            "where_clause": True,
            "only_label": False
        },
        "ans_file_ucf101_vqa_retrieval_ratio_return_all": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_retrieval_ratio.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_retrieval_ratio_return_all.json"),
            "return_all": True,
            "where_clause": False,
            "only_label": False
        }
    }
    config_8_22_iv2_ratio = {
        "ans_file_ucf101_vqa_retrieval_ratio_iv2": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_retrieval_ratio.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_retrieval_ratio_iv2.json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False
        },
        "ans_file_ucf101_vqa_retrieval_ratio_gt_iv2": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_retrieval_ratio.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_retrieval_ratio_gt_iv2.json"),
            "return_all": False,
            "where_clause": True,
            "only_label": False
        },
        "ans_file_ucf101_vqa_retrieval_ratio_return_all_iv2": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_retrieval_ratio.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_ucf101_vqa_retrieval_ratio_return_all_iv2.json"),
            "return_all": True,
            "where_clause": False,
            "only_label": False
        }
    }

    config_8_22_ev5_description_cosine_max_mean = {
        "ans_file_8_22_ev5_description_cosine_max_mean": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_dataset_with_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_22_ev5_description_cosine_max_mean.json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False
        },
        "ans_file_8_22_ev5_description_cosine_max_mean_gt": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_dataset_with_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_22_ev5_description_cosine_max_mean_gt.json"),
            "return_all": False,
            "where_clause": True,
            "only_label": False
        },
        "ans_file_8_22_ev5_description_cosine_max_mean_return_all": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_dataset_with_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_22_ev5_description_cosine_max_mean_return_all.json"),
            "return_all": True,
            "where_clause": False,
            "only_label": False
        }
    }

    config_8_22_ev5_description_cosine_mean = {
        "ans_file_8_22_ev5_description_cosine_mean": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_dataset_with_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_22_ev5_description_cosine_mean.json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False
        },
        "ans_file_8_22_ev5_description_cosine_mean_gt": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_dataset_with_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_22_ev5_description_cosine_mean_gt.json"),
            "return_all": False,
            "where_clause": True,
            "only_label": False
        },
        "ans_file_8_22_ev5_description_cosine_mean_return_all": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_dataset_with_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_22_ev5_description_cosine_mean_return_all.json"),
            "return_all": True,
            "where_clause": False,
            "only_label": False
        }
    }
    config_8_23_ev5_description_cosine_mean = {
        "ans_file_8_23_ev5_description_cosine_max_mean": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/8_22_results/ucf101_qa_datasets/ucf101_vqa_dataset_with_descriptions.json",
            "ans_path": os.path.join(ans_file_root, "test.json"),
            "return_all": False,
            "where_clause": False
        }
    }

    # cosine max min
    config_8_25_ev5_description_cosine_mean = {
        "ans_file_8_25_ev5_description_cosine_mean": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_25_experiments/datasets/ucf101_vqa_only_description.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_25_ev5_description_cosine_mean.json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False
        },
        "ans_file_8_25_ev5_description_cosine_mean_gt": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_25_experiments/datasets/ucf101_vqa_only_description.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_25_ev5_description_cosine_mean_gt.json"),
            "return_all": False,
            "where_clause": True,
            "only_label": False
        },
        "ans_file_8_25_ev5_description_cosine_mean_return_all": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_25_experiments/datasets/ucf101_vqa_only_description.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_25_ev5_description_cosine_mean_return_all.json"),
            "return_all": True,
            "where_clause": False,
            "only_label": False
        }
    }

    config_8_26_llava_description_cosine_max_mean = {
        "ans_file_8_26_llava_description_cosine_max_mean": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_25_experiments/datasets/ucf101_vqa_only_description.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_26_llava_description_cosine_max_mean.json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False
        },
        "ans_file_8_26_llava_description_cosine_max_mean_gt": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_25_experiments/datasets/ucf101_vqa_only_description.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_26_llava_description_cosine_max_mean_gt.json"),
            "return_all": False,
            "where_clause": True,
            "only_label": False
        },
        "ans_file_8_26_llava_description_cosine_max_mean_return_all": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_25_experiments/datasets/ucf101_vqa_only_description.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_8_26_llava_description_cosine_max_mean_return_all.json"),
            "return_all": True,
            "where_clause": False,
            "only_label": False
        }
    }


    p= "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_22_results/ucf101_qa_datasets/ucf101_vqa_dataset_with_descriptions.json"
    # iv2_encoder = InternVideo2Encoder()
    llava_encoder = LLaVAQwenEncoder()
    similarity_type = "cosine_max_mean"
    run_test_configurations(config_8_26_llava_description_cosine_max_mean, db_path_llava, llava_encoder, similarity_type)
    
    