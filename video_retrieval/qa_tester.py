import json
import os
from typing import Dict, List, Any, Optional, Callable
# from video_retrieval.encoders.intern_video2_encoder import InternVideo2Encoder
# from video_retrieval.encoders.e5v_encoder import E5VVideoEncoder
from video_retrieval.encoders.llava_qwen_encoder import LLaVAQwenEncoder
# from video_retrieval.encoders.vlm2vec_encoder import VLM2VecEncoder
from tqdm import tqdm
from PIL import Image
import copy
from video_retrieval.video_indexer import LanceDBVideoIndex, LanceDBVideoRetriever
from video_retrieval.stanford40action.utils import parse_phrases

import warnings
warnings.filterwarnings('ignore')

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
    
    def get_10descriptions_test(self):
        action_to_query = {}
        actions = set()
        for qa_pair in self.qa_pairs:
            if qa_pair["action"] in actions:
                action_to_query[qa_pair["action"]]["question"].append(qa_pair["question"])
            else:
                actions.add(qa_pair["action"])
                action_to_query[qa_pair["action"]] = {key: value for key, value in qa_pair.items()}
                action_to_query[qa_pair["action"]]["question"] = [qa_pair["question"]]
               
        new_qa_pairs = list(action_to_query.values())
        return new_qa_pairs
    
    def get_multi_phrase_vqa_pairs(self):
        for qa_pair in self.qa_pairs:
            response = qa_pair["question"]
            phrases = parse_phrases(response)
            qa_pair["question"] = phrases
        return self.qa_pairs
    
    def get_10descriptions_s40a(self):
        img_id_to_query = {}
        img_ids = set()
        for qa_pair in self.qa_pairs:
            ans = qa_pair["answer"]
            if ans in img_ids:
                img_id_to_query[ans]["question"].append(qa_pair["question"])
            else:
                img_ids.add(ans)
                img_id_to_query[ans] = {key: value for key, value in qa_pair.items()}
                img_id_to_query[ans]["question"] = [qa_pair["question"]]
        
        new_qa_pairs = list(img_id_to_query.values())
        return new_qa_pairs



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
    def __init__(self, db_path=None, encoder=None, similarity_type=None, table_name="video_embeddings"):
        self.db_path = db_path
        self.encoder = encoder
        self.similarity_type = similarity_type if similarity_type else "cosine_max_mean"
        self.retriever = LanceDBVideoRetriever(encoder=self.encoder, db_path=self.db_path, table_name=table_name)

    def load_encoder(self):
        pass

    def set_patch_weights(self, patch_weights):
        self.retriever.set_patch_weights(patch_weights)

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
        only_label = False,
    ):  
        
        qa_pairs = getattr(dataset, "qa_pairs", getattr(dataset, 'vqa_pairs', None))
        query_to_idx = {}
        qs  = []
        gt_action = []
        topk = 0
        topk_lst = []
        for idx, qa_pair in enumerate(qa_pairs):
            query_to_idx[qa_pair["question"]] = idx
            qs.append(qa_pair["question"])
            gt_action_single = qa_pair["action"]
            gt_action.append(f"action_category = '{gt_action_single}")
            # topk = max(topk, qa_pair["video_count"])
            topk_lst.append(qa_pair["video_count"])

        batch_size = 55
        total_batch = ((len(qs) -1) // batch_size) + 1
        ans_paths = []
        for batch_idx in tqdm(range(total_batch)):
            start = batch_idx * batch_size
            end = min(start + batch_size - 1, len(qa_pairs))
            ans_path_batch = ans_path.split(".")[0] + f"_{start}_{end}.json"
            ans_paths.append(ans_path_batch)
            qs_batch = qs[start:end+1]
            topk_lst_batch = topk_lst[start:end+1]
            if where_clause:
                results_multi_queries = self.retriever.search(
                    queries=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    similarity_type=self.similarity_type,
                    where_clause=None # gt_action
                )
            else:
                results_multi_queries = self.retriever.search(
                    queries=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    similarity_type=self.similarity_type
                )
                
            results_multi_queries = self.retriever.formated_results(results_multi_queries)    
            
            ans_file = []
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
                    "gt_num": gt_num,
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
            
            with open(ans_path_batch, "w", encoding='utf-8') as f:
                json.dump(ans_file, f, indent=2, ensure_ascii=False)
            
            print(f"Results dumped to: {ans_path_batch}")
            print(f"Total queries: {len(ans_file)}")

        if return_all:
            for ans_path in ans_paths:
                get_gt(ans_path)

        return ans_paths

    def test_dataset_batch_query_s40a(
        self, 
        dataset: UFC101QADataset, 
        ans_path = " ", 
        return_all = False,
        where_clause = False,
        only_label = False,
    ):  
        
        qa_pairs = getattr(dataset, "qa_pairs", getattr(dataset, 'vqa_pairs', None))
        query_to_idx = {}
        qs  = []
        topk_lst = []
        for idx, qa_pair in enumerate(qa_pairs):
            query_to_idx[qa_pair["question"]] = idx
            qs.append(qa_pair["question"])
            topk_lst.append(qa_pair["image_count"])
            # topk_lst.append(20)

        batch_size = 50
        total_batch = (len(qs) -1) // batch_size + 1
        ans_paths = []
        for batch_idx in tqdm(range(total_batch)):
            start = batch_idx * batch_size
            end = min(start + batch_size - 1, len(qa_pairs))
            ans_path_batch = ans_path.split(".")[0] + f"_{start}_{end}.json"
            ans_paths.append(ans_path_batch)
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
            
            ans_file = []
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
                gt_num = qa_pair["image_count"]
                for i, result in enumerate(results):
                    retrieved_video_ids.append(result["metadata"]["video_name"])
                    item = {
                        "image_id": result["metadata"]["video_name"],
                        "similarity": result["similarity"],
                        "rank": result["rank"],
                        "search_time": result["metadata"]["search_time"],
                        "action_category": result["metadata"]["video_name"].rsplit("_", 1)[0].strip(),
                        "image_index": result["metadata"]["video_index"],
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
                    "gt_image_ids": qa_pair["image_ids"],
                    "gt_num": qa_pair["image_count"],
                    "gt_action": qa_pair["action"],
                    "retrieved_image_ids": retrieved_video_ids,
                    "retrieved_num": len(results),
                    "items": items,
                    "avg_similarity": sum(similarities) / len(retrieved_video_ids) if retrieved_video_ids else 0,
                    "max_similarity": max(similarities) if similarities else 0,
                    "min_similarity": min(similarities) if similarities else 0,
                    "hit": hit
                }
                ans_file.append(ans)
            
            with open(ans_path_batch, "w", encoding='utf-8') as f:
                json.dump(ans_file, f, indent=2, ensure_ascii=False)
        
            print(f"Results dumped to: {ans_path_batch}")
            print(f"Total queries: {len(ans_file)}")
        
        if return_all:
            for ans_path in ans_paths:
                # get_gt_image(ans_path)
                get_gt_image_by_query(ans_path)

        return ans_paths
    
    def test_dataset_batch_query_merge_desc_s40a(
        self, 
        dataset: UFC101QADataset, 
        ans_path = " ", 
        return_all = False,
        where_clause = False,
        only_label = False,
    ):  
        
        qa_pairs = dataset.get_10descriptions_test()
        # qa_pairs = dataset.get_multi_phrase_vqa_pairs()
        # qa_pairs = dataset.get_10descriptions_s40a()
        qs  = []
        topk_lst = []
        for idx, qa_pair in enumerate(qa_pairs):
            qs.append(qa_pair["question"])
            topk_lst.append(qa_pair["image_count"])
            # topk_lst.append(20)

        batch_size = 50
        total_batch = ((len(qs) -1) // batch_size) + 1
        ans_paths = []
        for batch_idx in tqdm(range(total_batch)):
            start = batch_idx * batch_size
            end = min(start + batch_size - 1, len(qa_pairs))
            ans_path_batch = ans_path.split(".")[0] + f"_{start}_{end}.json"
            ans_paths.append(ans_path_batch)
            qs_batch = qs[start:end+1]
            topk_lst_batch = topk_lst[start:end+1]
            if self.similarity_type:
                results_multi_queries = self.retriever.search_10descriptions_test(
                    queries=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    similarity_type=self.similarity_type
                )
            else:
                results_multi_queries = self.retriever.search_10descriptions_test(
                    queries=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all
                )
            results_multi_queries = self.retriever.formated_results(results_multi_queries)    
            
            ans_file = []
            # self.retriever.dump_results(results_multi_queries,ans_path_batch)
            for query_idx, query_result in results_multi_queries.items():
                hit = 0
                similarities = []
                retrieved_video_ids = []
                items = []
                results = query_result["results"]
                question = query_result["query_text"]
                question_idx = query_idx + start
                qa_pair = qa_pairs[question_idx]
                gt_num = qa_pair["image_count"]
                for i, result in enumerate(results):
                    retrieved_video_ids.append(result["metadata"]["video_name"])
                    item = {
                        "image_id": result["metadata"]["video_name"],
                        "similarity": result["similarity"],
                        "rank": result["rank"],
                        "search_time": result["metadata"]["search_time"],
                        "action_category": result["metadata"]["video_name"].rsplit("_", 1)[0].strip(),
                        "image_index": result["metadata"]["video_index"],
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
                    "gt_image_ids": qa_pair["image_ids"],
                    "gt_num": qa_pair["image_count"],
                    "gt_action": qa_pair["action"],
                    "retrieved_image_ids": retrieved_video_ids,
                    "retrieved_num": len(results),
                    "items": items,
                    "avg_similarity": sum(similarities) / len(retrieved_video_ids) if retrieved_video_ids else 0,
                    "max_similarity": max(similarities) if similarities else 0,
                    "min_similarity": min(similarities) if similarities else 0,
                    "hit": hit
                }
                ans_file.append(ans)
            
            with open(ans_path_batch, "w", encoding='utf-8') as f:
                json.dump(ans_file, f, indent=2, ensure_ascii=False)
        
            print(f"Results dumped to: {ans_path_batch}")
            print(f"Total queries: {len(ans_file)}")
        
        if return_all:
            for ans_path in ans_paths:
                get_gt_image(ans_path)

        return ans_paths

    def test_dataset_batch_query_merge_description(
        self, 
        dataset: UFC101QADataset, 
        ans_path = " ", 
        return_all = False,
        where_clause = False,
        only_label = False,
    ):  
        
        qa_pairs = dataset.get_10descriptions_test()
        label_to_idx = {}
        qs  = []
        topk = 0
        topk_lst = []
        for idx, qa_pair in enumerate(qa_pairs):
            label_to_idx[qa_pair["action"]] = idx
            qs.append(qa_pair["question"])
            topk = max(topk, qa_pair["video_count"])
            topk_lst.append(qa_pair["video_count"])

        batch_size = 55
        total_batch = (len(qs) -1) // batch_size + 1
        ans_paths = []
        for batch_idx in tqdm(range(total_batch)):
            start = batch_idx * batch_size
            end = min(start + batch_size - 1, len(qa_pairs))
            ans_path_batch = ans_path.split(".")[0] + f"_{start}_{end}.json"
            ans_paths.append(ans_path_batch)
            qs_batch = qs[start:end+1]
            topk_lst_batch = topk_lst[start:end+1]
            if self.similarity_type:
                results_multi_queries = self.retriever.search_10descriptions_test(
                    queries=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    similarity_type=self.similarity_type
                )
            else:
                results_multi_queries = self.retriever.search_10descriptions_test(
                    queries=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all
                )
            results_multi_queries = self.retriever.formated_results(results_multi_queries)    
            
            ans_file = []
            # self.retriever.dump_results(results_multi_queries,ans_path_batch)
            for query_idx, query_result in results_multi_queries.items():
                hit = 0
                similarities = []
                retrieved_video_ids = []
                items = []
                results = query_result["results"]
                question = query_result["query_text"]
                question_idx = query_idx + start
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
            
            with open(ans_path_batch, "w", encoding='utf-8') as f:
                json.dump(ans_file, f, indent=2, ensure_ascii=False)
        
            print(f"Results dumped to: {ans_path_batch}")
            print(f"Total queries: {len(ans_file)}")

        if return_all:
            for ans_path in ans_paths:
                get_gt(ans_path)

        return ans_paths

    def test_dataset_batch_image_query_desc_s40a(
        self, 
        dataset: UFC101QADataset, 
        ans_path = " ", 
        return_all = False,
        where_clause = False,
        only_label = False,
        images_dir = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train",
        return_gt = True
    ):  
        
        qa_pairs = dataset.qa_pairs
        qs, topk_lst, return_gt_lst  = [], [], []

        for idx, qa_pair in enumerate(qa_pairs):
            q_image_path = qa_pair["action"] + "/" + qa_pair["question"]
            q_image_path = os.path.join(images_dir, q_image_path)
            q_image = Image.open(q_image_path).convert("RGB")
            qs.append(q_image)
            topk_lst.append(100)
            gt_lst = copy.deepcopy(qa_pair["image_ids"])
            return_gt_lst.append(gt_lst)

        if not return_gt:
            return_gt_lst = []

        batch_size = 50
        total_batch = ((len(qs) -1) // batch_size) + 1
        ans_paths = []
        for batch_idx in tqdm(range(total_batch)):
            start = batch_idx * batch_size
            end = min(start + batch_size - 1, len(qa_pairs))
            ans_path_batch = ans_path.split(".")[0] + f"_{start}_{end}.json"
            ans_paths.append(ans_path_batch)
            qs_batch = qs[start:end+1]
            topk_lst_batch = topk_lst[start:end+1]
            if self.similarity_type:
                results_multi_queries = self.retriever.search_image(
                    images=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    similarity_type=self.similarity_type,
                    return_gt = return_gt_lst
                )
            else:
                results_multi_queries = self.retriever.search_image(
                    images=qs_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    return_gt = return_gt_lst
                )
            
            ans_file = []
            
            for query_idx, query_result in results_multi_queries.items():
                hit = 0
                similarities = []
                retrieved_ids = []
                items = []
                question_idx = query_idx + start
                qa_pair = qa_pairs[question_idx]
                gt_num = qa_pair["image_count"]
                for i, (image_path, similarity, rank) in enumerate(query_result):
                    image_id = os.path.basename(image_path)
                    retrieved_ids.append(image_id)
                    item = {
                        "image_id": image_id,
                        "similarity": similarity,
                        "rank": rank,
                        "action_category": image_id.rsplit("_", 1)[0].strip(),
                    }
                    items.append(item)
                    similarities.append(similarity)

                    if i < gt_num and (image_id in qa_pair["image_ids"]):
                        hit += 1

                ans = {
                    "question_idx": question_idx,
                    "question": qa_pair["question"],
                    "cluster_label": qa_pair["cluster_label"],
                    "cluster_desc": qa_pair["cluster_desc"],
                    "gt_image_ids": qa_pair["image_ids"],
                    "gt_num": gt_num,
                    "gt_action": qa_pair["action"],
                    "retrieved_image_ids": retrieved_ids,
                    "retrieved_num": len(retrieved_ids),
                    "items": items,
                    "avg_similarity": sum(similarities[:gt_num]) / gt_num if similarities else 0,
                    "max_similarity": max(similarities[:gt_num]) if similarities else 0,
                    "min_similarity": min(similarities[:gt_num]) if similarities else 0,
                    "hit": hit
                }
                ans_file.append(ans)
            
            with open(ans_path_batch, "w", encoding='utf-8') as f:
                json.dump(ans_file, f, indent=2, ensure_ascii=False)
        
            print(f"Results dumped to: {ans_path_batch}")
            print(f"Total queries: {len(ans_file)}")
        
        return ans_paths
    
    def test_dataset_batch_hybrid_query_s40a(
        self, 
        dataset: UFC101QADataset, 
        ans_path = " ", 
        return_all = False,
        where_clause = False,
        only_label = False,
        return_gt = True,
        query_token_weight = [0.1, 0.9]
    ):  
        
        qa_pairs = dataset.qa_pairs
        qs_texts, qs_images, topk_lst, return_gt_lst  = [], [], [], []

        for qa_pair in qa_pairs:
            qs_texts.append(qa_pair["question_text"])
            qs_images.append(qa_pair["question_image"])
            topk_lst.append(20)
            if return_gt:
                gt_lst = copy.deepcopy(qa_pair["image_id"])
                return_gt_lst.append(gt_lst)

        if not return_gt:
            return_gt_lst = []
        
        # weighted query token sum
        # query_token_weight = [0.1, 0.9]
        if self.similarity_type and self.similarity_type == "colbert_maxsim_weighted_token_sum":
            self.retriever.index.set_query_token_weight(query_token_weight)

        batch_size = 300
        total_batch = ((len(qs_texts) -1) // batch_size) + 1
        ans_paths = []
        for batch_idx in tqdm(range(total_batch)):
            start = batch_idx * batch_size
            end = min(start + batch_size - 1, len(qa_pairs))
            ans_path_batch = ans_path.split(".")[0] + f"_{start}_{end}.json"
            ans_paths.append(ans_path_batch)
            qs_texts_batch, qs_images_batch = qs_texts[start:end+1], qs_images[start:end+1]
            topk_lst_batch = topk_lst[start:end+1]
            if self.similarity_type:
                results_multi_queries = self.retriever.search_hybrid_weighted_sum(
                    texts=qs_texts_batch,
                    image_paths=qs_images_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    similarity_type=self.similarity_type,
                    return_gt = return_gt_lst
                )
            else:
                # TODO:
                results_multi_queries = self.retriever.search_hybrid_weighted_sum(
                    texts=qs_texts_batch,
                    image_paths=qs_images_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    return_gt = return_gt_lst
                )
            
            ans_file = []
            
            for query_idx, query_result in results_multi_queries.items():
                hit = 0
                similarities = []
                retrieved_ids = []
                items = []
                question_idx = query_idx + start
                qa_pair = qa_pairs[question_idx]
                gt_num = qa_pair["image_count"]
                for i, (image_path, similarity, rank) in enumerate(query_result):
                    image_id = os.path.basename(image_path)
                    retrieved_ids.append(image_id)
                    item = {
                        "image_id": image_id,
                        "similarity": similarity,
                        "rank": rank,
                        "action_category": image_id.rsplit("_", 1)[0].strip(),
                    }
                    items.append(item)
                    similarities.append(similarity)

                    if i < gt_num and (image_id in qa_pair["image_id"]):
                        hit += 1

                ans = {
                    "question_idx": question_idx,
                    "question_text": qa_pair["question_text"],
                    "question_image": qa_pair["question_image"],
                    "gt_image_id": qa_pair["image_id"][0],
                    "gt_num": gt_num,
                    "gt_action": qa_pair["image_id"][0].rsplit("_", 1)[0].strip(),
                    "retrieved_image_ids": retrieved_ids,
                    "retrieved_num": len(retrieved_ids),
                    "items": items,
                    "avg_similarity": sum(similarities[:gt_num]) / gt_num if similarities else 0,
                    "max_similarity": max(similarities[:gt_num]) if similarities else 0,
                    "min_similarity": min(similarities[:gt_num]) if similarities else 0,
                    "hit": hit
                }
                ans_file.append(ans)
            
            with open(ans_path_batch, "w", encoding='utf-8') as f:
                json.dump(ans_file, f, indent=2, ensure_ascii=False)
        
            print(f"Results dumped to: {ans_path_batch}")
            print(f"Total queries: {len(ans_file)}")
        
        return ans_paths
    
    def test_dataset_batch_2images_query_s40a(
        self, 
        dataset: UFC101QADataset, 
        ans_path = " ", 
        return_all = False,
        where_clause = False,
        only_label = False,
        return_gt = True,
        query_token_weight = [0.1, 0.9]
    ):  
        
        qa_pairs = dataset.qa_pairs
        qs_texts, qs_images, topk_lst, return_gt_lst  = [], [], [], []

        for qa_pair in qa_pairs:
            qs_texts.append(qa_pair["question_desc2image"])
            qs_images.append(qa_pair["question_image"])
            topk_lst.append(20)
            gt_lst = copy.deepcopy(qa_pair["image_id"])
            return_gt_lst.append(gt_lst)

        if not return_gt:
            return_gt_lst = []
        
        # weighted query token sum
        # query_token_weight = [0.1, 0.9]
        if self.similarity_type and self.similarity_type == "colbert_maxsim_weighted_token_sum":
            self.retriever.index.set_query_token_weight(query_token_weight)

        batch_size = 300
        total_batch = ((len(qs_texts) -1) // batch_size) + 1
        ans_paths = []
        for batch_idx in tqdm(range(total_batch)):
            start = batch_idx * batch_size
            end = min(start + batch_size - 1, len(qa_pairs))
            ans_path_batch = ans_path.split(".")[0] + f"_{start}_{end}.json"
            ans_paths.append(ans_path_batch)
            qs_texts_batch, qs_images_batch = qs_texts[start:end+1], qs_images[start:end+1]
            topk_lst_batch = topk_lst[start:end+1]
            if self.similarity_type:
                results_multi_queries = self.retriever.search_2images_weighted_sum(
                    image1_paths=qs_images_batch,
                    image2_paths=qs_texts_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    similarity_type=self.similarity_type,
                    return_gt = return_gt_lst
                )
            else:
                # TODO:
                results_multi_queries = self.retriever.search_2images_weighted_sum(
                    image1_paths=qs_images_batch,
                    image2_paths=qs_texts_batch,
                    top_k=topk_lst_batch,
                    return_all = return_all,
                    return_gt = return_gt_lst
                )
            
            ans_file = []
            
            for query_idx, query_result in results_multi_queries.items():
                hit = 0
                similarities = []
                retrieved_ids = []
                items = []
                question_idx = query_idx + start
                qa_pair = qa_pairs[question_idx]
                gt_num = qa_pair["image_count"]
                for i, (image_path, similarity, rank) in enumerate(query_result):
                    image_id = os.path.basename(image_path)
                    retrieved_ids.append(image_id)
                    item = {
                        "image_id": image_id,
                        "similarity": similarity,
                        "rank": rank,
                        "action_category": image_id.rsplit("_", 1)[0].strip(),
                    }
                    items.append(item)
                    similarities.append(similarity)

                    if i < gt_num and (image_id in qa_pair["image_id"]):
                        hit += 1

                ans = {
                    "question_idx": question_idx,
                    "question_text": qa_pair["question_desc"],
                    "question_desc_image": qa_pair["question_desc2image"],
                    "question_image": qa_pair["question_image"],
                    "gt_image_id": qa_pair["image_id"][0],
                    "gt_num": gt_num,
                    "gt_action": qa_pair["image_id"][0].rsplit("_", 1)[0].strip(),
                    "retrieved_image_ids": retrieved_ids,
                    "retrieved_num": len(retrieved_ids),
                    "items": items,
                    "avg_similarity": sum(similarities[:gt_num]) / gt_num if similarities else 0,
                    "max_similarity": max(similarities[:gt_num]) if similarities else 0,
                    "min_similarity": min(similarities[:gt_num]) if similarities else 0,
                    "hit": hit
                }
                ans_file.append(ans)
            
            with open(ans_path_batch, "w", encoding='utf-8') as f:
                json.dump(ans_file, f, indent=2, ensure_ascii=False)
        
            print(f"Results dumped to: {ans_path_batch}")
            print(f"Total queries: {len(ans_file)}")
        
        return ans_paths

    @staticmethod
    def get_score(ans_paths = None):
        if isinstance(ans_paths, str):
            ans_paths = [ans_paths]
        results_paths = []
        for ans_file in ans_paths:

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
                    "gt_action": ans["gt_action"],
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
            results_paths.append(results_path)
        return results_paths
    
    @staticmethod
    def get_score_image(ans_paths = None):
        if isinstance(ans_paths, str):
            ans_paths = [ans_paths]
        results_paths = []
        for ans_file in ans_paths:

            with open(ans_file, "r") as f:
                answers = json.load(f)
            
            results = []

            for ans in answers:
                gt_num = ans["gt_num"]
                gt_image_ids = set(ans["gt_image_ids"])
                retrieved_image_ids = ans["retrieved_image_ids"]
                retrieved_num = ans["retrieved_num"]
                hit_by_id = 0
                
                for i, v in enumerate(retrieved_image_ids):
                    if i < gt_num and (v in gt_image_ids):
                        hit_by_id += 1
                        gt_image_ids.remove(v)
                
                clean_result = {
                    "Question": ans["question"],
                    "gt_action": ans["gt_action"],
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
            results_paths.append(results_path)
        return results_paths


def run_test_configurations(config, db_path, encoder, similarity_type, table_name="video_embeddings"):
    tester = DatasetTester(db_path, encoder, similarity_type, table_name)
    # optional
    # patch_weights = [0.8] * 9 + [2.8]
    # assert len(patch_weights) == 10
    # tester.set_patch_weights(patch_weights)

    for key, item in config.items():
        print(f"\n{'='*50}")
        print(f"Testing: {key}")
        print(f"{'='*50}")
        
        dataset_path = item["dataset_path"]
        dataset = UFC101QADataset(dataset_path)
        # ans_paths = tester.test_dataset_batch_query(
        #     dataset=dataset,
        #     ans_path=item["ans_path"],
        #     return_all=item.get("return_all", False),
        #     where_clause=item.get("where_clause", False),
        #     only_label=item.get("only_label", False)
        # )
        # ans_paths = tester.test_dataset_batch_query_merge_description(
        #     dataset=dataset,
        #     ans_path=item["ans_path"],
        #     return_all=item.get("return_all", False),
        #     where_clause=item.get("where_clause", False),
        #     only_label=item.get("only_label", False)
        # )
        # ans_paths = tester.test_dataset_batch_query_s40a(
        #     dataset=dataset,
        #     ans_path=item["ans_path"],
        #     return_all=item.get("return_all", False),
        #     where_clause=item.get("where_clause", False),
        #     only_label=item.get("only_label", False)
        # )
        # ans_paths = tester.test_dataset_batch_query_merge_desc_s40a(
        #     dataset=dataset,
        #     ans_path=item["ans_path"],
        #     return_all=item.get("return_all", False),
        #     where_clause=item.get("where_clause", False),
        #     only_label=item.get("only_label", False)
        # )
        # ans_paths = tester.test_dataset_batch_image_query_desc_s40a(
        #     dataset=dataset,
        #     ans_path=item["ans_path"],
        #     return_all=item.get("return_all", False),
        #     where_clause=item.get("where_clause", False),
        #     only_label=item.get("only_label", False)
        # )
        ans_paths = tester.test_dataset_batch_hybrid_query_s40a(
            dataset=dataset, 
            ans_path=item["ans_path"],
            return_all=item.get("return_all", False),
            where_clause=item.get("where_clause", False),
            only_label=item.get("only_label", False),
            return_gt=item.get("return_gt", True),
            query_token_weight=item.get("query_token_weight", [0.1, 0.9])
        )
        # ans_paths = tester.test_dataset_batch_2images_query_s40a(
        #     dataset=dataset, 
        #     ans_path=item["ans_path"],
        #     return_all=item.get("return_all", False),
        #     where_clause=item.get("where_clause", False),
        #     only_label=item.get("only_label", False),
        #     query_token_weight=item.get("query_token_weight", [0.1, 0.9])
        # )
        # results_path = tester.get_score_image(ans_paths)
        # results_path = tester.get_score(ans_paths)
        # print(f"Results stored at: {results_path}")

def get_gt(return_all_file):
    with open(return_all_file, "r") as f:
        data = json.load(f)
    gt_data = []
    for query_result in data:
        gt_action = query_result["gt_action"]
        gt_items = []
        retrieved_video_ids = []
        similarities = []
        for item in query_result["items"]:
            if item.get("action_actegory", None) == gt_action or item.get("action_category", None) == gt_action:
                gt_items.append(item)
                retrieved_video_ids.append(item["video_id"])
                similarities.append(item["similarity"])

        to_be_replaced = {"retrieved_video_ids", "retrieved_num", "items", "avg_similarity", "max_similarity", "min_similarity", "hit"}
        new_query_result = {}
        for key, value in query_result.items():
            if key not in to_be_replaced:
                new_query_result[key] = value
        updated = {
            "retrieved_video_ids": retrieved_video_ids,
            "retrieved_num": len(retrieved_video_ids),
            "items": gt_items,
            "avg_similarity": sum(similarities) / len(retrieved_video_ids) if retrieved_video_ids else 0,
            "max_similarity": max(similarities) if retrieved_video_ids else 0,
            "min_similarity": min(similarities) if retrieved_video_ids else 0,
            "hit": 1
        }
        new_query_result = new_query_result | updated
        # new_query_result = {
        #     "question_idx": query_result["question_idx"],
        #     "question": query_result["question"],
        #     "gt_video_ids": query_result["gt_video_ids"],
        #     "gt_num": query_result["gt_num"],
        #     "gt_action": query_result["gt_action"],
        #     "retrieved_video_ids": retrieved_video_ids,
        #     "retrieved_num": len(retrieved_video_ids),
        #     "items": gt_items,
        #     "avg_similarity": sum(similarities) / len(retrieved_video_ids) if retrieved_video_ids else 0,
        #     "max_similarity": max(similarities) if retrieved_video_ids else 0,
        #     "min_similarity": min(similarities) if retrieved_video_ids else 0,
        #     "hit": 1
        # }
        gt_data.append(new_query_result)

    file_path = return_all_file.replace("return_all", "gt")
    with open(file_path, "w") as f:
        json.dump(gt_data, f, indent=2)
    print(f"dump {file_path}")
    gt_result_path = DatasetTester.get_score(file_path)

def get_gt_image_by_query(return_all_file):
    with open(return_all_file, "r") as f:
        data = json.load(f)
    gt_data = []
    for query_result in data:
        gt_image = query_result["gt_image_ids"][0]
        gt_items = []
        retrieved_image_ids = []
        similarities = []
        for item in query_result["items"]:
            if item["image_id"] == gt_image:
                gt_items.append(item)
                retrieved_image_ids.append(item["image_id"])
                similarities.append(item["similarity"])

        to_be_replaced = {"retrieved_image_ids", "retrieved_num", "items", "avg_similarity", "max_similarity", "min_similarity", "hit"}
        new_query_result = {}
        for key, value in query_result.items():
            if key not in to_be_replaced:
                new_query_result[key] = value
        updated = {
            "retrieved_image_ids": retrieved_image_ids,
            "retrieved_num": len(retrieved_image_ids),
            "items": gt_items,
            "avg_similarity": sum(similarities) / len(retrieved_image_ids) if retrieved_image_ids else 0,
            "max_similarity": max(similarities) if retrieved_image_ids else 0,
            "min_similarity": min(similarities) if retrieved_image_ids else 0,
            "hit": 1
        }
        new_query_result = new_query_result | updated
        gt_data.append(new_query_result)

    file_path = return_all_file.replace("return_all", "gt")
    with open(file_path, "w") as f:
        json.dump(gt_data, f, indent=2)
    
    gt_result_path = DatasetTester.get_score_image(file_path)

def get_gt_image(return_all_file):
    with open(return_all_file, "r") as f:
        data = json.load(f)
    gt_data = []
    for query_result in data:
        gt_action = query_result["gt_action"]
        gt_items = []
        retrieved_image_ids = []
        similarities = []
        for item in query_result["items"]:
            if item.get("action_actegory", None) == gt_action or item.get("action_category", None) == gt_action:
                gt_items.append(item)
                retrieved_image_ids.append(item["image_id"])
                similarities.append(item["similarity"])

        to_be_replaced = {"retrieved_image_ids", "retrieved_num", "items", "avg_similarity", "max_similarity", "min_similarity", "hit"}
        new_query_result = {}
        for key, value in query_result.items():
            if key not in to_be_replaced:
                new_query_result[key] = value
        updated = {
            "retrieved_image_ids": retrieved_image_ids,
            "retrieved_num": len(retrieved_image_ids),
            "items": gt_items,
            "avg_similarity": sum(similarities) / len(retrieved_image_ids) if retrieved_image_ids else 0,
            "max_similarity": max(similarities) if retrieved_image_ids else 0,
            "min_similarity": min(similarities) if retrieved_image_ids else 0,
            "hit": 1
        }
        new_query_result = new_query_result | updated
        gt_data.append(new_query_result)

    file_path = return_all_file.replace("return_all", "gt")
    with open(file_path, "w") as f:
        json.dump(gt_data, f, indent=2)
    
    gt_result_path = DatasetTester.get_score_image(file_path)


if __name__ == "__main__":
    
    db_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/ufc101_db"
    db_path_iv2 = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/ufc101_iv2_db"
    db_path_llava = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/ufc101_llava_db"
    db_path_s40a_e5v = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/s40a_e5v_db"
    db_path_s40a_llava = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/s40a_llava_db"
    db_path_s40a_vlm2vec = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/s40a_vlm2vec_db"
    ans_file_root = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results/10_22_llave_s40a_imageText2image_no_patches_1_9_colbert_weighted_query_token_sum"
    os.makedirs(ans_file_root, exist_ok=True)
    
    config_10_15_llave_s40a_imageText2image_no_patches_9_91_colbert_weighted_query_token_sum = {
        "ans_file_10_15_llave_s40a_imageText2image_no_patches_9_91_colbert_weighted_query_token_sum": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_6_experiments/datasets/ImageText2Image4.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_10_15_llave_s40a_imageText2image_no_patches_9_91_colbert_weighted_query_token_sum.json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False
        }
    }

    config_10_22_llave_s40a_imageText2image_no_patches_1_9_colbert_weighted_query_token_sum = {
        "ans_file_10_22_llave_s40a_imageText2image_no_patches_1_9_colbert_weighted_query_token_sum": {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_6_experiments/datasets/ImageText2Image4.json",
            "ans_path": os.path.join(ans_file_root, "ans_file_10_22_llave_s40a_imageText2image_no_patches_1_9_colbert_weighted_query_token_sum.json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False,
            "query_token_weight": [0.1, 0.9]
        }
    }
    
    root = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/results/results_haolin2"
    os.makedirs(root, exist_ok=True)
    dir_name = "10_22_llave_s40a_imageText2image_vague_no_patches_imageTexthaolin_colbert_weighted_query_token_sum"
    # weights = [[0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0, 1]]
    # weights = [[0.5, 0.5]]
    # weights = [[0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1, 0]]
    weights = [[0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1, 0], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0, 1]]
    dirs = [os.path.join(root, f"{dir_name}_{int(iw*100)}_{int(tw*100)}") for [iw, tw] in weights]
    for dir_ in dirs:
        os.makedirs(dir_, exist_ok=True)
    configs = {}
    for [iw, tw], dir_ in zip(weights, dirs):
        config_name = f"ans_file_{os.path.basename(dir_)}"
        config_item = {
            "dataset_path": "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/datasets/Img2TextHaolin.json",
            "ans_path": os.path.join(dir_,config_name + ".json"),
            "return_all": False,
            "where_clause": False,
            "only_label": False,
            "return_gt": False,
            "query_token_weight": [iw, tw]
        }
        configs[config_name] = config_item
    
    # iv2_encoder = InternVideo2Encoder()
    llava_encoder = LLaVAQwenEncoder()
    # e5v_encoder = E5VVideoEncoder()
    # vlm2vec_encoder = VLM2VecEncoder()
    similarity_type = "colbert_maxsim_weighted_token_sum" # "cosine_max_mean" # "colbert_maxsim_mean"
    # retriever = LanceDBVideoRetriever(encoder=llava_encoder, db_path=db_path_s40a_llava, table_name="image_17patches_embeddings")
    # breakpoint()
    # print("end")
    run_test_configurations(configs, db_path_s40a_llava, llava_encoder, similarity_type, "image_embeddings")
    # dataset = UFC101QADataset("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_27_experiments/dataset/ucf101_vqa_multiple_descriptions_fixed.json")
    # get_gt("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/8_29_experiments/results/ufc101_llava_10desc_colbert_max_mean/ans_file_8_29_llava_10_descriptions_colbert_maxsim_mean_return_all_0_54.json")
    