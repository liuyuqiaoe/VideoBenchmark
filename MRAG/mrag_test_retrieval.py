import json
import time
from datasets import load_dataset
# from MRAG.eval.utils.dataloader import bench_data_loader 
from tqdm import tqdm
from PIL import Image
import io
import base64
from io import BytesIO
import copy
import torch
import os
from pathlib import Path
from typing import Optional, List
from openai import OpenAI
from transformers import AutoProcessor, LlavaForConditionalGeneration
import shortuuid
import shutil
from video_retrieval.encoders.e5v_encoder import E5VVideoEncoder
from video_retrieval import LanceDBVideoRetriever

def bench_data_loader(mode, image_placeholder="<image>", batch_size: int = 100):
    
    mode_lst = ["base", "using_gt_images", "using_retrieved_examples", "using_clip_retriever"]
    assert mode in mode_lst
    
    mrag_bench = load_dataset("uclanlp/MRAG-Bench", split="test")
    
    batch_items = []
    
    for item in tqdm(mrag_bench):
        qs_id = item['id'] 
        qs = item['question']
        ans = item['answer']
        gt_choice = item['answer_choice']
        scenario = item['scenario']
        choices_A = item['A']
        choices_B = item['B']
        choices_C = item['C']
        choices_D = item['D']
        gt_images = item['gt_images']
        gt_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib['bytes'])).convert("RGB") for ib in gt_images]
        
        image = item['image'].convert("RGB") 

        if scenario == 'Incomplete':
            gt_images = [gt_images[0]]        

        if mode == "base":
            prompt = f"Answer with the option's letter from the given choices directly.\n"
            image_files = [image]
        elif mode == "using_gt_images":
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
            image_files = [image] + gt_images
            if scenario == "Incomplete":
                prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
        elif mode == "using_retrieved_examples":
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
            retrieved_images = item["retrieved_images"]
            retrieved_images = [ib.convert("RGB") if isinstance(ib, Image.Image) else Image.open(io.BytesIO(ib["bytes"])) for ib in retrieved_images]
            if scenario == "Incomplete":
                retrieved_images = [retrieved_images[0]]
                prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
            image_files = [image] + retrieved_images
        else:
            prompt = f"You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly.\n"
            image_files = [image]
        
        qs += f"\n Choices:\nA: {choices_A}\nB: {choices_B}\nC: {choices_C}\nD: {choices_D}"
        prompt_question_part = qs
        prompt_instruction_part = prompt
        qs = prompt + qs
        
        processed_item = {
            "id": qs_id, 
            "question": qs, 
            "image_files": image_files, 
            "prompt": qs,
            "answer": ans,
            "gt_choice": gt_choice,
            "scenario": scenario,
            "prompt_question_part": prompt_question_part,
            "prompt_instruction_part": prompt_instruction_part,
            "aspect": item['aspect'],
            "gt_images": gt_images,
            "plain_question": item["question"],
            "choices": [["A", choices_A], ["B", choices_B], ["C", choices_C], ["D", choices_D]]
        }
        
        batch_items.append(processed_item)
        
        if len(batch_items) >= batch_size:
            yield batch_items
            batch_items = []
    
    if batch_items:
        yield batch_items

def dump_images(image, gt_images, item_id, output_dir):
    sub_dir = os.path.join(output_dir, f"question_{item_id}")
    if os.path.exists(sub_dir):
        # print(f"remove existing path {sub_dir}")
        shutil.rmtree(sub_dir)
    os.makedirs(sub_dir)
    input_image_path = os.path.join(sub_dir, "input_image.jpg")
    image.save(input_image_path)
    gt_dir = os.path.join(sub_dir, f"gt_images")
    os.makedirs(gt_dir, exist_ok=True)
    gt_image_paths = []
    for i, gt_image in enumerate(gt_images):
        file_name = f"gt_image_{i}.jpg"
        gt_image_path = os.path.join(gt_dir, file_name)
        gt_image_paths.append(gt_image_path)
        gt_image.save(gt_image_path)
    return input_image_path, gt_image_paths


def test_retrieval_mragbenchmark():
    mode = "using_clip_retriever"

    print("Testing retrieval on MRAG benchmark")
    
    db_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/mrag_e5v_db"
    ans_file = open("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/MRAG/experiments/results/retrival.jsonl", "w")
    encoder = E5VVideoEncoder(max_frames_per_video=1)
    retriever = LanceDBVideoRetriever(encoder=encoder, db_path=db_path)
    
    for item_lst in bench_data_loader(mode):

        q_lst = [item["question"] for item in item_lst]
        
        results = retriever.search_clean(q_lst, top_k=4, similarity_type="cosine_max_mean")
        
        output_dir = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/MRAG/experiments/images_tmp"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for query_idx, query_result in results.items():
            selected_image_paths = [r[0] for r in query_result]
            assert len(selected_image_paths) == 4

            item = item_lst[query_idx]
            input_image_path, gt_image_paths = dump_images(item["image_files"][0], item["gt_images"], item['id'], output_dir)
            ans_file.write(json.dumps({
                "id": item['id'],
                "question": item["question"],
                "prompt": item['prompt'],
                "answer": item["answer"],
                "gt_choice": item["gt_choice"],
                "prompt_question_part": item["prompt_question_part"],
                "prompt_instruction_part": item["prompt_instruction_part"],
                "image": input_image_path,
                "gt_images": gt_image_paths,
                "retrieved_images": selected_image_paths,
                "gt_answer": item['answer'],
                "shortuuid": shortuuid.uuid(),
                "approach": 'cosine_mean_max',
                "scenario": item['scenario'],
                "aspect": item['aspect'],
            }) + "\n")
            ans_file.flush()

def test_retrieval_mragbenchmark_image():
    mode = "using_clip_retriever"

    print("Testing retrieval on MRAG benchmark")

    db_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases/mrag_e5v_db"
    ans_file = open("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/MRAG/experiments/results/retrival_image.jsonl", "w")
    encoder = E5VVideoEncoder(max_frames_per_video=1)
    retriever = LanceDBVideoRetriever(encoder=encoder, db_path=db_path)
    
    for item_lst in bench_data_loader(mode):

        img_lst = [item["image_files"][0] for item in item_lst]
        
        results = retriever.search_image(img_lst, top_k=4, similarity_type="cosine_max_mean")
        
        output_dir = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/MRAG/experiments/images_tmp2"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for query_idx, query_result in results.items():
            selected_image_paths = [r[0] for r in query_result]
            assert len(selected_image_paths) == 4

            item = item_lst[query_idx]
            input_image_path, gt_image_paths = dump_images(item["image_files"][0], item["gt_images"], item['id'], output_dir)
            ans_file.write(json.dumps({
                "id": item['id'],
                "question": item["question"],
                "prompt": item['prompt'],
                "answer": item["answer"],
                "gt_choice": item["gt_choice"],
                "prompt_question_part": item["prompt_question_part"],
                "prompt_instruction_part": item["prompt_instruction_part"],
                "image": input_image_path,
                "gt_images": gt_image_paths,
                "retrieved_images": selected_image_paths,
                "gt_answer": item['answer'],
                "shortuuid": shortuuid.uuid(),
                "approach": 'cosine_mean_max',
                "scenario": item['scenario'],
                "aspect": item['aspect'],
            }) + "\n")
            ans_file.flush()

def test_retrieval_mragbenchmark_base(collection):
    mode = "base"

    print("Testing retrieval on MRAG benchmark (base)")

    ans_file = open(os.path.join(os.getcwd(),"colbert/tests/base.jsonl"), "w")
   
    # gpt_generator = GPTGenerator()
    for item in bench_data_loader(mode):
       
        qs = item["plain_question"]
        choices = item["choices"]
        
        output_dir = os.path.join(os.getcwd(), "colbert/tests/images_base")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        input_image_path, gt_image_paths = dump_images(item["image_files"][0], item["gt_images"], item['id'], output_dir)
        ans_file.write(json.dumps({
            "id": item['id'],
            "question": item["question"],
            "prompt": item['prompt'],
            "answer": item["answer"],
            "gt_choice": item["gt_choice"],
            "prompt_question_part": item["prompt_question_part"],
            "prompt_instruction_part": item["prompt_instruction_part"],
            "image": input_image_path,
            "gt_images": gt_image_paths,
            "gt_answer": item['answer'],
            "shortuuid": shortuuid.uuid(),
            "scenario": item['scenario'],
            "aspect": item['aspect'],
        }) + "\n")
        ans_file.flush()
        

if __name__ == "__main__":
    # test_retrieval_mragbenchmark()
    # test_retrieval_mragbenchmark_base()
    test_retrieval_mragbenchmark_image()
   