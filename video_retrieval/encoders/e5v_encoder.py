import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
from PIL import Image
import cv2
import os
from pathlib import Path
import time
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, AutoModel, LlavaNextForConditionalGeneration, LlavaNextConfig, LlavaNextProcessor
from video_retrieval import FrameExtractor, get_ucf101_video_paths, UCF101VideoPathCollector

class E5VVideoEncoder:
    def __init__(self, model_name = "royokong/e5-v", max_frames_per_video = 8, frame_cache_path = " "):

        print(f"Loading E5-V model: {model_name}")

        self.config = LlavaNextConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = LlavaNextProcessor.from_pretrained(model_name, config=self.config)
        self.processor.patch_size = self.config.vision_config.patch_size
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        
        self.model.eval()
        self.llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
        
        self.max_frames_per_video = max_frames_per_video
        self.frame_cache_path = frame_cache_path
        
        # Initialize frame extractor if cache path is provided
        if frame_cache_path != " " and os.path.exists(frame_cache_path):
            cache_dir = os.path.basename(frame_cache_path)
            experiment_root = os.path.dirname(frame_cache_path)
            self.extractor = FrameExtractor(cache_dir=cache_dir, max_frames=self.max_frames_per_video, experiment_root=experiment_root)
        else:
            self.extractor = FrameExtractor(max_frames=self.max_frames_per_video)

        print("E5-V model loaded successfully!")
    
    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        text_prompt = self.llama3_template.format('<sent>\nSummary above sentence in one word: ')
        text_inputs = self.processor([text_prompt.replace('<sent>', text) for text in texts], return_tensors="pt", padding=True).to('cuda', torch.float16)

        with torch.no_grad():
            text_embs = self.model(
                **text_inputs, 
                output_hidden_states=True, 
                return_dict=True
            ).hidden_states[-1][:, -1, :]

        text_embs = F.normalize(text_embs, dim=-1)

        return text_embs
    
    def encode_image(self, images):
        img_prompt = self.llama3_template.format('<image>\nSummary above image in one word: ')
        img_inputs = self.processor([img_prompt]*len(images), images, return_tensors="pt", padding=True).to('cuda', torch.float16)
        with torch.no_grad():
            img_embs = self.model(
                **img_inputs, 
                output_hidden_states=True, 
                return_dict=True
            ).hidden_states[-1][:, -1, :]
        
        img_embs = F.normalize(img_embs, dim=-1)

        return img_embs
    
    def encode_video(self, video_path, force_rebuild_frames = False):

        video_frames = self.extractor.extract_frames_batch([video_path], force_rebuild_frames)
        
        if video_path not in video_frames or not video_frames[video_path]:
            print(f"Failed to extract frames from {video_path}")
            return None
        
        frames = video_frames[video_path]
        frame_embeddings = self.encode_image(frames)
        
        return frame_embeddings

if __name__ == "__main__":
    print("="*60, "Testing E5VVideoEncoder Starts", "="*60)
    
    # experiment_root for frames cache
    experiment_root = os.path.join(os.getcwd(), "VideoBenchmark/experiment_test")
    os.makedirs(experiment_root, exist_ok=True)
    frame_cache_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiment_test/frame_cache_test"
    video_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"

    print("1. Loading E5VVideoEncoder...")
    e5v_encoder = E5VVideoEncoder(frame_cache_path = frame_cache_path)
    print("\n")

    print("2. Testing Encoding Video...")
    video_embedding = e5v_encoder.encode_video(video_path)
    print(f"video embedding size: {video_embedding.size()}")
    print("\n")

    print("3. Testing Encoding Text...")
    text = "ApplyEyeMakeup"
    text_embedding = e5v_encoder.encode_text(text)
    print(f"text embedding size: {text_embedding.size()}")
    print("\n")

    print("="*60, "Testing LanceDBVideoRetriever Ends", "="*60)