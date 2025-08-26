# pip install git+https://github.com/DeepLearnXMU/LLaVE
import torch
import copy
import os
import sys
import glob
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from video_retrieval.encoders.intern_video2_encoder import ImageProcessor, VideoProcessor

class LLaVAQwenEncoder:
    def __init__(self, pretrained = "zhibinlan/LLaVE-7B", model_name = "llava_qwen", max_frames_per_video = 8):

        print(f"Loading llava model: {model_name}")
        self.device = "cuda"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, device_map="auto")  # Add any other thing you want to pass in llava_model_args
        self.model.eval()
        self.video_processor = VideoProcessor()
        self.conv_template = "qwen_1_5" 
        self.max_frames_per_video = max_frames_per_video

        print("LLaVAQwen model loaded successfully!")
    
    def process_video(self, video_path, bound=None, max_num=1, num_segments=8, get_frame_by_duration=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        frame_indices = self.video_processor._get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        images = []
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            images.append(img)
    
        return images

    def encode_text(self, text):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], "\n")
        target_string = conv.get_prompt()
        target_input_ids = self.tokenizer(target_string, return_tensors="pt").input_ids.to(self.device)
        attention_mask=target_input_ids.ne(self.tokenizer.pad_token_id)

        with torch.no_grad():
            target_embed = self.model.encode_multimodal_embeddings(
                target_input_ids, 
                attention_mask=attention_mask
            )

        return target_embed
    def images_to_tensor(self, images):
        images_tensor = process_images(images, self.image_processor, self.model.config)
        images_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in images_tensor]
        return images_tensor

    def encode_image(self, image, image_size=None):
        if isinstance(image, Image.Image):
            image_lst = [image]
            image_tensor = self.images_to_tensor(image_lst)
            image_size = image.size
        else:
            image_tensor = image

        conv = copy.deepcopy(conv_templates[self.conv_template])
        question = DEFAULT_IMAGE_TOKEN + " Represent the given image with the following question: What is in the image"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], "\n")
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        image_sizes = [image_size]
        with torch.no_grad():
            img_embs = self.model.encode_multimodal_embeddings(
                input_ids, 
                attention_mask=attention_mask,
                images=image_tensor, 
                image_sizes=image_sizes
            )
        del input_ids
        del attention_mask
        return img_embs

    def encode_images(self, images):
        images_tensor = self.images_to_tensor(images)
        img_embs = []
        for image, image_tensor in zip(images, images_tensor):
            image_size = image.size
            img_emb = self.encode_image(image=image_tensor, image_size=image_size)
            if img_emb.dim() == 1:
                img_emb.unsqueeze(0)
            img_embs.append(img_emb)
        return torch.cat(img_embs)
    
    def encode_video(self, video_path, force_rebuild_frames=False):
        frames = self.process_video(video_path)
        video_emb = self.encode_images(frames)
        return video_emb

    def encode_image_from_paths(self, image_paths):
        # processed_images = []
        # for img_path in image_paths:
        #     if not os.path.exists(img_path):
        #         raise FileNotFoundError(f"Image not found: {img_path}")
        #     img = Image.open(img_path)
           
        #     if img.mode in ('P', 'LA', 'PA'):
        #         img = img.convert('RGBA').convert('RGB')
        #     else:
        #         img = img.convert('RGB')
        #     processed_images.append(img)
        # batch_size = 3
        # batch_max = ((len(processed_images) -1) // batch_size) + 1
        # img_embs = []
        # for batch_idx in tqdm(range(batch_max)):
        #     start_idx = batch_idx * batch_size
        #     end_idx = min(start_idx + batch_size - 1, len(processed_images)-1)
        #     batch_img_embs = self.encode_image(processed_images[start_idx:end_idx+1])
        #     if batch_img_embs.dim() == 1:
        #         batch_img_embs.unsqueeze(0)
        #     img_embs.append(batch_img_embs)
        #     del batch_img_embs
        # return torch.cat(img_embs, dim=0)
        pass


if __name__ == "__main__":
    print("="*60, "Testing LLaVAQwenEncoder Starts", "="*60)
    
    print("1. Loading LLaVAQwenEncoder...")
    llava_encoder = LLaVAQwenEncoder()
    print("\n")

    print("2. Testing Encoding Image...")
    image_path1 = "/research/d7/fyp25/yqliu2/projects/ColBERT/data/image_data/image_corpus/Biological_0_gt_77fa0cfd88f3bb00ed23789a476f0acd---d.jpg"
    image_path2 = "/research/d7/fyp25/yqliu2/projects/ColBERT/data/image_data/image_corpus/Biological_0_gt_409E7C55-DDA5-442E-BEF368457F16CAA7.jpg"
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    images = [image1, image2]
    image_embedding = llava_encoder.encode_images(images)
    print(f"video embedding size: {image_embedding.size()}")
    print("\n")

    print("3. Testing Encoding Text...")
    text1 = "ApplyEyeMakeup"
    text1_embedding = llava_encoder.encode_text(text1)
    print(f"text embedding size: {text1_embedding.size()}")
    text2 = "BabyCrawling"
    text2_embedding = llava_encoder.encode_text(text2)
    print(f"text embedding size: {text2_embedding.size()}")
    print("\n")

    print("4. Testing Encoding Video...")
    video_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
    video_embedding = llava_encoder.encode_video(video_path)
    print(f"video embedding size: {video_embedding.size()}")
    print("\n")

    print("5. Testing Similarity...")
    sim1= torch.matmul(text1_embedding, video_embedding.T)
    print(sim1)
    sim2= torch.matmul(text2_embedding, video_embedding.T)
    print(sim2)

    print("="*60, "Testing LanceDBVideoRetriever Ends", "="*60)

