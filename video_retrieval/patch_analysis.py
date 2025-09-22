import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from PIL import Image
import os
import json
import math
import copy
import sys
import glob
import time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from tqdm import tqdm
import asyncio
import base64
from io import BytesIO
from openai import AsyncOpenAI
from dotenv import load_dotenv
import random

# 在工作目录下建一个.env文件，文件里写：OPENAI_API_KEY=XXX (你的apikey,不加引号)
load_dotenv()

# 让gpt描述图片的prompt，随便改
prompt_templates = {
    "template_caption1": (
        "Here is an example of a comprehensive description for finding visually similar action images:\n"
        "Example:\n"
        "Image: Person brushing teeth, standing at sink, looking in mirror.\n"
        "Description: Person brushing teeth while standing at bathroom sink, looking at mirror, right hand holding toothbrush, left hand on sink, morning routine setting, side profile view, wearing casual t-shirt, average build.\n"
        "Now describe this new image: [your image]\n"
        "Generate a single, comprehensive description that captures the specific action variation, body positioning, viewing angle, clothing style, general build, and environmental context. Focus on visual features that help identify similar action variations for image-to-image retrieval.\n"
    ),
    "template_caption2": "Describe this image in detail."
}

def colbert_maxsim_mean(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None):
    '''
    Args:
        query_emb: size为[total_query_tokens, embedding_dim]的torch.Tensor,total_query_tokens是每个query的token数量之和,
            比如query1 = tok11, tok12, tok13, query2 = tok21, tok22, query_emb就是[tok11_emb, ...tok21_emb,tok22_emb]
        video_emb: size为[total_video_frames, embedding_idm]的torch.Tensor,total_video_frames是database每个video的帧数量
            (video)之和或者database每个图片的patch数量之和(image),所有video/image采用跟query_emb一样的方式拼成video_emb
        query_token_nums: 一个记录每个query token数量的List, 比如query_emb里示例里query1 token数量为3,query2 token数量为2, 
            query_token_num = [3,2]
        video_num: 记录database里video/image数量,因为每个库video切帧image切patch的数量是固定的,所以只需要记录总体video/image的数量
            就可以由video_emb算出帧数/patch数量

    Returns:
        torch.Tensor of size [query_num, video_num],即每个query对应每个video/image的similarity score
    '''
    query_norm = F.normalize(query_emb, p=2, dim=-1)  # [total_query_tokens, embedding_dim]
    video_norm = F.normalize(video_emb, p=2, dim=-1)  # [total_video_frames, embedding_dim]
    
    s = torch.matmul(query_norm, video_norm.T)  # [total_query_tokens, total_video_frames]
    
    # [total_query_tokens, total_video_frames] -> [total_query_tokens, video_num, frame_num]
    if video_num is not None:
        s = s.view(query_emb.shape[0], video_num, -1) 
    
    if query_token_nums is not None:
        similarities = []
        start_idx = 0
        for token_num in query_token_nums:
            end_idx = start_idx + token_num
            query_sim = s[start_idx:end_idx]  # [token_num, video_num, frame_num]
            max_sim_per_token = query_sim.max(dim=2).values  # [token_num, video_num]
            similarities.append(max_sim_per_token.mean(dim=0))  # [video_num]
            start_idx = end_idx
        return torch.stack(similarities)  # [num_queries, video_num]
    else:
        max_sim_per_token = s.max(dim=2).values  # [total_query_tokens, video_num]
        return max_sim_per_token.mean(dim=0)  # [video_num]

# copy from llava mm_utils.py，用于resize和padding，详细用法看encoder
def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

# copy from llava mm_utils.py，用于切patch，切之前要先resize和padding图片到相同规格，用法看encoder
def extract_patches(image, patch_size, overlap_ratio):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))

    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1

    y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
    x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2

    for y in range(y_start, y_start + num_patches_y * stride, stride):
        for x in range(x_start, x_start + num_patches_x * stride, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches

class LLaVAQwenEncoder:
    def __init__(self, pretrained = "zhibinlan/LLaVE-7B", model_name = "llava_qwen"):

        print(f"Loading llava model: {model_name}")
        self.device = "cuda"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, device_map="auto", attn_implementation="flash_attention_2") 
        self.model.eval()
        self.conv_template = "qwen_1_5" 

        print("LLaVAQwen model loaded successfully!")

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
        for image, image_tensor in tqdm(zip(images, images_tensor), unit="image"):
            image_size = image.size
            img_emb = self.encode_image(image=image_tensor, image_size=image_size)
            if img_emb.dim() == 1:
                img_emb.unsqueeze(0)
            img_embs.append(img_emb)
        return torch.cat(img_embs)
    
    def encode_image_from_paths(self, image_paths):
        processed_images = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path)
           
            if img.mode in ('P', 'LA', 'PA'):
                img = img.convert('RGBA').convert('RGB')
            else:
                img = img.convert('RGB')
            processed_images.append(img)

        img_embs = self.encode_images(processed_images)
        return img_embs

    def resize_images(self, images, target_resolution=(336, 336)):
        if isinstance(images, Image.Image):
            images = [images]
        return [resize_and_pad_image(img, target_resolution) for img in images]
    
    def get_images_patches(self, images, patch_size=112, overlap_ratio=0):
        if isinstance(images, Image.Image):
            images = [images]
        return [extract_patches(img, patch_size, overlap_ratio) for img in images]
        
    def encode_images_patches(self, images, target_resolution=(336,336), patch_size=84, overlap_ratio=0):
        if isinstance(images, Image.Image):
            images = [images]

        resized_images = self.resize_images(images, target_resolution)
        images_patches = self.get_images_patches(resized_images, patch_size=patch_size, overlap_ratio=overlap_ratio)
        assert len(images_patches) == len(images)

        img_embs = []
        for image, image_patches in zip(images, images_patches):
            # the last one is the original image
            image_patches.append(image)
            img_emb = self.encode_images(image_patches) # [9, hidden_dim]
            img_embs.append(img_emb)
        
        return img_embs
    
    def encode_images_patches_from_paths(self, image_paths):
        processed_images = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path)
           
            if img.mode in ('P', 'LA', 'PA'):
                img = img.convert('RGBA').convert('RGB')
            else:
                img = img.convert('RGB')
            processed_images.append(img)

        img_embs = self.encode_images_patches(processed_images)
        return img_embs

# 注意asyncio的语法，直接问cusor，这个可以一次处理5个请求，比较快
class GPTGenerator:
    def __init__(self, api_key="", model_name="gpt-4o"):
        self.model_name = model_name
        print(f"Initializing GPT API with model: {model_name}")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        try:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.chatanywhere.tech")
        except Exception as e:
            print(f"Error loading openai client: {e}")
        
        print("GPT API client initialized successfully!")
    
    def _image_to_base64(self, image: Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    async def generate_description(self, image_path, prompt, max_retries = 10):
        image = Image.open(image_path)
        if image.mode in ('P', 'LA', 'PA'):
            image = image.convert('RGBA').convert('RGB')
        else:
            image = image.convert('RGB')
        
        base64_image = self._image_to_base64(image)
        
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
        start_time = time.time()
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=200,
                    temperature=0.1
                )
                
                description = response.choices[0].message.content.strip()
                return (image_path, description)
                
            except Exception as e:
                error_msg = str(e)
                
                if "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        elapsed = time.time() - start_time
                        if elapsed > 45:  # After 45 seconds, reduce wait time
                            base_wait = min(base_wait, 6) 
                            base_wait += random.uniform(0, 1)
                        print(f"Rate limit hit for {image_path}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded for {image_path} after {max_retries} retries")
                        return (image_path, None)
                else:
                    print(f"Error processing {image_path}: {error_msg}")
                    return (image_path, None)
            
            return (image_path, None)

# 注意asyncio语法，在主函数要用asyncio.run()来调用， 直接问gpt。调用GPTGenerator.generate_description的模版，这里输入是image paths list，你可以改一下GPTGenerato.generate_description让他接受List[Image.Image]
async def process_batch(generator: GPTGenerator, image_paths: List[str], prompt: str, max_concurrent: int = 5) -> List[Tuple[str, str]]:
    
    print(f"Processing {len(image_paths)} images concurrently with max {max_concurrent} concurrent requests...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(image_path):
        async with semaphore:
            return await generator.generate_description(image_path, prompt)
    
    tasks = [process_with_semaphore(path) for path in image_paths]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Exception occurred for {image_paths[i]}: {result}")
            processed_results.append((image_paths[i], None))
        else:
            processed_results.append(result)
    
    processed_results.sort(key=lambda x: image_paths.index(x[0]))
    
    # Print summary
    successful = sum(1 for _, desc in processed_results if desc is not None)
    print(f"Successfully processed {successful}/{len(image_paths)} images")
    
    return processed_results 


