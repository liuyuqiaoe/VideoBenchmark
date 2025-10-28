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
from llava.mm_utils import tokenizer_image_token, process_images, resize_and_pad_image, extract_patches
from video_retrieval.encoders.intern_video2_encoder import ImageProcessor, VideoProcessor
from tqdm import tqdm

class LLaVAQwenEncoder:
    def __init__(self, pretrained = "zhibinlan/LLaVE-7B", model_name = "llava_qwen", max_frames_per_video = 8):

        print(f"Loading llava model: {model_name}")
        self.device = "cuda"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, device_map="auto", attn_implementation="sdpa")  # "flash_attention_2"
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
    
    def encode_texts(self, texts):
        texts = [texts] if isinstance(texts, str) else texts
        text_embs_lst = [self.encode_text(text) for text in texts]
        return torch.concat(text_embs_lst, dim=0)

    def get_text_token_emb(self, text):
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], "\n")
        target_string = conv.get_prompt()
        target_input_ids = self.tokenizer(target_string, return_tensors="pt").input_ids.to(self.device)
        attention_mask=target_input_ids.ne(self.tokenizer.pad_token_id)

        (input_ids, 
         position_ids, 
         attention_mask, 
         past_key_values, 
         inputs_embeds, 
         labels) = self.model.prepare_inputs_labels_for_multimodal(target_input_ids, attention_mask=attention_mask, position_ids=None, past_key_values=None, labels=None, images=None)
        with torch.no_grad():
            target_embed = self.model.get_model()(
                input_ids=input_ids, 
                position_ids=position_ids, 
                attention_mask=attention_mask, 
                past_key_values=past_key_values, 
                inputs_embeds=inputs_embeds, 
                # labels=None, # labels=labels
                use_cache=False,
                output_attentions=None,
                return_dict=True, 
                output_hidden_states=True,
                ).hidden_states[-1][:,:-1,:]

        return target_embed

    def get_image_token_emb(self, image, image_size=None):
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

        (input_ids, 
         position_ids, 
         attention_mask, 
         past_key_values, 
         inputs_embeds, 
         labels) = self.model.prepare_inputs_labels_for_multimodal(input_ids, attention_mask=attention_mask, position_ids=None, past_key_values=None, labels=None, images=image_tensor, modalities=["image"], image_sizes=image_sizes)
        with torch.no_grad():
            target_embed = self.model.get_model()(
                input_ids=input_ids, 
                position_ids=position_ids, 
                attention_mask=attention_mask, 
                past_key_values=past_key_values, 
                inputs_embeds=inputs_embeds, 
                # labels=None, # labels=labels
                use_cache=False,
                output_attentions=None,
                return_dict=True, 
                output_hidden_states=True,
                ).hidden_states[-1][:,:-1,:]

        del input_ids
        del attention_mask
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
    
    def encode_image_text_pair(self, text, image, image_size=None):
        if isinstance(image, Image.Image):
            image_lst = [image]
            image_tensor = self.images_to_tensor(image_lst)
            image_size = image.size
        else:
            image_tensor = image
        
        if not isinstance(text, str):
            text = text[0]

        conv = copy.deepcopy(conv_templates[self.conv_template])
        question = DEFAULT_IMAGE_TOKEN + " " + text
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

    def encode_image_text_pairs(self, texts, images, image_size=None):
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(texts, str):
            texts = [texts]

        img_text_embs = []
        for text, image in tqdm(zip(texts, images), unit="pair"):
            img_text_emb = self.encode_image_text_pair(text, image)
            if img_text_emb.dim() == 1:
                img_text_emb.unsqueeze(0)
            img_text_embs.append(img_text_emb)
        return torch.cat(img_text_embs)

    def encode_image_text_pairs_from_paths(self, texts, image_paths):
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
        img_text_embs = self.encode_image_text_pairs(texts, processed_images)
        return img_text_embs

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
    
    def encode_video(self, video_path, force_rebuild_frames=False):
        frames = self.process_video(video_path)
        video_emb = self.encode_images(frames)
        return video_emb

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


if __name__ == "__main__":
    print("="*60, "Testing LLaVAQwenEncoder Starts", "="*60)
    
    print("1. Loading LLaVAQwenEncoder...")
    llava_encoder = LLaVAQwenEncoder()
    print("\n")

    # print("2. Testing Encoding Image...")
    # query_image = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_156.jpg"
    # gt_image = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/test/fishing/fishing_159.jpg"
    # neg_image = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_191.jpg"
    # images = [query_image, gt_image, neg_image]
    # images = [Image.open(img_path) for img_path in images]
    # text = "Replace this person in this image with a middle-aged man with a stocky build wearing a green fishing vest over a white t-shirt, paired with brown waders, while sporting a baseball cap and sunglasses as he stands in the water holding a fishing rod."
    # query_emb = llava_encoder.encode_image_text_pair(text, images[0])
    # query_img_emb = llava_encoder.encode_image(images[0])
    # gt_img_emb = llava_encoder.encode_image(images[1])
    # neg_img_emb = llava_encoder.encode_image(images[2])
    # sim0 = torch.matmul(query_emb, query_img_emb.T)
    # sim1 = torch.matmul(query_emb, gt_img_emb.T)
    # sim2 = torch.matmul(query_emb, neg_img_emb.T)
    # image_embedding = llava_encoder.get_image_token_emb(image1)
    # print(f"sim0: {sim0}, sim1: {sim1}, sim2: {sim2}")
    # print("\n")

    # print("3. Testing Encoding Text...")
    # text1 = "applauding"
    # text1_embedding = llava_encoder.encode_text(text1)
    # # text1_embedding = llava_encoder.get_text_token_emb(text1)
    # print(f"text embedding size: {text1_embedding.size()}")
    # text2 = "BabyCrawling"
    # text2_embedding = llava_encoder.encode_text(text2)
    # print(f"text embedding size: {text2_embedding.size()}")
    # print("\n")
    # print("4. Testing Encoding Video...")
    # video_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
    # video_embedding = llava_encoder.encode_video(video_path)
    # print(f"video embedding size: {video_embedding.size()}")
    # print("\n")

    # print("5. Testing Similarity...")
    # sim1= torch.matmul(text1_embedding, video_embedding.T)
    # print(sim1)
    # sim2= torch.matmul(text2_embedding, video_embedding.T)
    # print(sim2)

    # print("6. Testing Encoding Image Text Pairs...")
    # texts = ["Replace this person in this image with a middle-aged man with a stocky build wearing a green fishing vest over a white t-shirt, paired with brown waders, while sporting a baseball cap and sunglasses as he stands in the water holding a fishing rod."] * 3
    # images = ["/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_156.jpg"] * 3
    # embs = llava_encoder.encode_image_text_pairs_from_paths(texts, images)
    # breakpoint()
    # print(f"embs size: {embs.size()}")
    # print("\n")

    texts = ["Replace this person in this image with a middle-aged man with a stocky build wearing a green fishing vest over a white t-shirt, paired with brown waders, while sporting a baseball cap and sunglasses as he stands in the water holding a fishing rod."] * 3
    embs = llava_encoder.encode_texts(texts)

    print("="*60, "Testing LanceDBVideoRetriever Ends", "="*60)

