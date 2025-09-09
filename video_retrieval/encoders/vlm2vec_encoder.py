import sys
import os

vlm2vec_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/models/VLM2Vec"
if vlm2vec_path not in sys.path:
    sys.path.insert(0, vlm2vec_path)

import torch
from PIL import Image
from src.arguments import ModelArguments, DataArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS, VLM_IMAGE_TOKENS, Qwen2_VL_process_fn
from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info
from src.utils import batch_to_device

from video_retrieval.utils import resize_and_pad_image, extract_patches

class VLM2VecEncoder:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", checkpoint_path="TIGER-Lab/VLM2Vec-Qwen2VL-7B"):

        print(f"Loading VLM2Vec model: {checkpoint_path}")
        self.model_args = ModelArguments(
            model_name='Qwen/Qwen2-VL-7B-Instruct',
            checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-7B',
            pooling='last',
            normalize=True,
            model_backbone='qwen2_vl',
            lora=True
        )
        self.data_args = DataArguments()
        self.processor = load_processor(self.model_args, self.data_args)
        self.model = MMEBModel.load(self.model_args)
        self.model.eval()
        print("VLM2Vec model loaded successfully!")
    
    def encode_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        processor_inputs = {
            "text": texts,
            "images": [None] * len(texts),
        }
        inputs = Qwen2_VL_process_fn(
            processor_inputs,
            self.processor
        )
        inputs = batch_to_device(inputs, "cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tgt_output = self.model(tgt=inputs)["tgt_reps"]
        del inputs
        return tgt_output
    
    def encode_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        text_inputs = [f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image'] * len(images)
        processor_inputs = {
            "text": text_inputs,
            "images": images
        }
        inputs = Qwen2_VL_process_fn(
            processor_inputs,
            self.processor)
        inputs = batch_to_device(inputs, "cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                qry_output = self.model(qry=inputs)["qry_reps"]
        del inputs
        return torch.cat([emb.unsqueeze(0) for emb in qry_output], dim=0)
    
    def encode_video(self, video_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
            videos=video_inputs,
            return_tensors="pt"
        )
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
        inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
        qry_output = self.model(qry=inputs)["qry_reps"]
        del inputs
        return qry_output
    
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

        batch_size = 4
        total_batch = ((len(image_paths) - 1) // batch_size) + 1
        img_embs = []
        for batch_idx in range(total_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx+batch_size-1, len(image_paths)-1)
            img_embs.append(self.encode_images(processed_images[start_idx:end_idx+1]).to("cpu"))
        return torch.cat(img_embs,dim=0)

    def resize_images(self, images, target_resolution=(336, 336)):
        if isinstance(images, Image.Image):
            images = [images]
        return [resize_and_pad_image(img, target_resolution) for img in images]
    
    def get_images_patches(self, images, patch_size=112, overlap_ratio=0):
        if isinstance(images, Image.Image):
            images = [images]
        return [extract_patches(img, patch_size, overlap_ratio) for img in images]
        
    def encode_images_patches(self, images, target_resolution=(336,336), patch_size=112, overlap_ratio=0):
        if isinstance(images, Image.Image):
            images = [images]

        resized_images = self.resize_images(images, target_resolution)
        images_patches = self.get_images_patches(resized_images, patch_size=patch_size, overlap_ratio=overlap_ratio)
        assert len(images_patches) == len(images)
        processed_images = []
        patch_num = len(images_patches[0]) + 1
        for image, image_patches in zip(images, images_patches):
            image_patches.append(image)
            processed_images += image_patches
        assert len(processed_images) == len(images) * patch_num
        batch_size = 4
        total_batch = ((len(processed_images) - 1) // batch_size) + 1
        img_embs = []
        for batch_idx in range(total_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx+batch_size-1, len(processed_images)-1)
            img_embs.append(self.encode_images(processed_images[start_idx:end_idx+1]).to("cpu"))
        img_embs = torch.cat(img_embs,dim=0).view(len(images), patch_num, -1)
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
    print("="*60, "Testing VLM2VecEncoder Starts", "="*60)
    
    print("1. Loading VLM2VecEncoder...")
    vlm2vec_encoder = VLM2VecEncoder()
    print("\n")

    print("2. Testing Encoding Image...")
    image_path1 = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/applauding/applauding_001.jpg"
    image_path2 = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/feeding_a_horse/feeding_a_horse_001.jpg"
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    images = [image1, image2]
    image_embedding = vlm2vec_encoder.encode_images_patches(images)
    # image_embedding = vlm2vec_encoder.get_image_token_emb(image1)
    print(f"image embeddings num: {len(image_embedding)}")
    print(f"image embeddings size: {image_embedding[0].size()}") # torch.Size([5, 3584])
    print("\n")
    del image_embedding

    print("3. Testing Encoding Text...")
    text1 = "applauding"
    text1_embedding = vlm2vec_encoder.encode_texts(text1)
    # text1_embedding = llava_encoder.get_text_token_emb(text1)
    print(f"text embedding size: {text1_embedding.size()}") # torch.Size([1, 3584])
    text2 = "BabyCrawling"
    text2_embedding = vlm2vec_encoder.encode_texts(text2)
    print(f"text embedding size: {text2_embedding.size()}")
    print("\n")

    print("4. Testing Encoding Video...")
    video_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"
    video_embedding = vlm2vec_encoder.encode_video(video_path)
    print(f"video embedding size: {video_embedding.size()}") # torch.Size([1, 3584])
    print("\n")
    del video_embedding

    # print("5. Testing Similarity...")
    # sim1= torch.matmul(text1_embedding, video_embedding.T)
    # print(sim1)
    # sim2= torch.matmul(text2_embedding, video_embedding.T)
    # print(sim2)

    # print("6. Testing Encoding Images...")
    # image_path1 = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/applauding/applauding_001.jpg"
    # image_paths = [image_path1] * 21
    # image_embedding = vlm2vec_encoder.encode_image_from_paths(image_paths)
    # # image_embedding = vlm2vec_encoder.get_image_token_emb(image1)
    # print(f"image embeddings num: {len(image_embedding)}")
    # print(f"image embeddings size: {image_embedding[0].size()}") # torch.Size([5, 3584])
    # print("\n")

    print("6. Testing Encoding Images...")
    image_path1 = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/applauding/applauding_001.jpg"
    image_paths = [image_path1] * 2
    image_embedding = vlm2vec_encoder.encode_images_patches_from_paths(image_paths)
    # image_embedding = vlm2vec_encoder.get_image_token_emb(image1)
    print(f"image embeddings num: {len(image_embedding)}") # 2
    print(f"image embeddings size: {image_embedding.size()}") # torch.Size([2, 10, 3584])
    print("\n")

    print("="*60, "Testing VLM2VecEncoder Ends", "="*60)

    