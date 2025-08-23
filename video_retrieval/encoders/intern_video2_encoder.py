import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448

class ImageProcessor:
    def __init__(self, input_size=448):
        self.input_size = input_size
        self.transform = self._build_transform()
    
    def _build_transform(self):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(self, image, min_num=1, max_num=6, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set((i, j) for n in range(min_num, max_num + 1) 
                           for i in range(1, n + 1) for j in range(1, n + 1) 
                           if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        target_aspect_ratio = self._find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, self.input_size)
        
        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = ((i % (target_width // self.input_size)) * self.input_size,
                   (i // (target_width // self.input_size)) * self.input_size,
                   ((i % (target_width // self.input_size)) + 1) * self.input_size,
                   ((i // (target_width // self.input_size)) + 1) * self.input_size)
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size, self.input_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def process_image(self, image, max_num=6):
        images = self.dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

class VideoProcessor:
    def __init__(self, input_size=448):
        self.input_size = input_size
        self.image_processor = ImageProcessor(input_size)
    
    def _get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
        return frame_indices
    
    def _get_num_frames_by_duration(self, duration):
        local_num_frames = 4
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)
        return num_frames
    
    def process_video(self, video_path, bound=None, max_num=1, num_segments=32, get_frame_by_duration=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        pixel_values_list, num_patches_list = [], []
        
        if get_frame_by_duration:
            duration = max_frame / fps
            num_segments = self._get_num_frames_by_duration(duration)
        
        frame_indices = self._get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = self.image_processor.dynamic_preprocess(img, use_thumbnail=True, max_num=max_num)
            pixel_values = [self.image_processor.transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

class InternVideo2Encoder:
    def __init__(self, model_name="OpenGVLab/InternVideo2_5_Chat_8B"):
        print(f"Loading InternVideo2 model: {model_name}")
        
        self.model_name = model_name
        self.input_size = INPUT_SIZE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda().to(torch.bfloat16)
        self.model.eval()
        self.image_processor = ImageProcessor(self.input_size)
        self.video_processor = VideoProcessor(self.input_size)
        
        # Import conversation
        self.import_conversation()
        
        print("InternVideo2 model loaded successfully!")
    
    def import_conversation(self):
        cache_dir = os.environ.get("TRANSFORMERS_CACHE")
        model_path = self.model_name
        model_dir = "models--" + model_path.replace("/", "--")
        model_dir = os.path.join(cache_dir, model_dir)
        snapshots_dir = os.path.join(model_dir, "snapshots")
        conversation_files = glob.glob(os.path.join(snapshots_dir, "*", "conversation.py"))
        conversation_dir = os.path.dirname(conversation_files[0])
        sys.path.append(conversation_dir)
        try:
            from conversation import get_conv_template, Conversation
            self.get_conv_template = get_conv_template
            self.Conversation = Conversation
            print("Successfully import conversation!")
        except Exception as e:
            raise Exception(f"Import Error: {e}")
    
    def encode_image(self, image, max_num=6):
        pass
    
    def encode_video(self, video_path, bound=None, max_num=1, num_segments=8, get_frame_by_duration=False):
        with torch.no_grad():
            pixel_values, num_patches_list = self.video_processor.process_video(video_path, bound, max_num, num_segments, get_frame_by_duration)
            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
            outputs = self.model.vision_model(pixel_values)
            video_embs = outputs.pooler_output
        return video_embs
    def encode_video2(self, video_path, bound=None, max_num=1, num_segments=8, get_frame_by_duration=False):
        with torch.no_grad():
            pixel_values, num_patches_list = self.video_processor.process_video(video_path, bound, max_num, num_segments, get_frame_by_duration)
            pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
            video_embs = self.model.extract_feature(pixel_values)
            breakpoint()
        return video_embs
    def encode_video3(self, video_path, bound=None, max_num=1, num_segments=8, get_frame_by_duration=False):
        # with torch.no_grad():
        #     pixel_values, num_patches_list = self.video_processor.process_video(video_path, bound, max_num, num_segments, get_frame_by_duration)
        #     # video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        #     pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
        #     video_embs = self.model.extract_feature(pixel_values)
        #     breakpoint()
        # return video_embs
        pass
    def encode_text(self, text):
        with torch.no_grad():
            model_inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
            outputs = self.model.language_model.model(**model_inputs)
            text_embs = outputs.last_hidden_state
        text_proj = torch.nn.Linear(4096, 1024, bias=False).to(dtype=torch.bfloat16, device=self.model.device)
        text_embs = text_proj(text_embs)
        return text_embs
    
    def encode_text_with_template(self, text, template_name='phi3-chat'):
        template = self.get_conv_template(template_name)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())
        
        template.append_message(template.roles[0], text)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        
        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.model.device)
        with torch.no_grad():
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        
        return input_embeds, eos_token_id
    
    def generate_response(self, input_embeds, generation_config):
        # outputs = self.model.language_model.generate(
        #     inputs_embeds=input_embeds,
        #     generation_config=generation_config,
        #     use_cache=True
        # )
        pass
    
    def encode_video_text(self, video_path, text, num_segments=128, max_num=1):
        # pixel_values, num_patches_list = self.encode_video(
        #     video_path, num_segments=num_segments, max_num=max_num
        # )
        # pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
        
        # input_embeds, eos_token_id = self.encode_text(text)
        
        # return pixel_values, input_embeds, num_patches_list, eos_token_id
        pass
    
    def generate_video_response(self, pixel_values, text, num_patches_list, generation_config):
        # vit_embeds = self.model.extract_feature(pixel_values)
        
        # input_embeds, eos_token_id = self.encode_text(text)
        # generation_config['eos_token_id'] = eos_token_id
        
        # B, N, C = input_embeds.shape
        # input_embeds = input_embeds.reshape(B * N, C)
        # input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'].reshape(B * N)
        
        # selected = (input_ids == self.model.img_context_token_id)
        # input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        # input_embeds = input_embeds.reshape(B, N, C)
        
        # outputs = self.model.language_model.generate(
        #     inputs_embeds=input_embeds,
        #     generation_config=generation_config,
        #     use_cache=True
        # )
        pass
    def save_embeddings(self, embeddings_dict, save_dir, save_format='torch'):
        pass

if __name__ == "__main__":
    print("="*60, "Testing InternVideo2Encoder Starts", "="*60)
    
    video_path = "/research/d7/fyp25/yqliu2/projects/ColBERT/VideoBenchmark/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"

    print("1. Loading E5VVideoEncoder...")
    iv2_encoder = InternVideo2Encoder()
    print("\n")

    print("2. Testing Encoding Video...")
    video_embedding = iv2_encoder.encode_video(video_path)
    print(f"video embedding size: {video_embedding.size()}")
    print("\n")

    print("3. Testing Encoding Text...")
    text = "ApplyEyeMakeup"
    text_embedding = iv2_encoder.encode_text(text)
    print(f"text embedding size: {text_embedding.size()}")
    print("\n")

    print("="*60, "Testing LanceDBVideoRetriever Ends", "="*60)

