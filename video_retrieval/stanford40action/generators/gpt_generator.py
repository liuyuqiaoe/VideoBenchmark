import asyncio
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple, List
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
import random
import os
import time
import json
from tqdm import tqdm

load_dotenv()

prompt_templates = {
    "template_caption1": (
        "Here is an example of a comprehensive description for finding visually similar action images:\n"
        "Example:\n"
        "Image: Person brushing teeth, standing at sink, looking in mirror.\n"
        "Description: Person brushing teeth while standing at bathroom sink, looking at mirror, right hand holding toothbrush, left hand on sink, morning routine setting, side profile view, wearing casual t-shirt, average build.\n"
        "Now describe this new image: [your image]\n"
        "Generate a single, comprehensive description that captures the specific action variation, body positioning, viewing angle, clothing style, general build, and environmental context. Focus on visual features that help identify similar action variations for image-to-image retrieval.\n"
    ),
    "template_caption2": "Describe this image in detail.",
    "template_desc": (
        "You are an expert image analyst. Please analyze the person in the image and write a single, natural sentence briefly describing their appearance.\n"
        "Write a comprehensive description in one sentence that includes:\n" 
        "Physical appearance (age, gender),\n"
        "Clothing and accessories (what they're wearing, colors, style).\n" 
        "Here are some examples of the desired format:\n" 
        "A young woman wearing a white t-shirt and dark blue jeans.\n"
        "A teenage girl wearing a bright yellow tank top and black shorts.\n"
        "Generate the description now:\n"
    )
}

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
    
    async def get_image_from_text(self, text, image_path, max_retries = 10):
        start_time = time.time()
        for attempt in range(max_retries):
            try:
                response = await self.client.images.generate(
                    model="gpt-image-1",
                    prompt=text,
                    size="256x256",
                    quality="medium",
                    n=1
                )
                
                img_bytes = base64.b64decode(response.data[0].b64_json)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                img.save(image_path, format="JPEG", quality=85, optimize=True)
                return (text, image_path)
                
            except Exception as e:
                error_msg = str(e)
                
                if "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        elapsed = time.time() - start_time
                        if elapsed > 45:  # After 45 seconds, reduce wait time
                            wait_time = min(wait_time, 6) 
                            wait_time += random.uniform(0, 1)
                        print(f"Rate limit hit for {image_path}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded for {image_path} after {max_retries} retries")
                        return (text, None)
                else:
                    print(f"Error processing {image_path}: {error_msg}")
                    return (text, None)
            
            return (text, None)

    async def get_text_embedding(self, text, max_retries=10):
        start_time = time.time()
        for attempt in range(max_retries):
            try:
                response = await self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
                
            except Exception as e:
                error_msg = str(e)
                
                if "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        elapsed = time.time() - start_time
                        if elapsed > 45:  # After 45 seconds, reduce wait time
                            wait_time = min(wait_time, 6) + random.uniform(0, 1)
                        print(f"Rate limit hit for text embedding. Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded for text embedding after {max_retries} retries")
                        return None
                else:
                    print(f"Error getting text embedding: {error_msg}")
                    return None
        
        return None

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

async def process_batch_text_embedding(generator: GPTGenerator, texts: List[str], max_concurrent: int = 5):
    if isinstance(texts, str):
        texts = [texts]

    print(f"Processing {len(texts)} texts concurrently with max {max_concurrent} concurrent requests...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(text):
        async with semaphore:
            return await generator.get_text_embedding(text)
    
    tasks = [process_with_semaphore(text) for text in texts]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Exception occurred for {texts[i]}: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)

    return processed_results

async def generate_descs(selected_images_file, output_path):
    with open(selected_images_file, "r") as f:
        selected_images = json.load(f)

    data = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
            data = data["descs"]

    generator = GPTGenerator()
    prompt = prompt_templates["template_caption1"]

    i = 0
    for action_label, img_paths in selected_images.items():
        if i == 0:
            i += 1
            continue
      
        res = await process_batch(generator=generator, image_paths=img_paths, prompt=prompt, max_concurrent=5)
        data[action_label] = res
        output = {
            "selected_images": selected_images,
            "descs": data
            }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

# testing function
async def test():
    generator = GPTGenerator()
    text = "A middle-aged man wearing a green fishing vest, white t-shirt, and brown waders, with a cap and sunglasses."
    image_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/tmp/test.jpg"
    
    result = await generator.get_image_from_text(text=text, image_path=image_path)

async def test_batch_process():
    generator = GPTGenerator()
    image_paths = [
        "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/test/fishing/fishing_159.jpg",
        "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_232.jpg"
    ]
    prompt = prompt_templates["template_desc"]
    result = await process_batch(generator=generator, image_paths=image_paths, prompt=prompt, max_concurrent=5)
    print(result)

async def modify_dataset():
    dataset_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_6_experiments/datasets/ImageText2Image4.json"
    with open(dataset_path, "r") as f:
        data = json.load(f)
    generator = GPTGenerator()
    prompt = prompt_templates["template_desc"]
    for qa_pair in data["qa_pairs"]:
        res = await process_batch(generator=generator, image_paths=qa_pair["image_path"], prompt=prompt, max_concurrent=5)
        qa_pair["question_text"] = res[0][1]
    output_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/datasets/ImageText2Image5.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

async def check_models():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.chatanywhere.tech")
    models = await client.models.list()
    for model in models.data:
        print(model.id)

def test_new():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.chatanywhere.org")
    with open("/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/datasets/ImageText2Image5.json", "r") as f:
        dataset_base = json.load(f)
    output_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/datasets/ImageImage2Image.json"
    new_qa_pairs = []
    img_root = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/10_22_experiments/images"
    idx = 0
    for i in tqdm(range(1, len(dataset_base["qa_pairs"]), 5)):
        qa_pair = dataset_base["qa_pairs"][i]
        img_path = os.path.join(img_root, f"new_{qa_pair['image_id'][0]}.jpg")
        if i == 36:
            qa_pair = dataset_base["qa_pairs"][i+1]
            img_path = os.path.join(img_root, f"new_{qa_pair['image_id'][0]}.jpg")
            print(qa_pair["question_idx"])
            print(qa_pair["question_text"])
            response = client.images.generate(
                model="gpt-image-1",
                prompt=qa_pair["question_text"],
                size="1024x1536",
                quality="low",
                n=1
            )

            img_bytes = base64.b64decode(response.data[0].b64_json)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img.save(img_path, format="JPEG", quality=85, optimize=True)
        content = {
            "question_idx": idx,
            "question_desc": qa_pair["question_text"],
            "question_desc2image": img_path,
            "question_image": qa_pair["question_image"],
            "image_count": 1,
            "image_id": qa_pair["image_id"],
            "image_path": qa_pair["image_path"],
        }
        new_qa_pairs.append(content)
        idx += 1
    dataset = {
        "dataset_name": "Standford40Actions_ImageImage2Image",
        "query_type": "2 images",
        "total_questions": len(new_qa_pairs),
        "qa_pairs": new_qa_pairs
    }
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    # asyncio.run(generate_descs(selected_images_file="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_16_experiments/results/label_split_basedon_query_nopatches/selected_images.json", output_path="/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_16_experiments/results/label_split_basedon_query_nopatches/desc.json"))
    # asyncio.run(test())
    test_new()