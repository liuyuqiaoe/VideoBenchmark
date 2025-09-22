from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torch
from PIL import Image
import base64
import io
from io import BytesIO
import json
import os
from tqdm import tqdm
from openai import OpenAI

prompt_templates = {
    "template1": (
    "Given the action label: '{action_label}', describe this image in detail. Focus on how the action '{action_label}' is being performed, "
    "Also mention relevant details "
    "about the people, objects, interactions, and environment."
    ),
    "template2": (
    "Given the action label: '{action_label}', describe this image in detail. Focus on how the action '{action_label}' is being performed, "
    ),
    "template3": (
    "Briefly describe this person doing {action_label} in this image, focusing on the action, the clothes and appearance of the person."
    ),
    "template4": (
    "Describe the action '{action_label}' in this image. Also briefly mention about the people and the environment."
    ),
    "template5": (
        "Generate 5 diverse phrases that describe the image with action label {action_label} but focus on different semantic aspects: objects, attributes, actions, scene, style.\n"
        "Output format: - Objects: [phrase]\n - Attributes: [phrase]\n - Actions: [phrase]\n - Scene: [phrase]\n - Style: [phrase]"
    )
}

class LLaVAOneVisionQwen2Generator:
    def __init__(self, model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf"):
        
        print(f"Loading LLaVA-OneVision-Qwen2 model: {model_name}")
    
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 
        )

        print("LLaVA-OneVision-Qwen2 model loaded successfully!")
        
    def generate_description(self, prompt, images):
        if isinstance(images, Image.Image):
            images = [images]

        content = [{"type": "text", "text": prompt}]
        for i, img in enumerate(images):
            content.append({
                "type": "image"
            })
        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print(prompt)
        inputs = self.processor(images=images, text=prompt, return_tensors='pt').to("cuda", torch.float16)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=254,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        del inputs
        
        response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip()
        
        del outputs
        torch.cuda.empty_cache()
        return response

    def _image_to_base64(self, image: Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

if __name__ == "__main__":
    generator = LLaVAOneVisionQwen2Generator()
    selected_imgs_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/selected_images.json"
    dataset_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/Stanford40Action_ImageLabel10Description10template5.json"
    with open(selected_imgs_path, "r") as f:
        selected_imgs = json.load(f)
    
    dataset = None
    if os.path.isfile(dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    
    if not dataset:
        dataset = {
            "dataset_name": "Stanford40Action_ImageLabelDescripion10template5",
            "description": "Descriptions are generated with both action label and specific image as inputs. 10 images are randomly selected from each action label."
        }
        offset = 0
        dataset["total_questions"] = offset
        qa_pairs = []
    else:
        offset = dataset.get( "total_questions", 0)
        dataset["total_questions"] = offset
        qa_pairs = dataset.get("qa_pairs", [])
        dataset["qa_pairs"] = qa_pairs

    count = 0
    # for action_label, imgs in selected_imgs.items():
    #     prompt = prompt_templates["template5"].format(action_label=action_label)
    #     for img_path in tqdm(imgs, unit="image"):
    #         count += 1
    #         if count <= offset:
    #             continue
    #         img = Image.open(img_path).convert("RGB")
    #         description = generator.generate_description(prompt, img)
    #         qa_pair = {
    #             "question": description,
    #             "answer": os.path.basename(img_path).split(".")[0].strip(),
    #             "action": action_label,
    #             "qestion_idx": offset
    #         }
    #         offset += 1
    #         qa_pairs.append(qa_pair)

    #     dataset["total_questions"] = offset
    #     dataset["qa_pairs"] = qa_pairs
    #     with open(dataset_path, "w") as f:
    #         json.dump(dataset, f, indent=2)

    for action_label, imgs in selected_imgs.items():
        prompt = prompt_templates["template5"].format(action_label=action_label)
        for img_path in tqdm(imgs, unit="image"):
            count += 1
            if count <= offset:
                continue
            img = Image.open(img_path).convert("RGB")
            for i in range(10):
                description = generator.generate_description(prompt, img)
                qa_pair = {
                    "question": description,
                    "answer": os.path.basename(img_path).split(".")[0].strip(),
                    "action": action_label,
                    "offset": offset,
                    "question_idx": offset * 10 + i
                }
                qa_pairs.append(qa_pair)
            offset += 1
        dataset["total_questions"] = offset
        dataset["qa_pairs"] = qa_pairs
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)





    # label_idx = 0
    # for action_label, imgs in selected_imgs.items():
    #     if label_idx >=1:
    #         break
    #     label_idx += 1
    #     prompt = prompt_templates["template5"].format(action_label=action_label)
    #     for img_idx, img_path in enumerate(imgs):
    #         if img_idx >1:
    #             break
    #         img = Image.open(img_path).convert("RGB")
    #         print(f"img path: {img_path}")
    #         for i in range(10):
    #             description = generator.generate_description(prompt, img)
    #             print(description)
            
    
       
        


    