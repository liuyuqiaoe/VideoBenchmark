import torch
import asyncio
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import networkx as nx
from tqdm import tqdm
import json

from video_retrieval.stanford40action.generators.gpt_generator import GPTGenerator, process_batch_text_embedding

class ResNet50Encoder():
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to("cuda")

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def encode_images_from_paths(self, image_paths, batch_size=20):
        img_embs_lst = []
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:min(i+batch_size, len(image_paths))]
            batch_tensors = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                img_tensor = self.transform(img)
                batch_tensors.append(img_tensor)
            batch_tensors = torch.stack(batch_tensors).to("cuda")
        
            with torch.no_grad():
                batch_embs = self.model(batch_tensors).view(batch_tensors.size(0), -1)
                if batch_embs.dim() == 3:
                    batch_embs.squeeze()
                elif batch_embs.dim() == 1:
                    batch_embs.unsqueeze(dim=0)
                img_embs_lst.append(batch_embs)
        img_embs = torch.vstack(img_embs_lst)

        return img_embs
    
def find_diverse_groups_mst(embeddings, cluster_labels, 
                           similarity_threshold=0.75,
                           min_group_size=2):
    results = {}
    
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
            
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) < min_group_size:
            continue
        
        cluster_embeddings = embeddings[cluster_indices]
        similarity_matrix = cosine_similarity(cluster_embeddings)
        # Create graph with ALL edges below threshold
        G = nx.Graph()
        for i in range(len(cluster_indices)):
            G.add_node(i)
        
        # Add edges for all pairs with similarity <= threshold
        for i in range(len(cluster_indices)):
            for j in range(i+1, len(cluster_indices)):
                similarity = similarity_matrix[i, j]
                if similarity <= similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
        
        print(f"Cluster {cluster_id}: {G.number_of_edges()} edges below threshold {similarity_threshold}")
        
        # Find connected components
        components = list(nx.connected_components(G))
        print(f"Components: {components}")
        
        # Filter by size
        diverse_groups = []
        for component in components:
            if len(component) >= min_group_size:
                group_indices = [cluster_indices[idx] for idx in component]
                diverse_groups.append(group_indices)
        
        results[cluster_id] = diverse_groups
    
    return results

def cluster_embeddings_kmeans(embeddings, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def get_embs_file(descs_file, output_path):
    with open(descs_file, "r") as f:
        data = json.load(f)
    descs = data["descs"]
    res = {}

    resnet50_encoder = ResNet50Encoder()
    gpt_generator = GPTGenerator()

    for action_label, desc_lst in tqdm(descs.items()):
        img_paths = []
        descs = []
        res[action_label] = []
        for desc in desc_lst:
            img_paths.append(desc[0])
            descs.append(desc[1])
        
        image_embedding = resnet50_encoder.encode_images_from_paths(img_paths).cpu().numpy().tolist()
        assert len(image_embedding) == len(img_paths)

        desc_embedding = asyncio.run(process_batch_text_embedding(gpt_generator, descs)) # list of length num_texts, each element is a list with length 1536
        assert len(desc_embedding) == len(img_paths)

        for i, desc in enumerate(desc_lst):
            content = {
                "image_name": os.path.basename(desc[0]),
                "description": desc[1],
                "image_embedding": image_embedding[i],
                "desc_embedding": desc_embedding[i]
            }
            res[action_label].append(content)
        
    with open(output_path, "w") as f:
        json.dump(res, f, indent=2)
        
if __name__ == "__main__":
    descs_file =  "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_16_experiments/results/label_split_basedon_query_nopatches/desc.json"
    output_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/experiments/9_16_experiments/results/label_split_basedon_query_nopatches/embs_file.json"
    get_embs_file(descs_file, output_path)

    # print("="*60, "Testing image_selector Starts", "="*60)
    
    # print("1. Loading ResNet50Encoder...")
    # resnet50_encoder = ResNet50Encoder()
    # print("\n")

    # print("2. Testing Encoding Image...")
    # images = [
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_156.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_232.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_216.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_273.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_092.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_139.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_002.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_106.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_257.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_202.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_042.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_205.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_125.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_150.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_117.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_194.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_097.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_189.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_040.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_024.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_160.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_096.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_181.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_188.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_027.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_119.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_148.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_261.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_085.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_050.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_037.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_064.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_054.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_191.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_012.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_154.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_032.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_052.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_026.jpg",
    #     "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/Stanford40Actions/StanfordActionDataset/train/fishing/fishing_061.jpg"
    # ]
    # image_embedding = resnet50_encoder.encode_images_from_paths(images)
    # print(f"image embedding size: {image_embedding.size()}") # torch.Size([40, 2048])
    # print("\n")

    # print("3. Loading GPTGenerator...")
    # gpt_generator = GPTGenerator()
    # print("\n")

    # print("4. Testing Get Text Embedding...")
    # texts = [
    #     "Person climbing an indoor rock wall, wearing an orange jacket with stripes and grey shorts, positioned side-on to the viewer. The climber grips handholds with both hands above head level while balancing on footholds, demonstrating physical engagement and focus. The setting appears to be an artificial climbing facility with color-coded holds on a vertical surface, suggesting a sporting or recreational context. The individual's build is athletic, with concentration on upper body and leg muscle usage in the climbing motion. The angle is slightly overhead, emphasizing the ascent and height.",
    #     "Image: Person indoor rock climbing on climbing wall, smiling at camera.\nDescription: Young male engaged in indoor rock climbing, hanging on a handhold with his right hand and clinging onto a foothold with the left foot on a vertical climbing wall. The view is from above and slightly to the side, capturing a top-down perspective with emphasis on his cheerful expression directed toward the camera. He is wearing a striped short-sleeve t-shirt and dark pants, with a climbing harness visible, suggesting a casual and sporty style appropriate for an indoor climbing gym setting. The individual has a slim build and is characterized by light brown, slightly messy hair. The wall is speckled with various colored hand and footholds, indicating a complex climbing route.",
    #     "Person rock climbing on indoor climbing wall, mid-action ascending motion, left hand reaching upward for grip, right hand holding firm on rock hold, body angled towards wall, right leg bent and pressing against wall, wearing gray t-shirt and black pants, athletic build, medium brown hair in ponytail, indoor setting with red climbing mats below, slight upward side view, other climbers visible in background, ambient gym lighting.",
    #     "Person engaged in indoor rock climbing, positioned on a vertical climbing wall. The person is shown from the back, mid-climb, with feet spread wide apart and both arms reaching upwards to grasp holds. The wall is green with scattered colorful climbing holds and large red circular patterns. The climber wears a white tank top, beige shorts, and white climbing shoes, with a safety harness strapped around the waist and thighs. The person has a slim build, brown hair tied back, and the scene suggests an indoor sports facility with focused lighting.",
    #     "A young girl is rock climbing on an artificial climbing wall, depicted from a side profile. She is wearing a safety helmet and harness, securing her for the climb. Her left hand is gripping a hold slightly above her, while her right hand reaches out to another grip. The left foot is planted on a lower hold, providing stability, and the right foot is lifted, preparing to push upward. She is dressed in a striped short-sleeve shirt and fitted black pants, with white socks and sneakers. Her hair is in a braid, cascading over her shoulder. The setting is likely an indoor climbing facility or an outdoor climbing wall, with clear wooden panels and climbing grips visible. The scene captures a moment of active engagement in the sport, emphasizing the intensity and focus required in rock climbing.",
    #     "Person rock climbing on indoor climbing wall, viewed from above. The climber is a woman with long dark hair in a ponytail, wearing a grey t-shirt and multicolored shorts. Her right arm is extended upward reaching for a handhold, while her left arm holds onto a lower grip. Both legs are spread wide, demonstrating a dynamic climbing pose with her left leg bent and right leg extended. The climber wears climbing shoes with heel marks visible. Various climbing holds and route markers in bright colors are scattered across the wall. The setting is an indoor climbing gym, with the climber appearing fit and focused on her next move."
    # ]
    
    # text_embedding = asyncio.run(process_batch_text_embedding(gpt_generator, texts)) # list of length num_texts, each element is a list with length 1536

    # print(f"text embedding len: {len(text_embedding)}") 
    # print("\n")

    # print("5. Testing cluster_embeddings_kmeans...")
    # image_embedding = image_embedding.cpu().numpy() if isinstance(image_embedding, torch.Tensor) else image_embedding
    # cluster_labels = cluster_embeddings_kmeans(image_embedding, 3)
    # print(cluster_labels)
    # breakpoint()
    # print("\n")

    # print("6. Testing find_diverse_groups_mst...")
    # cluster_labels = [1] * 15 + [2] * 5 + [3] + [4] * 12 + [5] * 7
    # cluster_labels = np.array(cluster_labels)
    # image_embedding = image_embedding.cpu().numpy()
    # groups = find_diverse_groups_mst(image_embedding, cluster_labels, 0.65, 2)
    # print(groups)
    # print("\n")

    # print("="*60, "Testing LanceDBVideoRetriever Ends", "="*60)

