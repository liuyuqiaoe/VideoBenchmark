import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from PIL import Image
import cv2
import os
from pathlib import Path
import json
import pickle
import time
from datetime import datetime
import lancedb
from lancedb.pydantic import Vector, LanceModel
import pandas as pd
from tqdm import tqdm


class E5Vschema(LanceModel):
    id: str
    video_path: str
    video_name: str
    action_category: str
    video_index: int
    frame_index: int
    total_frames: int
    embedding: Vector(4096) # frame-level embeddings
    frame_description: str
    batch_file: str
    batch_idx: int
    # Embedding metadata
    encoder_type: str  # "e5v"
    embedding_dim: int  # 4096
    num_frames: int  # Number of frames
    embedding_shape: List[int]  # (1, 4096) for each frame
    video_embedding_shape: List[int]  # (num_frames, 4096) for the full video embedding

class InternVideo2Schema(LanceModel):
    id: str
    video_path: str
    video_name: str
    action_category: str
    video_index: int
    frame_index: int
    total_frames: int
    embedding: Vector(1024) # frame-level embeddings
    frame_description: str
    batch_file: str
    batch_idx: int
    # Embedding metadata
    encoder_type: str  # "internvideo2"
    embedding_dim: int  # 1024
    num_frames: int  # Number of frames
    embedding_shape: List[int]  # (1, 1024) for each frame
    video_embedding_shape: List[int]  # (num_frames, 1024) for the full video embedding

class LLaVAQwenSchema(LanceModel):
    id: str
    video_path: str
    video_name: str
    action_category: str
    video_index: int
    frame_index: int
    total_frames: int
    embedding: Vector(3584) # frame-level embeddings
    frame_description: str
    batch_file: str
    batch_idx: int
    # Embedding metadata
    encoder_type: str  # "internvideo2"
    embedding_dim: int  # 1024
    num_frames: int  # Number of frames
    embedding_shape: List[int]  # (1, 1024) for each frame
    video_embedding_shape: List[int]  # (num_frames, 1024) for the full video embedding

class VLM2VecSchema(LanceModel):
    id: str
    video_path: str
    video_name: str
    action_category: str
    video_index: int
    frame_index: int
    total_frames: int
    embedding: Vector(3584) # frame-level embeddings
    frame_description: str
    batch_file: str
    batch_idx: int
    # Embedding metadata
    encoder_type: str  # "internvideo2"
    embedding_dim: int  # 1024
    num_frames: int  # Number of frames
    embedding_shape: List[int]  # (1, 1024) for each frame
    video_embedding_shape: List[int]  # (num_frames, 1024) for the full video embedding

def safe_tensor_to_numpy(tensor):
    if tensor is None:
        return None

    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    
    return tensor.cpu().numpy()

def colbert_maxsim_max(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None):
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
            similarities.append(max_sim_per_token.max(dim=0).values)  # [video_num]
            start_idx = end_idx
        return torch.stack(similarities)  # [num_queries, video_num]
    else:
        max_sim_per_token = s.max(dim=2).values  # [total_query_tokens, video_num]
        return max_sim_per_token.max(dim=0).values  # [video_num]

def colbert_maxsim_mean(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None):
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

def colbert_maxsim_mean_weighted(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None, patch_weights: torch.Tensor = None):
    query_norm = F.normalize(query_emb, p=2, dim=-1)  # [total_query_tokens, embedding_dim]
    video_norm = F.normalize(video_emb, p=2, dim=-1)  # [total_video_frames, embedding_dim]
    
    s = torch.matmul(query_norm, video_norm.T)  # [total_query_tokens, total_video_frames]
    
    # [total_query_tokens, total_video_frames] -> [total_query_tokens, video_num, frame_num]
    if video_num is not None:
        s = s.view(query_emb.shape[0], video_num, -1) 
    
    if patch_weights is not None:
        patch_weights = patch_weights.to(s.device)
        # Reshape weights to match the s dimension: [1, 1, frame_num]
        weights = patch_weights.view(1, 1, -1)
        s = s * weights
    
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

def cosine_mean(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None):
    """
    query_emb: [total_query_tokens, embedding_dim]
    video_emb: [total_video_frames, embedding_dim]
    query_token_nums: List of token counts for each query (e.g., [5, 9] for 2 queries)
    video_num: Number of videos (for reshaping)
    """
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
            similarities.append(query_sim.mean(dim=(0, 2)))  # [video_num]
            start_idx = end_idx
        return torch.stack(similarities)  # [num_queries, video_num]
    else:
        return s.mean(dim=(0, 2))  # [video_num]

def cosine_max_mean(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None) -> torch.Tensor:
    query_norm = F.normalize(query_emb, p=2, dim=-1)  # [total_query_tokens, embedding_dim]
    video_norm = F.normalize(video_emb, p=2, dim=-1)  # [total_video_frames, embedding_dim]
    
    s = torch.matmul(query_norm, video_norm.T)  # [total_query_tokens, total_video_frames]
    
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

def dot_mean(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None) -> torch.Tensor:
    s = torch.matmul(query_emb, video_emb.T)  # [total_query_tokens, total_video_frames]
    
    # [total_query_tokens, total_video_frames] -> [total_query_tokens, video_num, frame_num]
    if video_num is not None:
        s = s.view(query_emb.shape[0], video_num, -1)  

    if query_token_nums is not None:
        similarities = []
        start_idx = 0
        for token_num in query_token_nums:
            end_idx = start_idx + token_num
            query_sim = s[start_idx:end_idx]  # [token_num, video_num, frame_num]
            similarities.append(query_sim.mean(dim=(0, 2)))  # [video_num]
            start_idx = end_idx
        return torch.stack(similarities)  # [num_queries, video_num]
    else:
        return s.mean(dim=(0, 2))  # [video_num]

def dot_max(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None) -> torch.Tensor:
    s = torch.matmul(query_emb, video_emb.T)  # [total_query_tokens, total_video_frames]
    
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

def euclidean_similarity(query_emb: torch.Tensor, video_emb: torch.Tensor, query_token_nums: List[int] = None, video_num: int = None) -> torch.Tensor:
    distances = torch.cdist(query_emb, video_emb, p=2)  # [total_query_tokens, total_video_frames]
    
    # [total_query_tokens, total_video_frames] -> [total_query_tokens, video_num, frame_num]
    if video_num is not None:
        distances = distances.view(query_emb.shape[0], video_num, -1)  
    
    if query_token_nums is not None:
        similarities = []
        start_idx = 0
        for token_num in query_token_nums:
            end_idx = start_idx + token_num
            query_dist = distances[start_idx:end_idx]  # [token_num, video_num, frame_num]
            similarities.append(-query_dist.mean(dim=(0, 2)))  # [video_num]
            start_idx = end_idx
        return torch.stack(similarities)  # [num_queries, video_num]
    else:
        return -distances.mean(dim=(0, 2))  # [video_num]

SIMILARITY_FUNCTIONS = {
    "cosine_mean": cosine_mean,
    "cosine_max_mean": cosine_max_mean,
    "dot_mean": dot_mean,
    "dot_max": dot_max,
    "euclidean": euclidean_similarity,
    "colbert_maxsim_max": colbert_maxsim_max,
    "colbert_maxsim_mean": colbert_maxsim_mean,
    "colbert_maxsim_mean_weighted": colbert_maxsim_mean_weighted
}

ENCODER_CONFIGS = {
    "e5v": {
        "schema_class": E5Vschema,
        "embedding_dim": 4096,
        "similarity_methods": ["cosine_mean", "cosine_max_mean", "dot_mean", "dot_max", "euclidean", "colbert_maxsim_max", "colbert_maxsim_mean"],
        "default_similarity": "cosine_max_mean"
    },
    "internvideo2": {
        "schema_class": InternVideo2Schema,
        "embedding_dim": 1024,
        "similarity_methods": ["cosine_mean", "cosine_max_mean", "dot_mean", "dot_max", "euclidean", "colbert_maxsim", "colbert_maxsim_max", "colbert_maxsim_mean"],
        "default_similarity": "cosine_max_mean"
    },
    "llavaqwen": {
        "schema_class": LLaVAQwenSchema,
        "embedding_dim": 3584,
        "similarity_methods": ["cosine_mean", "cosine_max_mean", "dot_mean", "dot_max", "euclidean", "colbert_maxsim", "colbert_maxsim_max", "colbert_maxsim_mean", "colbert_maxsim_mean_weighted"],
        "default_similarity": "cosine_max_mean"
    },
    "vlm2vec": {
        "schema_class": VLM2VecSchema,
        "embedding_dim": 3584,
        "similarity_methods": ["cosine_mean", "cosine_max_mean", "dot_mean", "dot_max", "euclidean", "colbert_maxsim", "colbert_maxsim_max", "colbert_maxsim_mean", "colbert_maxsim_mean_weighted"],
        "default_similarity": "cosine_max_mean"
    }
}

def get_encoder_config(encoder_name: str):
    return ENCODER_CONFIGS.get(encoder_name, ENCODER_CONFIGS["e5v"])

def detect_encoder_type(encoder):
    if encoder is None:
        return "e5v" 
    
    class_name = encoder.__class__.__name__.lower()
    if "internvideo2" in class_name:
        return "internvideo2"
    elif "e5" in class_name:
        return "e5v"
    elif "llavaqwen" in class_name:
        return "llavaqwen"
    elif "vlm2vec" in class_name:
        return "vlm2vec"

    return "e5v"  

class LanceDBVideoIndex:
    def __init__(self, table_name = "video_embeddings", 
                 db_path = "./lancedb", max_frames_per_video = 8, encoder_type = "e5v"):
        self.table_name = table_name
        self.db_path = db_path
        self.max_frames_per_video = max_frames_per_video
        self.encoder_type = encoder_type
        
        self.encoder_config = get_encoder_config(encoder_type)
        
        # add patch weights
        self.custom_patch_weights = None
        
        self.db = lancedb.connect(db_path)
        
        try:
            self.table = self.db.open_table(table_name)
            print(f"Loaded existing table: {table_name}")
        except:
            schema_class = self.encoder_config["schema_class"]
            self.table = self.db.create_table(table_name, schema=schema_class)
            print(f"Created new {encoder_type} table: {table_name}")
        
        print(f"LanceDB index initialized with {max_frames_per_video} frames per video (encoder: {encoder_type})")
    
    def _get_patch_weights(self, num_frames, similarity_type):
        if similarity_type != "colbert_maxsim_mean_weighted":
            return None
        
        if self.encoder_type != "llavaqwen":
            return None
        
        if self.custom_patch_weights is not None:
            return self.custom_patch_weights
        else:
            return torch.ones(num_frames, dtype=torch.float32)
    
    def set_patch_weights(self, patch_weights):
        if isinstance(patch_weights, list):
            patch_weights = torch.tensor(patch_weights, dtype=torch.float32)
        self.custom_patch_weights = patch_weights.clone()
        print(f"Set custom patch weights for {patch_weights.shape[0]} embeddings: {patch_weights.tolist()}")
    
    def get_schema(self, table_name = "video_embeddings"):
        if table_name == self.table_name:
            return self.table.schema
        table = self.db.open_table(table_name)
        return table.schema
    
    def add_columns(self, update_data, table_name ="video_embeddings"):
        if not update_data:
            return

        if table_name == self.table_name:
            table = self.table
        table = self.db.open_table(table_name)
        table.add_columns(update_data)
            
    def add_videos(self, video_frame_embeddings: List[np.ndarray], video_paths, metadata = None):
        if len(video_frame_embeddings) != len(video_paths):
            raise ValueError("Number of video embeddings must match number of video paths")
        
        records = []
        
        for video_idx, (frame_embeddings, video_path) in enumerate(zip(video_frame_embeddings, video_paths)):
            video_name = os.path.basename(video_path)
            action_category = os.path.basename(os.path.dirname(video_path)) 
            total_frames = len(frame_embeddings)
            
            frame_embedding_shape = [1, self.encoder_config["embedding_dim"]]  # [1, 4096] for E5V frames, [1, 1024] for internvideo2
            video_embedding_shape = [total_frames, self.encoder_config["embedding_dim"]] # [8, 4096] for E5V frames, [8, 1024] for internvideo 2
            
            for frame_idx, frame_embedding in enumerate(frame_embeddings):
                frame_id = f"video_{video_idx}_frame_{frame_idx}"
                
                record = {
                    "id": frame_id,
                    "video_path": video_path,
                    "video_name": video_name,
                    "action_category": action_category,
                    "video_index": video_idx,
                    "frame_index": frame_idx, # frame index inside video frames
                    "total_frames": total_frames,
                    "embedding": frame_embedding.tolist(), 
                    "frame_description": f"Frame {frame_idx} of {video_name}",
                    "batch_file": metadata[video_idx]["batch_file"],
                    "batch_idx": metadata[video_idx]["batch_idx"],
                    "encoder_type": self.encoder_type,
                    "embedding_dim": self.encoder_config["embedding_dim"],
                    "num_frames": total_frames, # same as total_frames
                    "embedding_shape": frame_embedding_shape,  
                    "video_embedding_shape": video_embedding_shape 
                }
                
                # if metadata and video_idx < len(metadata):
                #     record.update(metadata[video_idx])
                
                records.append(record)
    
        df = pd.DataFrame(records)
        self.table.add(df)
        
        print(f"Added {len(video_paths)} videos with {len(records)} independent frame records to LanceDB index")
    
    def add_images(self, video_frame_embeddings: List[np.ndarray], video_paths, metadata = None):
        if len(video_frame_embeddings) != len(video_paths):
            raise ValueError("Number of image embeddings must match number of image paths")
        
        records = []
        
        for video_idx, (frame_embeddings, video_path) in enumerate(zip(video_frame_embeddings, video_paths)):
            video_name = video_path
            action_category = "None"
            total_frames = len(frame_embeddings)
            
            frame_embedding_shape = [1, self.encoder_config["embedding_dim"]]  # [1, 4096] for E5V frames, [1, 1024] for internvideo2
            video_embedding_shape = [total_frames, self.encoder_config["embedding_dim"]] # [8, 4096] for E5V frames, [8, 1024] for internvideo 2
            
            for frame_idx, frame_embedding in enumerate(frame_embeddings):
                frame_id = f"image_{video_idx}_frame_{frame_idx}"
                
                record = {
                    "id": frame_id,
                    "video_path": video_path,
                    "video_name": video_name,
                    "action_category": action_category,
                    "video_index": video_idx,
                    "frame_index": frame_idx, # frame index inside video frames
                    "total_frames": total_frames,
                    "embedding": frame_embedding.tolist(), 
                    "frame_description": f"Frame {frame_idx} of {video_name}",
                    "batch_file": metadata[video_idx]["batch_file"],
                    "batch_idx": metadata[video_idx]["batch_idx"],
                    "encoder_type": self.encoder_type,
                    "embedding_dim": self.encoder_config["embedding_dim"],
                    "num_frames": total_frames, # same as total_frames
                    "embedding_shape": frame_embedding_shape,  
                    "video_embedding_shape": video_embedding_shape 
                }
                
                # if metadata and video_idx < len(metadata):
                #     record.update(metadata[video_idx])
                
                records.append(record)
        
        df = pd.DataFrame(records)
        self.table.add(df)
        
        print(f"Added {len(video_paths)} videos with {len(records)} independent frame records to LanceDB index")
    
    def _calculate_video_similarities(self, query_embedding: np.ndarray, similarity_type: str = None, query_token_nums: List[int] = None) -> Dict[str, Dict]:
        """
        query_embedding: [total_tokens_num, embedding_dim]
        """
        if similarity_type is None:
            similarity_type = self.encoder_config["default_similarity"] # cosine mean
        
        if similarity_type not in self.encoder_config["similarity_methods"]:
            raise ValueError(f"Similarity method '{similarity_type}' not supported for encoder '{self.encoder_type}'. Available methods: {self.encoder_config['similarity_methods']}")
        query_emb = torch.tensor(query_embedding, dtype=torch.float32)
        
        query_emb_flat = query_emb
        
        results = self.table.to_pandas()
        
        # Group by video_path and process each group
        grouped = results.groupby('video_path')
        
        video_frames = {}
        video_paths = []
        video_indices = []
        V_embs_list = []
        
        num_frames = results.iloc[0]['num_frames']  # All videos have same frame count
        embedding_dim = len(results.iloc[0]['embedding'])
        
        for video_path, group in grouped:
            video_index = group.iloc[0]['video_index']
            
            # Sort by frame_index to ensure proper ordering (raw image embedding at largest frame index)
            group_sorted = group.sort_values('frame_index')
            frame_embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in group_sorted['embedding'].tolist()]
            frame_indices = group_sorted['frame_index'].tolist()
            
            actual_frames = len(frame_embeddings)
            if actual_frames < num_frames:
                # Padding
                last_frame = frame_embeddings[-1]
                padding_frames = [last_frame] * (num_frames - actual_frames)
                frame_embeddings.extend(padding_frames)
                
                last_frame_index = frame_indices[-1]
                padding_indices = [last_frame_index] * (num_frames - actual_frames)
                frame_indices.extend(padding_indices)
            
            video_embedding = torch.stack(frame_embeddings)  # [num_frames, embedding_dim]
            
            video_frames[video_path] = {
                'video_index': video_index,
                'frames': frame_embeddings,
                'frame_indices': frame_indices,
                'emb_index': len(V_embs_list)
            }
            
            video_paths.append(video_path)
            video_indices.append(video_index)
            V_embs_list.append(video_embedding)
        
        V_embs = torch.stack(V_embs_list)  # [video_num, num_frames, embedding_dim]
        
        patch_weights = self._get_patch_weights(num_frames, similarity_type)
        
        # TODO: frame mask?
        # [num_queries, video_num] or [video_num] (for single query)
        similarities = self._calculate_batched_similarity(query_emb_flat, V_embs, None, similarity_type, query_token_nums, patch_weights)

        video_similarities = {}
        for video_path in video_paths:
            emb_index = video_frames[video_path]['emb_index']  
            
            if similarities.dim() == 1:
                # Single query: [video_num]]
                video_sim = similarities[emb_index]
                query_similarities = [video_sim.item()]  
            else:
                # Multiple queries: [num_queries, video_num]
                video_sim = similarities[:, emb_index]  # [num_queries]
                query_similarities = video_sim.tolist()  
            
            video_similarities[video_path] = {
                'video_index': video_frames[video_path]['video_index'],
                'frame_indices': video_frames[video_path]['frame_indices'],
                'query_similarities': query_similarities  # [sim] for single query, [sim1, sim2, ...] for multi-query
            }
        
        return video_similarities
    
    def search(self, query_embedding: np.ndarray, queries, query_token_nums: List[int] = None, top_k: Union[int, List[int]] = 5, where_clause = [], return_all = False, similarity_type: str = None, encoder_type: str = None, encoder_config: dict = None):
        """
        query_embedding: [total_tokens_num, embedding_dim]
        queries: List of original query texts
        query_token_nums: List of token counts for each query (None for single query)
        """
        if similarity_type is None and encoder_config is not None:
            similarity_type = encoder_config["default_similarity"]
        
        if encoder_config is not None and similarity_type not in encoder_config["similarity_methods"]:
            raise ValueError(f"Similarity method '{similarity_type}' not supported for encoder '{encoder_type}'. Available methods: {encoder_config['similarity_methods']}")
        
        num_queries = 1 if query_token_nums is None else len(query_token_nums)
        
        start_time = time.time()

        if isinstance(top_k, int):
            top_k_list = [top_k] * num_queries
        else:
            if len(top_k) != num_queries:
                raise ValueError(f"Length of top_k list ({len(top_k)}) must match number of queries ({num_queries})")
            top_k_list = top_k

        results = self._search_with_similarity(query_embedding, query_token_nums, top_k_list, where_clause, return_all, similarity_type)
        # results = {
        #     query_idx: [
        #         (video_path, similarity, video_index, rank),
        #         ...
        #     ],...
        # }
        search_time = time.time() - start_time
        
        # Format results with metadata
        query_results = {}
        
        for query_idx, query_results_list in results.items():
            formatted_results = []
       
            for video_path, similarity, video_index, rank in query_results_list:
                metadata = {
                    "video_path": video_path,
                    "video_name": os.path.basename(video_path).rsplit('.', 1)[0],
                    "action_category": os.path.basename(os.path.dirname(video_path)),
                    "similarity": similarity,
                    "video_index": video_index,
                    "search_time": search_time,
                    "encoder_type": encoder_type,
                    "embedding_dim": encoder_config["embedding_dim"] if encoder_config else None,
                    "similarity_type": similarity_type,
                    "search_mode": "accurate",
                    "query_embedding_shape": query_embedding.shape,
                    "video_embedding_shape": (8, encoder_config["embedding_dim"]) if encoder_type == "internvideo2" and encoder_config else (1, encoder_config["embedding_dim"]) if encoder_config else None,
                    "query_text": queries[query_idx] if query_idx < len(queries) else f"query_{query_idx}",
                    "query_index": query_idx
                }
                formatted_results.append((video_path, similarity, rank, metadata))
            query_results[query_idx] = formatted_results
     
            # query_results = {
            #     query_idx: [
            #         (video_path, similarity, metadata),
            #         ...
            #     ],
            #     ...
            # }
        
        return query_results
    
    def search_clean(self, query_embedding: np.ndarray, queries = None, query_token_nums: List[int] = None, top_k: Union[int, List[int]] = 5, where_clause: str = " ", return_all = False, similarity_type: str = None, encoder_type: str = None, encoder_config: dict = None, return_gt = []):
        if similarity_type is None and encoder_config is not None:
            similarity_type = encoder_config["default_similarity"]
        
        if encoder_config is not None and similarity_type not in encoder_config["similarity_methods"]:
            raise ValueError(f"Similarity method '{similarity_type}' not supported for encoder '{encoder_type}'. Available methods: {encoder_config['similarity_methods']}")
        
        num_queries = 1 if query_token_nums is None else len(query_token_nums)
        
        start_time = time.time()

        if isinstance(top_k, int):
            top_k_list = [top_k] * num_queries
        else:
            if len(top_k) != num_queries:
                raise ValueError(f"Length of top_k list ({len(top_k)}) must match number of queries ({num_queries})")
            top_k_list = top_k
        
        results = self._search_with_similarity(query_embedding, query_token_nums, top_k_list, where_clause, return_all, similarity_type, return_gt)
        
        query_results = {}
        
        for query_idx, query_results_list in results.items():
            formatted_results = []
            for video_path, similarity, video_index, rank in query_results_list:
                formatted_results.append((video_path, similarity, rank))
            query_results[query_idx] = formatted_results
            
        return query_results


    def _search_with_similarity(self, query_embedding: np.ndarray, query_token_nums: List[int] = None, top_k: List[int] = None, where_clause = [], return_all = False, similarity_type = None, return_gt = []):
        if return_all:
            results = self.table.to_pandas()
        elif not where_clause:  
            results = self.table.to_pandas()
        else:
            # TODO:
            # results = self.table.where(where_clause[0]).to_pandas()
            results = self.table.to_pandas()
        
        # video_similarities = {
        #     video_path:{              
        #         'video_index': video_index,
        #         'frame_indices': frame_indices,
        #         'query_similarities': query_similarities  # [sim] for single query, [sim1, sim2, ...] for multi-query
        #     },...
        # }
        video_similarities = self._calculate_video_similarities(query_embedding, similarity_type, query_token_nums)
        
        # Filter results, where clause
        filtered_similarities = {}
        video_paths_in_results = set(results['video_path'].unique())
        
        for video_path, similarity_data in video_similarities.items():
            if video_path in video_paths_in_results:
                filtered_similarities[video_path] = similarity_data
        
        num_queries = 1 if query_token_nums is None else len(query_token_nums)
        
        query_results = {}
        for query_idx in range(num_queries):
            query_similarities = {}
            for video_path, similarity_data in filtered_similarities.items():
                query_similarities_list = similarity_data['query_similarities']
                query_sim = query_similarities_list[query_idx]
                
                query_similarities[video_path] = {
                    'similarity': query_sim,
                    'video_index': similarity_data['video_index']
                }
      
            sorted_videos = sorted(
                query_similarities.items(),
                key=lambda x: x[1]['similarity'],
                reverse=True
            )
            
            search_results = []
            if (not return_all) and (not return_gt):
                sorted_videos = sorted_videos[:top_k[query_idx]]
            
            # TODO: add return_gt and rank
            if not return_gt:
                for rank, (video_path, video_data) in enumerate(sorted_videos):
                    search_results.append((
                        video_path,
                        video_data['similarity'],
                        video_data['video_index'],
                        rank
                    ))
            else:
                return_gt_query = return_gt[query_idx]
                for rank, (video_path, video_data) in enumerate(sorted_videos):
                    video_id = os.path.basename(video_path)
                    if rank < top_k[query_idx]:
                        search_results.append((
                            video_path,
                            video_data['similarity'],
                            video_data['video_index'],
                            rank
                        ))
                        if video_id in return_gt_query:
                            return_gt_query.remove(video_id)
                    elif return_gt_query:
                        if video_id in return_gt_query:
                            search_results.append((
                                video_path,
                                video_data['similarity'],
                                video_data['video_index'],
                                rank
                            )) 
                            return_gt_query.remove(video_id)
                    else:
                        break
            
            query_results[query_idx] = search_results
            
        # query_results = {
        #     query_idx: [
        #         (video_path, similarity, video_index, rank)
        #     ],...
        # }
        
        return query_results
    
    def _calculate_batched_similarity(self, query_emb: torch.Tensor, V_embs: torch.Tensor, frame_masks = None, similarity_type = None, query_token_nums: List[int] = None, patch_weights: torch.Tensor = None):
        """
        query_emb: [total_query_tokens, embedding_dim]
        V_embs: [video_num, frame_num, embedding_dim]
        patch_weights: [frame_num] tensor with weights for each patch/raw image (for weighted similarity functions)
        """
        if similarity_type not in SIMILARITY_FUNCTIONS:
            raise ValueError(f"Unknown similarity type: {similarity_type}. Available types: {list(SIMILARITY_FUNCTIONS.keys())}")
    
        similarity_func = SIMILARITY_FUNCTIONS[similarity_type]
        
        video_num, frame_num, embedding_dim = V_embs.shape
        
        if frame_masks is not None:
            # TODO: not implemented now
            V_embs_flat = V_embs.view(-1, embedding_dim) 
            valid_frames = frame_masks.view(-1) 
            V_embs_valid = V_embs_flat[valid_frames] 
            if patch_weights is not None:
                similarities = similarity_func(query_emb, V_embs_valid, query_token_nums, video_num, patch_weights)
            else:
                similarities = similarity_func(query_emb, V_embs_valid, query_token_nums, video_num)
        else:
            # TODO: confirm the video order keeps unchanged during resahpe process
            # [video_num, frame_num, embedding_dim] -> [video_num * frame_num, embedding_dim]
            V_embs_flat = V_embs.view(-1, embedding_dim) 
            if patch_weights is not None:
                similarities = similarity_func(query_emb, V_embs_flat, query_token_nums, video_num, patch_weights)
            else:
                similarities = similarity_func(query_emb, V_embs_flat, query_token_nums, video_num)
        
        # The similaritiese shape:
        # Single query: [video_num]
        # Multiple queries: [num_queries, video_num]
        return similarities

    def get_stats(self):
        count = len(self.table)
        df = self.table.to_pandas()
        total_videos_count = df["video_path"].nunique()
        total_labels_count = df["action_category"].nunique()
        return {
            "total_videos_count": total_videos_count,
            "total_frames": count,
            "total_labels_count": total_labels_count,
            "table_name": self.table_name,
            "db_path": self.db_path,
            "max_frames_per_video": self.max_frames_per_video,
            "embedding_shape": df["embedding_shape"][0] if count > 0 else None,
            "video_embedding_shape": df["video_embedding_shape"][0] if count > 0 else None,
            "num_frames": df["num_frames"][0] if count > 0 else None
        }
    
    def clear_index(self):
        self.db.drop_table(self.table_name)
        self.table = self.db.create_table(self.table_name, schema=E5Vschema)
        print("Cleared LanceDB index")

class LanceDBVideoRetriever:
    def __init__(self, encoder = None, table_name = "video_embeddings", db_path = "./lancedb", max_frames_per_video = 8, encoder_type = None):
        self.encoder = encoder 
        
        if encoder is not None:
            detected_type = detect_encoder_type(encoder)
            if encoder_type is not None and encoder_type != detected_type:
                print(f"Warning: Specified encoder_type '{encoder_type}' differs from detected type '{detected_type}'")
                print(f"Using detected type: {detected_type}")
            encoder_type = detected_type
        elif encoder_type is None:
            encoder_type = "e5v" 
        
        self.encoder_type = encoder_type
        self.index = LanceDBVideoIndex(table_name, db_path, max_frames_per_video, encoder_type)
        self.max_frames_per_video = max_frames_per_video
        
        self.encoder_config = get_encoder_config(encoder_type)
        print(f"LanceDB video retrieval system initialized with {max_frames_per_video} frames per video (encoder: {encoder_type}).")
        print(f"Available similarity methods: {self.encoder_config['similarity_methods']}")

    def set_patch_weights(self, patch_weights):
        self.index.set_patch_weights(patch_weights)

    def set_encoder(self, encoder):
        self.encoder = encoder
        
        if encoder is not None:
            new_encoder_type = detect_encoder_type(encoder)
            
            if new_encoder_type != self.encoder_type:
                print(f"Warning: New encoder type '{new_encoder_type}' differs from current type '{self.encoder_type}'")
                print(f"New encoder config: {get_encoder_config(new_encoder_type)}")
                self.encoder_type = new_encoder_type
                self.encoder_config = get_encoder_config(new_encoder_type)

    def build_index(self, video_paths, batch_file="None", batch_idx=-1, force_rebuild_frames=False):
        print(f"Building index for {len(video_paths)} videos with {self.max_frames_per_video} frames each...")
        if not self.encoder:
            print("No encoder, please set an encoder, or using LanceDBVideoRetriever.build_index_no_encoder() instead.")
            return
        
        video_frame_embeddings = []
        successful_paths = []
        metadata = []
        
        for i, video_path in enumerate(tqdm(
            video_paths, 
            total=len(video_paths),
            desc="Encoding videos",
            unit="video"
        )):
            # tensor
            frame_embeddings = self.encoder.encode_video(video_path, force_rebuild_frames)
            
            if frame_embeddings is not None:
                # frame_embeddings: (num_frames, embedding_dim)
                video_frame_embeddings.append(safe_tensor_to_numpy(frame_embeddings)) # ndarray
                successful_paths.append(video_path)
                video_name = os.path.basename(video_path)
                action_category = os.path.basename(os.path.dirname(video_path))
                metadata.append({
                    "video_path": video_path,
                    "video_name": video_name,
                    "action_category": action_category,
                    "index": i, # index inside batch, can be removed?
                    "num_frames": frame_embeddings.shape[0], 
                    "batch_file": batch_file,
                    "batch_idx": batch_idx # index of batch, gotten by batch file
                })
               
            else:
                print(f"Failed to encode {video_path}")
        
        if video_frame_embeddings:
            self.index.add_videos(video_frame_embeddings, successful_paths, metadata)
        else:
            print("No videos were successfully encoded")
    
    def build_index_image(self, video_paths, batch_file="None", batch_idx=-1, force_rebuild_frames=False):
        print(f"Building index for {len(video_paths)} images...")
        if not self.encoder:
            print("No encoder, please set an encoder, or using LanceDBVideoRetriever.build_index_no_encoder() instead.")
            return
        img_embeddings = self.encoder.encode_image_from_paths(video_paths)
        video_frame_embeddings = []
        successful_paths = []
        metadata = []
        
        for i, video_path in enumerate(tqdm(
            video_paths, 
            total=len(video_paths),
            desc="Encoding images",
            unit="image"
        )):
            # tensor: [1, dim]
            # frame_embeddings = self.encoder.encode_video(video_path, force_rebuild_frames)
            frame_embeddings = img_embeddings[i]
            if frame_embeddings.dim() == 1:
                frame_embeddings = frame_embeddings.unsqueeze(0)
            if frame_embeddings is not None:
                # frame_embeddings: (num_frames=1, embedding_dim)
                video_frame_embeddings.append(safe_tensor_to_numpy(frame_embeddings)) # ndarray
                successful_paths.append(video_path)
                video_name = video_path
                action_category = "None"
                metadata.append({
                    "video_path": video_path,
                    "video_name": video_name,
                    "action_category": action_category,
                    "index": i, # index inside batch, can be removed?
                    "num_frames": frame_embeddings.shape[0], # 1
                    "batch_file": batch_file,
                    "batch_idx": batch_idx # index of batch, gotten by batch file
                })
               
            else:
                print(f"Failed to encode {video_path}")
        
        if video_frame_embeddings:
            self.index.add_images(video_frame_embeddings, successful_paths, metadata)
        else:
            print("No images were successfully encoded")

    def build_index_image_patches(self, video_paths, batch_file="None", batch_idx=-1, force_rebuild_frames=False):
        print(f"Building index for {len(video_paths)} images...")
        if not self.encoder:
            print("No encoder, please set an encoder, or using LanceDBVideoRetriever.build_index_no_encoder() instead.")
            return
        
        img_embeddings = self.encoder.encode_images_patches_from_paths(video_paths) # List[torch.Tensor(10, hidden_dim)]
        video_frame_embeddings = []
        successful_paths = []
        metadata = []
        
        for i, video_path in enumerate(tqdm(
            video_paths, 
            total=len(video_paths),
            desc="Encoding images",
            unit="image"
        )):
            frame_embeddings = img_embeddings[i]
            if frame_embeddings.dim() == 1:
                frame_embeddings = frame_embeddings.unsqueeze(0)
            if frame_embeddings is not None:
                # frame_embeddings: (num_patches, embedding_dim)
                video_frame_embeddings.append(safe_tensor_to_numpy(frame_embeddings)) 
                successful_paths.append(video_path)
                video_name = video_path
                action_category = os.path.basename(video_path).split(".")[0].split("_")[0].strip()
                metadata.append({
                    "video_path": video_path,
                    "video_name": video_name,
                    "action_category": action_category,
                    "index": i, # index inside batch, can be removed?
                    "num_frames": frame_embeddings.shape[0], # 10
                    "batch_file": batch_file,
                    "batch_idx": batch_idx # index of batch, gotten from batch file
                })
               
            else:
                print(f"Failed to encode {video_path}")
        
        if video_frame_embeddings:
            self.index.add_images(video_frame_embeddings, successful_paths, metadata)
        else:
            print("No images were successfully encoded")

    def build_index_no_encoder(self, video_paths, batch_file="None", batch_idx=-1, force_rebuild_frames = False, video_embeddings=None):
        pass

    def search_no_encoder():
        pass
        
    def search(self, queries, top_k = 5, where_clause = [], return_all = False, similarity_type = None):
        if not self.encoder:
            print("No encoder, please set an encoder.")
            return {}
        
        if isinstance(queries, str):
            queries = [queries]
        
        if isinstance(where_clause, str):
            where_clause = [where_clause]
        elif not where_clause :
            where_clause = []

        if not queries:
            print("No queries provided.")
            return {}
        
        # TODO: internvideo2 encode texts by batch
        if self.encoder_type in ["internvideo2", "llavaqwen"]:
            # encode each query separately and concatenate
            query_embeddings = []
            for query in queries:
                query_emb = self.encoder.encode_text(query)
                if query_emb.dim() == 3:
                    query_emb = query_emb.squeeze(0)
                if query_emb is not None:
                    query_embeddings.append(query_emb)

            if not query_embeddings:
                print("Failed to encode any queries")
                return {}

            query_embedding = torch.cat(query_embeddings, dim=0)  # torch.bfloat16 tensor: [query_num, num_tokens, embedding_dim]
            query_embedding = query_embedding.view(-1, query_embedding.size(-1)) # [total_num_tokens, embedding_dim]
            query_token_nums = [emb.shape[0] for emb in query_embeddings]  # List of token counts per query
        else:
            # encode all queries at once
            query_embedding = self.encoder.encode_text(queries) # torch.float16 tensor: [query_num, embedding_dim]
            if query_embedding is None:
                print("Failed to encode queries")
                return {}
            query_token_nums = [1] * len(queries)  
        
        if query_embedding is None:
            print("Failed to encode queries")
            return {}

        results = self.index.search( 
            safe_tensor_to_numpy(query_embedding),
            queries, 
            query_token_nums,
            top_k, 
            where_clause, 
            return_all, 
            similarity_type, 
            self.encoder_type, 
            self.encoder_config
        )
        # results = {
        #     query_idx: [
        #         (video_path, similarity, metadata, rank),
        #         ...
        #     ],
        #     ...
        # }
        return results

    def search_10descriptions_test(self, queries, top_k = 5, where_clause = " ", return_all = False, similarity_type = None):
        if not self.encoder:
            print("No encoder, please set an encoder.")
            return {}
        
        if isinstance(queries, str):
            queries = [queries]
        
        if not queries:
            print("No queries provided.")
            return {}
        
        query_token_nums = []

        if self.encoder_type in ["internvideo2", "llavaqwen"]:
            query_embeddings = []
            for query in queries:
                query_emb = []
                for sentence in query:
                    sentence_emb = self.encoder.encode_text(sentence)
                    if sentence_emb.dim() == 3:
                        sentence_emb = sentence_emb.squeeze(0)
                    elif sentence_emb.dim() == 1:
                        sentence_emb = sentence_emb.unsqueeze(0)

                    if sentence_emb is not None:
                        query_emb.append(sentence_emb)

                query_emb = torch.cat(query_emb, dim=0)
                query_token_nums.append(query_emb.shape[0])
                # query_emb = query_emb.unsqueeze(0)
                query_embeddings.append(query_emb)

        else:
            query_embeddings = []
            for query in queries:
                if hasattr(self.encoder, "encode_texts"):
                    query_emb = self.encoder.encode_texts(query) # torch.float16 tensor: [sentence_num, embedding_dim]
                else:
                    query_emb = self.encoder.encode_text(query)
                
                if query_emb is None:
                    print("Failed to encode queries")
                    return {}
                
                query_token_nums.append(query_emb.shape[0])
                # query_emb = query_emb.unsqueeze(0)
                query_embeddings.append(query_emb)

        if not query_embeddings:
                print("Failed to encode any queries")
                return {}
        
        query_embedding = torch.cat(query_embeddings, dim=0)  # torch.bfloat16 tensor: [query_num, num_tokens, embedding_dim]
        query_embedding = query_embedding.view(-1, query_embedding.size(-1)) # [total_num_tokens, embedding_dim]
  
        results = self.index.search( 
            safe_tensor_to_numpy(query_embedding),
            queries, 
            query_token_nums,
            top_k, 
            where_clause, 
            return_all, 
            similarity_type, 
            self.encoder_type, 
            self.encoder_config
        )
        
        return results


    def search_clean(self, queries, top_k = 5, where_clause = " ", return_all = False, similarity_type = None):
        if not self.encoder:
            print("No encoder, please set an encoder.")
            return {}
        
        if isinstance(queries, str):
            queries = [queries]
        
        if not queries:
            print("No queries provided.")
            return {}
        
        # TODO: internvideo2 encode texts by batch
        if self.encoder_type == "internvideo2":
            # encode each query separately and concatenate
            query_embeddings = []
            for query in queries:
                query_emb = self.encoder.encode_text(query)
                if query_emb.dim() == 3:
                    query_emb = query_emb.squeeze(0)
                if query_emb is not None:
                    query_embeddings.append(query_emb)

            if not query_embeddings:
                print("Failed to encode any queries")
                return {}

            query_embedding = torch.cat(query_embeddings, dim=0)  # torch.bfloat16 tensor: [query_num, num_tokens, embedding_dim]
            query_embedding = query_embedding.view(-1, query_embedding.size(-1)) # [total_num_tokens, embedding_dim]
            query_token_nums = [emb.shape[0] for emb in query_embeddings]  # List of token counts per query
        else:
            # encode all queries at once
            query_embedding = self.encoder.encode_text(queries) # torch.float16 tensor: [query_num, embedding_dim]
            if query_embedding is None:
                print("Failed to encode queries")
                return {}
            query_token_nums = [1] * len(queries)  
        
        if query_embedding is None:
            print("Failed to encode queries")
            return {}

        results = self.index.search_clean( 
            safe_tensor_to_numpy(query_embedding),
            queries, 
            query_token_nums,
            top_k, 
            where_clause, 
            return_all, 
            similarity_type, 
            self.encoder_type, 
            self.encoder_config
        )
        
        return results
    
    def search_image(self, images, top_k = 5, where_clause = " ", return_all = False, similarity_type = None, return_gt = []):
        if isinstance(images, Image.Image):
            images = [images]
        
        query_embedding = self.encoder.encode_images(images)
        query_token_nums = [1] * len(images)  
        
        if query_embedding is None:
            print("Failed to encode images")
            return {}

        results = self.index.search_clean( 
            safe_tensor_to_numpy(query_embedding),
            images, 
            query_token_nums,
            top_k, 
            where_clause, 
            return_all, 
            similarity_type, 
            self.encoder_type, 
            self.encoder_config,
            return_gt
        )
        
        return results
    
    def search_hybrid(self, texts, image_paths, top_k = 20, where_clause = " ", return_all = False, similarity_type = None, return_gt = []):
        assert len(texts) == len(image_paths)
        query_embedding = self.encoder.encode_image_text_pairs_from_paths(texts, image_paths)
        query_token_nums = [1] * len(image_paths)  
        
        if query_embedding is None:
            print("Failed to encode images")
            return {}

        results = self.index.search_clean(
            query_embedding=safe_tensor_to_numpy(query_embedding), 
            query_token_nums=query_token_nums, 
            top_k=top_k, 
            where_clause=where_clause, 
            return_all = return_all, 
            similarity_type=similarity_type, 
            encoder_type=self.encoder_type, 
            encoder_config=self.encoder_config, 
            return_gt=return_gt
            )
        
        return results

    def formated_results(self, results):
        if not isinstance(results, dict):
            print("Warning: Expected dictionary format for results")
            return
        
        dump_data = {}
        
        for query_idx, query_results in results.items():
            query_data = []
            for i, (video_path, similarity, rank, metadata) in enumerate(query_results):
                result_entry = {
                    "rank": rank,
                    "video_path": video_path,
                    "similarity": float(similarity), 
                    "metadata": {
                        "video_name": metadata["video_name"],
                        "action_category": metadata["action_category"],
                        "video_index": int(metadata["video_index"]),
                        "search_time": metadata["search_time"],
                        "encoder_type": metadata["encoder_type"],
                        "embedding_dim": int(metadata["embedding_dim"]),
                        "similarity_type": metadata["similarity_type"],
                        "search_mode": metadata["search_mode"],
                        "query_embedding_shape": [int(metadata["query_embedding_shape"][0]), int(metadata["query_embedding_shape"][1])],
                        "video_embedding_shape": [int(metadata["video_embedding_shape"][0]), int(metadata["video_embedding_shape"][1])],
                        "query_text": metadata.get("query_text", f"query_{query_idx}"),
                        "query_index": metadata.get("query_index", query_idx)
                    }
                }
                query_data.append(result_entry)
            
            dump_data[query_idx] = {
                "query_index": query_idx,
                "query_text": metadata.get("query_text", f"query_{query_idx}") if query_results else f"query_{query_idx}",
                "results": query_data,
                "total_results": len(query_results)
            }
        return dump_data

    def pure_results(self, results):
        pure_data = {}
        
        for query_idx, query_results in results.items():
            query_data = []
            for i, (video_path, similarity, rank, metadata) in enumerate(query_results):
                query_data.append(video_path)
            query_text = metadata.get("query_text", f"query_{query_idx}") if query_results else f"query_{query_idx}"
            pure_data[query_idx] = {
                "query_tex": query_text,
                "query_data": query_data
                }
                
        return pure_data

    def dump_results(self, results, output_file = "search_results.json"):
        if not isinstance(results, dict):
            print("Warning: Expected dictionary format for results")
            return
        
        dump_data = self.formated_results(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results dumped to: {output_file}")
        print(f"Total queries: {len(results)}")
        for query_idx, query_results in results.items():
            print(f"Query {query_idx}: {len(query_results)} results")
        
    
    def get_index_stats(self):
        return self.index.get_stats()
    
    def clear_index(self):
        self.index.clear_index()
