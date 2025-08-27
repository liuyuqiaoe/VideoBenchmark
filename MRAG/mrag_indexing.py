from video_retrieval import UCF101VideoPathCollector, LanceDBVideoRetriever
from video_retrieval.encoders.e5v_encoder import E5VVideoEncoder
import os
from tqdm import tqdm
import json
from typing import Optional
from pathlib import Path

def get_image_paths(root_dir: Optional[str] = None):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    if not os.path.exists(root_dir):
        print(f"Warning: Directory '{root_dir}' does not exist")
        return []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if Path(file_path).suffix.lower() in image_extensions:
                if os.path.isfile(file_path):
                    image_paths.append(file_path)
    return image_paths

def get_image_batch(dataset_dir, experiment_root):
    image_paths = get_image_paths(dataset_dir)
    print(f"lengh of image paths: {len(image_paths)}")
    collector = UCF101VideoPathCollector(dataset_dir, experiment_root)
    collector.video_paths = image_paths
    image_paths_file = collector.dump_results(file_name="mrag_image_paths")
    image_batch_file = collector.json_create_batches(batch_size=128, file_name="mrag_image_paths", experiment_root=experiment_root)
    metadata = collector.load_metadata(file_name="mrag_image_paths", experiment_root=experiment_root) 
    return image_batch_file

def indexing_images(start_idx, end_idx):
    dataset_dir = "/research/d7/fyp25/yqliu2/projects/ColBERT/data/image_data/image_corpus"
    experiment_root = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/MRAG/experiments"
    # image_batch_file = get_image_batch(dataset_dir=dataset_dir, experiment_root=experiment_root)
    # batch_file_path = UCF101VideoPathCollector.json_create_batches(batch_size=64, file_name=file_name)
    paths_in_batch = UCF101VideoPathCollector.get_batch_range(start_idx, end_idx, "mrag_image_paths", experiment_root)
    db_path_root = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/databases"
    file_name = "mrag_image_paths"
    progress_file = f"{file_name}_completed_e5v.json"
    progress_file = os.path.join(experiment_root, progress_file)

    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            completed = json.load(f)
    else:
        completed = []

    if not os.path.exists(db_path_root):
        os.makedirs(db_path_root)
        print(f"Create database directory {db_path_root}")

    db_path = os.path.join(db_path_root, "mrag_e5v_db")
    encoder = E5VVideoEncoder(max_frames_per_video=1)
    retriever = LanceDBVideoRetriever(encoder=encoder, db_path=db_path)
    batch_file = f"{file_name}_batches.json"
    for batch_idx in tqdm(range(start_idx, end_idx + 1), desc="Processing batches"):
        if batch_idx in completed:
            print(f"Skipping completed batch {batch_idx}")
            continue
            
        video_paths = paths_in_batch[batch_idx]
        print(f"Building index for batch {batch_idx} ({len(video_paths)} videos)")
        
        retriever.build_index_image(video_paths=video_paths, batch_file=batch_file, batch_idx=batch_idx)
        
        # Save progress
        completed.append(batch_idx)
        with open(progress_file, 'w') as f:
            json.dump(completed, f)
        
if __name__ == "__main__":
    indexing_images(131, 149)
