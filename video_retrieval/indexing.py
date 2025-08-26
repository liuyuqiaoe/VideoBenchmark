from video_retrieval import UCF101VideoPathCollector, LanceDBVideoRetriever
from video_retrieval.encoders.llava_qwen_encoder import LLaVAQwenEncoder
import os
from tqdm import tqdm
import json

def indexing_ufc_101(start_idx, end_idx):
    ufc101_path = "/research/d7/fyp25/yqliu2/projects/VideoBenchmark/UCF-101"
    collector =  UCF101VideoPathCollector(dataset_dir=ufc101_path)
    # video_paths = collector.collect_video_paths()
    # video_paths_file = collector.dump_results()
    # batch_file_path = UCF101VideoPathCollector.json_create_batches(batch_size=64)
    paths_in_batch = UCF101VideoPathCollector.get_batch_range(start_idx, end_idx)
    print(paths_in_batch)
    db_path_root = os.path.join(os.getcwd(), "databases")
    file_name = "ucf101_video_paths"
    root = os.path.join(os.getcwd(), "UCF101_list")
    progress_file = f"{file_name}_completed_llava.json"
    progress_file = os.path.join(root, progress_file)

    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            completed = json.load(f)
    else:
        completed = []

    if not os.path.exists(db_path_root):
        os.makedirs(db_path_root)
        print(f"Create database directory {db_path_root}")

    db_path = os.path.join(db_path_root, "ufc101_llava_db")
    encoder = LLaVAQwenEncoder()
    retriever = LanceDBVideoRetriever(encoder=encoder, db_path=db_path)
    batch_file = f"{file_name}_batches.json"
    for batch_idx in tqdm(range(start_idx, end_idx + 1), desc="Processing batches"):
        if batch_idx in completed:
            print(f"Skipping completed batch {batch_idx}")
            continue
            
        video_paths = paths_in_batch[batch_idx]
        print(f"Building index for batch {batch_idx} ({len(video_paths)} videos)")
        
        retriever.build_index(video_paths=video_paths, batch_file=batch_file, batch_idx=batch_idx)
        
        # Save progress
        completed.append(batch_idx)
        with open(progress_file, 'w') as f:
            json.dump(completed, f)
    
if __name__ == "__main__":
    indexing_ufc_101(171, 208)