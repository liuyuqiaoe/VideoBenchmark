import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import json
import pickle
import hashlib
from typing import List, Optional, Dict, Tuple
import argparse
from tqdm import tqdm
import time

class FrameExtractor:
    def __init__(self, cache_dir = "frame_cache", max_frames = 8, experiment_root = " "):
        if experiment_root == " " or not os.path.exists(experiment_root):
            self.experiment_root = os.path.join(os.getcwd(),"VideoBenchmark/frames")
            print(f"No input or invalid experiment_root {experiment_root}, set experiment_root to default: {self.experiment_root}")
        self.experiment_root = experiment_root
        os.makedirs(self.experiment_root, exist_ok=True)
        tmp_dir = os.path.join(self.experiment_root, cache_dir)
        self.cache_dir = Path(tmp_dir)
        self.max_frames = max_frames
        self.metadata_file = self.cache_dir / "metadata.json"
        
        self.cache_dir.mkdir(exist_ok=True)
        
        self.metadata = self._load_metadata()
        
        print(f"Frame extractor initialized with cache dir: {self.cache_dir}")
        print(f"Max frames per video: {max_frames}")
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def _get_video_hash(self, video_path):
        stat = os.stat(video_path)
        hash_input = f"{video_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_frame_path(self, video_hash):
        return self.cache_dir / f"{video_hash}_frames"
    
    def _get_metadata_path(self, video_hash):
        return self.cache_dir / f"{video_hash}_meta.pkl"
    
    def extract_frames(self, video_path, force_rebuild = False) -> Optional[List[Image.Image]]:
        video_path = str(video_path)
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
        
        # Generate hash for video
        video_hash = self._get_video_hash(video_path)
        frame_path = self._get_frame_path(video_hash)
        meta_path = self._get_metadata_path(video_hash)
        
        # Check if frames are already cached
        if not force_rebuild and frame_path.exists() and meta_path.exists():
            try:
                with open(meta_path, 'rb') as f:
                    frame_metadata = pickle.load(f)
                
                frame_files = [frame_path / f"frame_{i:03d}.jpg" for i in range(frame_metadata['num_frames'])]
                if all(f.exists() for f in frame_files):
                    
                    frames = []
                    for frame_file in frame_files:
                        frame = Image.open(frame_file).convert('RGB')
                        frames.append(frame)
                    
                    # Update metadata
                    frame_paths = [str(f) for f in frame_files]
                    self.metadata[video_path] = {
                        "hash": video_hash,
                        "num_frames": len(frames),
                        "frame_size": frames[0].size if frames else None,
                        "cached_at": time.time(),
                        "video_info": frame_metadata.get('video_info', {}),
                        "frame_files": frame_paths,
                        "experiment_root": self.experiment_root
                    }
                    self._save_metadata()
                    
                    return frames
                else:
                    print(f"Some cached frames missing for {Path(video_path).name}, re-extracting...")
            except Exception as e:
                print(f"Error loading cached frames: {e}, re-extracting...")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        video_info = {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration
        }
    
        if total_frames <= self.max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frame_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            print(f"No frames extracted from {video_path}")
            return None
        
        try:
            frame_path.mkdir(exist_ok=True)

            frame_paths = []
            
            for i, frame in enumerate(frames):
                frame_file = frame_path / f"frame_{i:03d}.jpg"
                frame.save(frame_file, 'JPEG', quality=85)
                frame_paths.append(str(frame_file))
            
            frame_metadata = {
                'num_frames': len(frames),
                'frame_size': frames[0].size,
                'video_info': video_info,
                'extracted_at': time.time()
            }
            
            with open(meta_path, 'wb') as f:
                pickle.dump(frame_metadata, f)
            
            self.metadata[video_path] = {
                "hash": video_hash,
                "num_frames": len(frames),
                "frame_size": frames[0].size,
                "cached_at": time.time(),
                "video_info": video_info,
                "frame_files": frame_paths,
                "experiment_root": self.experiment_root
            }
            self._save_metadata()
            
        except Exception as e:
            print(f"Warning: Could not cache frames: {e}")
        
        return frames
    
    def extract_frames_batch(self, video_paths: List[str], force_rebuild = False) -> Dict[str, List[Image.Image]]:
        results = {}
        
        for video_path in tqdm(video_paths, desc="Extracting frames"):
            frames = self.extract_frames(video_path, force_rebuild)
            if frames is not None:
                results[video_path] = frames
        
        return results
    
    def get_cache_stats(self) -> Dict:
        total_videos = len(self.metadata)
        total_frames = sum(meta.get('num_frames', 0) for meta in self.metadata.values())
        
        # Calculate cache size
        cache_size = 0
        for video_path, meta in self.metadata.items():
            video_hash = meta.get('hash', '')
            frame_path = self._get_frame_path(video_hash)
            if frame_path.exists():
                for frame_file in frame_path.glob("*.jpg"):
                    cache_size += frame_file.stat().st_size
        
        return {
            "total_videos": total_videos,
            "total_frames": total_frames,
            "cache_size_mb": cache_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "experiment_root": self.experiment_root
        }
    
    def clear_cache(self, video_path: str = None):
        if video_path is None:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            print("Cleared all frame cache")
        else:
            # Clear specific video
            if video_path in self.metadata:
                video_hash = self.metadata[video_path]['hash']
                frame_path = self._get_frame_path(video_hash)
                meta_path = self._get_metadata_path(video_hash)
                
                if frame_path.exists():
                    import shutil
                    shutil.rmtree(frame_path)
                
                if meta_path.exists():
                    meta_path.unlink()
                
                del self.metadata[video_path]
                self._save_metadata()
                print(f"Cleared cache for {Path(video_path).name}")
    
    def list_cached_videos(self):
        return list(self.metadata.keys())

def get_ucf101_video_paths(dataset_dir = ""):
    if dataset_dir == "" or not os.path.exists(dataset_dir):
        print("Please input the valid dataset path")
        return
    video_extensions = {'.avi', '.mp4', '.mov', '.mkv', '.wmv'}
    video_paths = []
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(file)
            if file_path.suffix.lower() in video_extensions:
                full_path = os.path.join(root, file)
                video_paths.append(full_path)

    return video_paths

class UCF101VideoPathCollector:
    def __init__(self, dataset_dir = " ", experiment_root = " "):
        if experiment_root == " " or not os.path.exists(experiment_root):
            self.experiment_root = os.path.join(os.getcwd(), "VideoBenchmark/UCF101_list")
            print(f"No input or invalid experiment_root {experiment_root}, set experiment_root to default: {self.experiment_root}")
        self.experiment_root = experiment_root
        os.makedirs(self.experiment_root, exist_ok=True)
        self.dataset_dir = dataset_dir
        self.video_extensions = {'.avi', '.mp4', '.mov', '.mkv', '.wmv'}
        self.video_paths = []
    
    @staticmethod
    def get_action_labels(dataset_dir):
        """
        {'PlayingDhol', 'BoxingPunchingBag', 'PlayingPiano', 'CricketShot', 'BreastStroke', 
        'MoppingFloor', 'BalanceBeam', 'PlayingSitar', 'BaseballPitch', 'LongJump', 
        'SumoWrestling', 'PushUps', 'Surfing', 'Rafting', 'BlowingCandles', 'ParallelBars', 
        'BlowDryHair', 'PizzaTossing', 'BrushingTeeth', 'ApplyEyeMakeup', 'PlayingViolin', 
        'ApplyLipstick', 'MilitaryParade', 'Skiing', 'SkyDiving', 'Biking', 'HandstandWalking', 
        'BodyWeightSquats', 'Diving', 'PommelHorse', 'CuttingInKitchen', 'PlayingFlute', 
        'TableTennisShot', 'CliffDiving', 'RopeClimbing', 'WallPushups', 'Bowling', 
        'JugglingBalls', 'FrisbeeCatch', 'VolleyballSpiking', 'HulaHoop', 'SoccerJuggling', 
        'PlayingDaf', 'TrampolineJumping', 'FrontCrawl', 'SalsaSpin', 'UnevenBars', 'TennisSwing', 
        'Archery', 'BoxingSpeedBag', 'HeadMassage', 'SoccerPenalty', 'TaiChi', 'HorseRace', 
        'PoleVault', 'JumpRope', 'Rowing', 'HammerThrow', 'SkateBoarding', 'FieldHockeyPenalty', 
        'Swing', 'Punch', 'CricketBowling', 'BandMarching', 'PlayingGuitar', 'JavelinThrow', 
        'GolfSwing', 'ShavingBeard', 'Mixing', 'Knitting', 'BasketballDunk', 'Nunchucks', 
        'WalkingWithDog', 'Drumming', 'Kayaking', 'Typing', 'FloorGymnastics', 'Fencing', 
        'Basketball', 'WritingOnBoard', 'PlayingTabla', 'PlayingCello', 'ThrowDiscus', 'PullUps', 
        'HandstandPushups', 'StillRings', 'Hammering', 'BabyCrawling', 'HighJump', 'Haircut', 'IceDancing', 
        'RockClimbingIndoor', 'BenchPress', 'Shotput', 'Lunges', 'JumpingJack', 'HorseRiding', 'Skijet', 
        'Billiards', 'CleanAndJerk', 'YoYo'}
        """
        if dataset_dir == " " or not os.path.exists(dataset_dir):
            print("Please input the valid dataset path")
            return []
        
        action_labels = set()
        for root, dirs, files in os.walk(dataset_dir):
            if len(dirs) > 0:
                tmp_s = set(dirs)
                action_labels = action_labels.union(tmp_s)
            else:
                break
        
        print(f"Found {len(action_labels)} action labels in {dataset_dir}")
        return action_labels

    def collect_video_paths(self):
        if self.dataset_dir == " " or not os.path.exists(self.dataset_dir):
            print("Please input the valid dataset path")
            return []
        
        self.video_paths = []
        
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                file_path = Path(file)
                if file_path.suffix.lower() in self.video_extensions:
                    full_path = os.path.join(root, file)
                    self.video_paths.append(full_path)
        
        print(f"Found {len(self.video_paths)} video files in {self.dataset_dir}")
        return self.video_paths

    def dump_results(self, file_name = "ucf101_video_paths"):
        if not self.video_paths:
            print("No video paths to dump. Run collect_video_paths() first.")
            return
        
        json_file = f"{file_name}.json"
        root = self.experiment_root
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        file_path = os.path.join(root, json_file)

        with open(file_path, 'w') as f:
            json.dump(self.video_paths, f, indent=2)
        print(f"Dumped {len(self.video_paths)} video paths to {file_path}")
        return file_path

    @staticmethod
    def json_to_list(file_name = "ucf101_video_paths", experiment_root = " "):
        if experiment_root == " " or not os.path.exists(experiment_root):
            invalid_root = experiment_root
            experiment_root = os.path.join(os.getcwd(), "VideoBenchmark/UCF101_list")
            print(f"No input or invalid experiment_root: {invalid_root}, set experiment_root to default: {experiment_root}")
        os.makedirs(experiment_root, exist_ok=True)
        if not (file_name.split(".")[-1] == "json"):
            file_name = f"{file_name}.json"
        file_path = os.path.join(experiment_root, file_name)
        if not os.path.isfile(file_path):
            print(f"file {file_path} does not exist")
            return []
        with open(file_path, 'r') as f:
            video_paths = json.load(f)
        print(f"Loaded {len(video_paths)} video paths")
        print(f"First few paths: {video_paths[:3]}")
        return video_paths
    
    @staticmethod
    def json_create_batches(batch_size = 64, file_name = "ucf101_video_paths", experiment_root = " "):
        if experiment_root == " " or not os.path.exists(experiment_root):
            invalid_root = experiment_root
            experiment_root = os.path.join(os.getcwd(), "VideoBenchmark/UCF101_list")
            print(f"No input or invalid experiment_root: {invalid_root}, set experiment_root to default: {experiment_root}")
        os.makedirs(experiment_root, exist_ok=True)

        video_paths = UCF101VideoPathCollector.json_to_list(file_name, experiment_root)
        if len(video_paths) == 0:
            return
        
        total_videos = len(video_paths)
        total_batches = (total_videos + batch_size - 1) // batch_size 
        batch_data = {
            "metadata": {
                "total_videos": total_videos,
                "batch_size": batch_size,
                "total_batches": total_batches,
                "json_file_path": video_paths
            },
            "batches": []
        }

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_videos)
            batch_paths = video_paths[start_idx:end_idx]
            
            batch_info = {
                "batch_index": batch_idx,
                "batch_size": len(batch_paths),
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "video_paths": batch_paths
            }
            
            batch_data["batches"].append(batch_info)
        
        batch_file = f"{file_name}_batches.json"
        batch_file_path = os.path.join(experiment_root, batch_file)
        
        with open(batch_file_path, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        print(f"Batching complete! Created {total_batches} batches saved to {batch_file_path}")
        return batch_file_path
        
    @staticmethod
    def load_batch_file(file_name = "ucf101_video_paths", experiment_root = " "):
        if experiment_root == " " or not os.path.exists(experiment_root):
            invalid_root = experiment_root
            experiment_root = os.path.join(os.getcwd(), "VideoBenchmark/UCF101_list")
            print(f"No input or invalid experiment_root: {invalid_root}, set experiment_root to default: {experiment_root}")
        os.makedirs(experiment_root, exist_ok=True)

        if file_name.find("_batches") == -1:
            batch_file = f"{file_name}_batches.json"
        else:
            batch_file = f"{file_name}.json"

        # file_path = os.path.join(os.getcwd(), "VideoBenchmark/UCF101_list")
        batch_file_path = os.path.join(experiment_root, batch_file)
        
        if not os.path.isfile(batch_file_path):
            print(f"Batch file {batch_file_path} does not exist")
            print(f"The file_name should be either the video_paths_file_name or video_paths_file_name_batches")
            return {}
    
        with open(batch_file_path, 'r') as f:
            batch_data = json.load(f)
        
        return batch_data

    @staticmethod
    def load_metadata(file_name = "ucf101_video_paths", experiment_root = " "):
        batch_data = UCF101VideoPathCollector.load_batch_file(file_name, experiment_root)
        if not batch_data:
            print("No batch data.")
            return {}
        metadata = batch_data.get("metadata", {})
        if metadata:
            print(f"Metadata: {metadata['total_videos']} videos, {metadata['total_batches']} batches, batch size: {metadata['batch_size']}")
        return metadata

    @staticmethod
    def get_batch_range(start_batch, end_batch, file_name = "ucf101_video_paths", experiment_root = " "):
        batch_data = UCF101VideoPathCollector.load_batch_file(file_name, experiment_root)
        if not batch_data:
            return []
        
        batches = batch_data.get("batches", [])
        if start_batch < 0 or end_batch >= len(batches) or start_batch > end_batch:
            print(f"Invalid batch range {start_batch}-{end_batch}. Valid range: 0-{len(batches) - 1}")
            return []
        
        output = {}
        for batch_idx in range(start_batch, end_batch + 1):
            batch_paths = batches[batch_idx]['video_paths']
            output[batch_idx] = batch_paths
        
        print(f"Loaded video paths from batches {start_batch}-{end_batch}")
        return output

if __name__ == "__main__":
    print("="*60, "Testing UCF101VideoPathCollector Starts", "="*60)
    dataset_dir = os.path.join(os.getcwd(), "VideoBenchmark/UCF-101")
    experiment_root = os.path.join(os.getcwd(), "VideoBenchmark/experiment_test")
    os.makedirs(experiment_root, exist_ok=True)
    file_name = "test_video_paths"
    batch_range_test = (2,3)
    collector = UCF101VideoPathCollector(dataset_dir, experiment_root)
    

    print("1. test UCF101VideoPathCollector.collect_video_paths()...")
    video_paths =  collector.collect_video_paths()
    print("\n")

    print("2. test UCF101VideoPathCollector.dump_results()...")
    video_paths_file_path = collector.dump_results(file_name)
    print("\n")

    print("3. test UCF101VideoPathCollector.json_to_list()...")
    video_paths =  collector.json_to_list(video_paths_file_path, experiment_root)
    print("\n")

    print("4. test UCF101VideoPathCollector.json_create_batches()...")
    batch_file_path = collector.json_create_batches(64, file_name, experiment_root)
    print("\n")

    print("5. test UCF101VideoPathCollector.load_metadata()...")
    metadata = collector.load_metadata(file_name, experiment_root)
    print("\n")

    print("6. test UCF101VideoPathCollector.get_batch_range()...")
    output = collector.get_batch_range(batch_range_test[0], batch_range_test[1], file_name, experiment_root)
    print(f"The batch length of batch_{batch_range_test[0]} is {len(output[batch_range_test[0]])}")
    print(f"First 3 video paths in batch_{batch_range_test[0]}: {output[batch_range_test[0]][:3]}")
    print("="*60, "Testing UCF101VideoPathCollector Ends", "="*60)

    # test FrameExtractor:
    print("="*60, "Testing FrameExtractor Starts", "="*60)
    cache_dir = "frame_cache_test"
    max_frames = 8
    experiment_root = os.path.join(os.getcwd(), "VideoBenchmark/experiment_test")
    extractor = FrameExtractor(cache_dir, max_frames, experiment_root)
    test_video_paths =list(output.values())[0]

    print("1. test FrameExtractor.extract_frames()...")
    print("Case 1: frames have never been cached before")
    frames_1 = extractor.extract_frames(test_video_paths[0])
    print(f"Length of output frames: {len(frames_1)}")
    print("Case 2: frames are already been cached")
    frames_2 = extractor.extract_frames(test_video_paths[0])
    print(f"Length of output frames: {len(frames_2)}")
    print("\n")

    print("2. test FrameExtractor.extract_frames_batch()...")
    results = extractor.extract_frames_batch(test_video_paths)
    print(f"Length of cached videos: {len(list(results.keys()))}")
    print(f"Length of frames generated by {list(results.keys())[0]}: {len(results[list(results.keys())[0]])}")
    print("\n")

    print("3. test FrameExtractor.get_cache_stats()...")
    stats = extractor.get_cache_stats()
    print(f"stats:\n{stats}")
    print("\n")

    print("4. test FrameExtractor.list_cached_videos()...")
    cached_videos = extractor.list_cached_videos()
    print(f"Number of cached_videos: {len(cached_videos)}")
    
    print("="*60, "Testing FrameExtractor Ends", "="*60)
    