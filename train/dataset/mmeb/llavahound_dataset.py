import os
import json
from typing import Dict, List, Optional, Any, Union
from torch.utils.data import Dataset
from PIL import Image


class LlavaHoundDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        json_file: str, 
        max_samples: Optional[int] = None,
        root_path: str = "",
        video_frame_rate: int = 2,
        max_frames: int = 8,
        concat_query_cot: bool = True,
        concat_target_cot: bool = True
    ):
        self.data_dir = data_dir
        self.json_file = json_file
        self.max_samples = max_samples
        self.root_path = root_path
        self.video_frame_rate = video_frame_rate
        self.max_frames = max_frames
        self.concat_query_cot = concat_query_cot
        self.concat_target_cot = concat_target_cot
        self.samples = []
        
        json_path = os.path.join(data_dir, json_file)
        
        if not os.path.exists(json_path):
            raise ValueError(f"JSON file not found: {json_path}")
        
        print(f"Loading dataset from {json_path}")
        print(f"COT configuration: concat_query_cot={self.concat_query_cot}, concat_target_cot={self.concat_target_cot}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]
        
        print(f"Found {len(data)} samples in JSON file (max_samples={max_samples})")
        
        self.samples = self._process_data(data)
        
        print(f"Loaded {len(self.samples)} valid samples")
    
    def _build_full_path(self, path: Union[str, List[str]]) -> List[str]:
        if isinstance(path, str):
            if path == "images/blank.jpg":
                return []
            paths = [path]
        elif isinstance(path, list):
            paths = path
        else:
            return []
        
        full_paths = []
        for p in paths:
            if not p or p.strip() == "" or p == "images/blank.jpg":
                continue
            
            full_path = os.path.join(self.root_path, p)
            full_paths.append(full_path)
        
        return full_paths
    
    def _extract_conversation_text(self, conversations: List[Dict], role: str) -> str:
        for conv in conversations:
            if conv.get("from") == role:
                value = conv.get("value", "")
                if value.startswith("<video>"):
                    value = value.replace("<video>", "").strip()
                return value
        return ""
    
    def _build_text_with_cot(self, base_text: str, cot_text: str, concat_cot: bool) -> str:
        if concat_cot and cot_text:
            return f"{base_text} {cot_text}".strip()
        else:
            return base_text.strip()
    
    def _process_data(self, data: List[Dict]) -> List[Dict[str, Any]]:
        processed_samples = []
        
        for idx, item in enumerate(data):
            try:
                dataset_name = item.get("dataset_name", "")
                
                query_conversations = item.get("qry", {}).get("conversations", [])
                pos_conversations = item.get("pos", {}).get("conversations", [])
                
                query_video_paths = item.get("qry", {}).get("video", [])
                pos_video_paths = item.get("pos", {}).get("video", [])
                
                query_cot = item.get("query_cot", "")
                pos_cot = item.get("pos_cot", "")
                
                if dataset_name == "llavahound_caption_retrieval":
                    query_base_text = self._extract_conversation_text(query_conversations, "human")
                    target_base_text = self._extract_conversation_text(pos_conversations, "gpt")
                
                elif dataset_name == "llavahound_qa":
                    query_base_text = self._extract_conversation_text(query_conversations, "human")
                    target_base_text = self._extract_conversation_text(pos_conversations, "gpt")
                
                elif dataset_name == "llavahound_video_retrieval":
                    query_base_text = self._extract_conversation_text(query_conversations, "gpt")
                    target_base_text = self._extract_conversation_text(pos_conversations, "human")
                else:
                    print(f"Warning: Unknown dataset name: {dataset_name} for sample {idx}")
                    continue
                
                query_text = query_base_text.strip()
                target_text = target_base_text.strip()
                actual_query_cot = query_cot
                actual_target_cot = pos_cot
                
                full_query_video_paths = self._build_full_path(query_video_paths)
                full_target_video_paths = self._build_full_path(pos_video_paths)
                
                has_query_video = len(full_query_video_paths) > 0
                has_target_video = len(full_target_video_paths) > 0
                has_query_text = bool(query_text)
                has_target_text = bool(target_text)
                
                if has_query_video and has_query_text and has_target_text and not has_target_video:
                    task = 'video2text'
                elif has_query_text and not has_query_video and has_target_video and not has_target_text:
                    task = 'text2video'
                elif has_query_text and not has_query_video and has_target_text and not has_target_video:
                    task = 'text2text'
                elif has_query_video and not has_query_text and has_target_text and not has_target_video:
                    task = 'video2text'
                elif has_query_text and not has_query_video and has_target_text and has_target_video:
                    task = 'text2textvideo'
                elif has_query_video and has_query_text and has_target_text and has_target_video:
                    task = 'textvideo2textvideo'
                elif not has_query_text and not has_query_video and has_target_text and has_target_video:
                    task = 'textvideo2textvideo'
                else:
                    task = f'unknown_{dataset_name}'
                    print(f"Warning: Unknown task type for sample {idx}, dataset: {dataset_name}")
                    print(f"  Query: text={has_query_text}, video={has_query_video}")
                    print(f"  Target: text={has_target_text}, video={has_target_video}")
                
                video_params = {
                    "total_pixels": 300 * 32 * 32 * self.max_frames,
                    "min_pixels": 128 * 32 * 32,
                    "max_pixels": 300 * 32 * 32,
                    "max_frames": self.max_frames,
                    "fps": self.video_frame_rate,
                }
                
                sample_dict = {
                    'task': task,
                    'dataset_name': dataset_name,
                    'query_text': query_text,
                    'query_video_paths': full_query_video_paths,
                    'query_cot': actual_query_cot,
                    'target_text': target_text,
                    'target_video_paths': full_target_video_paths,
                    'target_cot': actual_target_cot,
                    'video_params': video_params,
                    'original_idx': idx,
                }

                if 'error' in item:
                    sample_dict['error'] = item['error']
                
                processed_samples.append(sample_dict)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return processed_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        query_dict = {
            'text': sample['query_text'],
        }

        if sample['query_video_paths']:
            query_dict['video'] = sample['query_video_paths']
            query_dict["video_params"] = sample["video_params"] 

        if self.concat_query_cot and sample.get('query_cot'):
            query_dict['cot'] = sample['query_cot']

        target_dict = {
            'text': sample['target_text'],
        }

        if sample['target_video_paths']:
            target_dict['video'] = sample['target_video_paths']
            target_dict["video_params"] = sample["video_params"]

        if self.concat_target_cot and sample.get('target_cot'):
            target_dict['cot'] = sample['target_cot']

        result = {
            'task': sample['task'],
            'dataset_name': sample['dataset_name'],
            'query': query_dict,
            'target': target_dict,
            'video_params': sample['video_params'],
            'original_idx': sample['original_idx'],
        }

        return result
    
    def get_stats(self):
        stats = {
            'total_samples': len(self.samples),
            'task_distribution': {},
            'dataset_distribution': {},
            'has_query_video': 0,
            'has_target_video': 0,
            'has_query_text': 0,
            'has_target_text': 0,
        }
        
        for sample in self.samples:
            task = sample['task']
            dataset = sample['dataset_name']
            
            stats['task_distribution'][task] = stats['task_distribution'].get(task, 0) + 1
            stats['dataset_distribution'][dataset] = stats['dataset_distribution'].get(dataset, 0) + 1
            
            if sample['query_video_paths']:
                stats['has_query_video'] += 1
            if sample['target_video_paths']:
                stats['has_target_video'] += 1
            if sample['query_text']:
                stats['has_query_text'] += 1
            if sample['target_text']:
                stats['has_target_text'] += 1
        
        return stats
