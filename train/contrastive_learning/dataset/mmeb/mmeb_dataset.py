import os
import json
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
from PIL import Image


class MMEBDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        json_file: str = "A-OKVQA.json",
        max_samples: Optional[int] = None,
        root_path: str = "./MMEB-train/",
        concat_query_cot: bool = True,
        concat_target_cot: bool = True
    ):
        self.data_dir = data_dir
        self.json_file = json_file
        self.max_samples = max_samples
        self.root_path = root_path
        self.concat_query_cot = concat_query_cot
        self.concat_target_cot = concat_target_cot
        self.samples = []
        
        json_path = os.path.join(data_dir, json_file)
        
        if not os.path.exists(json_path):
            raise ValueError(f"JSON file not found: {json_path}")
        
        print(f"Loading dataset from {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]
        
        print(f"Found {len(data)} samples in JSON file (max_samples={max_samples})")
        print(f"Config: concat_query_cot={self.concat_query_cot}, concat_target_cot={self.concat_target_cot}")
        
        self.samples = self._process_data(data)
        
        print(f"Loaded {len(self.samples)} valid samples")
    
    def _build_full_image_path(self, image_path: str) -> str:
        if not image_path or image_path.strip() == "":
            return ""

        if os.path.isabs(image_path):
            return image_path

        full_path = os.path.join(self.root_path, image_path)

        return full_path
    
    def _process_data(self, data: List[Dict]) -> List[Dict[str, Any]]:
        processed_samples = []

        for idx, item in enumerate(data):
            try:
                query_text = item.get('qry', '')
                query_cot = item.get('query_cot', '')

                full_query_text = query_text

                target_text = item.get('pos_text', '')
                target_cot = item.get('pos_cot', '')

                full_target_text = target_text
                
                query_image_path = item.get('qry_image_path', '')
                target_image_path = item.get('pos_image_path', '')
                
                full_query_image_path = self._build_full_image_path(query_image_path)
                full_target_image_path = self._build_full_image_path(target_image_path)
                
                has_query_image = bool(full_query_image_path)
                has_target_image = bool(full_target_image_path)
                has_query_text = bool(full_query_text.strip())
                has_target_text = bool(full_target_text.strip())
                
                if has_query_image and has_query_text and has_target_text and not has_target_image:
                    task = 'textimage2text'
                elif has_query_image and has_query_text and has_target_image and not has_target_text:
                    task = 'textimage2image'
                elif has_query_image and has_query_text and has_target_image and has_target_text:
                    task = 'textimage2textimage'
                elif has_query_image and not has_query_text and has_target_text:
                    task = 'image2text'
                elif not has_query_image and has_query_text and has_target_image:
                    task = 'text2image'
                elif not has_query_image and has_query_text and has_target_text:
                    task = 'text2text'
                elif has_query_image and not has_query_text and has_target_image:
                    task = 'image2image'
                else:
                    task = 'unknown'
                    print(f"Warning: Unknown task type for sample {idx}")
                
                neg_text = item.get('neg_text', '')
                neg_image_path = item.get('neg_image_path', '')
                full_neg_image_path = self._build_full_image_path(neg_image_path)
                
                sample_dict = {
                    'task': task,
                    'query_text': full_query_text,
                    'query_image_path': full_query_image_path,
                    'query_cot': query_cot,
                    'target_text': full_target_text,
                    'target_image_path': full_target_image_path,
                    'target_cot': target_cot,
                    'neg_text': neg_text,
                    'neg_image_path': full_neg_image_path,
                    'dataset_name': os.path.splitext(self.json_file)[0],
                    'original_idx': idx,
                }

                if 'error' in item:
                    sample_dict['error'] = item['error']
                
                processed_samples.append(sample_dict)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        return processed_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        query_image_path = sample['query_image_path']
        target_image_path = sample['target_image_path']

        query_dict = {
            'text': sample['query_text'],
        }

        if query_image_path:
            query_dict['image'] = query_image_path

        if self.concat_query_cot and sample.get('query_cot'):
            query_dict['cot'] = sample['query_cot']

        target_dict = {}

        if sample['target_text']:
            target_dict['text'] = sample['target_text']

        if target_image_path:
            target_dict['image'] = target_image_path

        if self.concat_target_cot and sample.get('target_cot'):
            target_dict['cot'] = sample['target_cot']

        result = {
            'task': sample['task'],
            'query': query_dict,
            'target': target_dict,
            'dataset_name': sample['dataset_name'],
            'original_idx': sample['original_idx'],
        }

        if sample.get('neg_text') or sample.get('neg_image_path'):
            result['neg_text'] = sample['neg_text']
            if sample['neg_image_path']:
                result['neg_image_path'] = sample['neg_image_path']

        return result
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            image = Image.open(image_path)
            return image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None