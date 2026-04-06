import os
import json
import base64
import io
import PIL
from PIL import Image
from typing import Dict, List, Optional, Any, Union
from torch.utils.data import Dataset, Sampler
import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import math
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from collators.qwen2_vision_process import (
    smart_resize,
    IMAGE_FACTOR,
    MIN_PIXELS,
    MAX_PIXELS,
    VIDEO_MIN_PIXELS,
    VIDEO_MAX_PIXELS,
    VIDEO_TOTAL_PIXELS,
    FPS,
    FPS_MAX_FRAMES,
    process_vision_info
)

Image.MAX_IMAGE_PIXELS = None


class MMEBUnifiedDataset(Dataset):
    def __init__(
        self,
        data_configs: List[Dict],
        tokenizer=None,
        image_processor=None,
        max_length: int = 1024,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.split = split

        self.sub_datasets: List[Dataset] = []
        self.sub_dataset_names: List[str] = []
        self.sub_dataset_weights: List[float] = []
        self.cum_sizes: List[int] = []

        total = 0

        for config in data_configs:
            dataset_type = config["type"]
            data_path = config["path"]
            weight = config.get("weight", 1.0)
            max_samples = config.get("max_samples", None)

            print(f"[MMEBUnifiedDataset] Loading {dataset_type} from {data_path}")

            try:
                if dataset_type == "llavahound":
                    from .mmeb.llavahound_dataset import LlavaHoundDataset

                    json_file = config.get("json_file", None)
                    if json_file is None:
                        raise ValueError(f"json_file is required for llavahound dataset")

                    root_path = config.get("root_path", "/data/vlm2vec_train/")
                    video_frame_rate = config.get("video_frame_rate", 2)
                    max_frames = config.get("max_frames", 8)
                    concat_query_cot = config.get("concat_query_cot", True)
                    concat_target_cot = config.get("concat_target_cot", True)

                    dataset = LlavaHoundDataset(
                        data_dir=data_path,
                        json_file=json_file,
                        max_samples=max_samples,
                        root_path=root_path,
                        video_frame_rate=video_frame_rate,
                        max_frames=max_frames,
                        concat_query_cot=concat_query_cot,
                        concat_target_cot=concat_target_cot
                    )

                elif dataset_type == "mmeb":
                    from .mmeb.mmeb_dataset import MMEBDataset

                    json_file = config.get("json_file", None)
                    if json_file is None:
                        raise ValueError(f"json_file is required for mmeb dataset")

                    root_path = config.get("root_path", "/data/vlm2vec_train/MMEB-train/")
                    concat_query_cot = config.get("concat_query_cot", True)
                    concat_target_cot = config.get("concat_target_cot", True)

                    dataset = MMEBDataset(
                        data_dir=data_path,
                        json_file=json_file,
                        max_samples=max_samples,
                        root_path=root_path,
                        concat_query_cot=concat_query_cot,
                        concat_target_cot=concat_target_cot
                    )

                else:
                    raise ValueError(f"Unsupported dataset type: {dataset_type}. Only 'llavahound' and 'mmeb' are supported.")

                ds_len = len(dataset)
                if ds_len == 0:
                    print(f"[MMEBUnifiedDataset] WARNING: {dataset_type} has 0 samples, skip.")
                    continue

                dataset_name = json_file.replace('.json', '') if json_file else dataset_type

                self.sub_datasets.append(dataset)
                self.sub_dataset_names.append(dataset_name)
                self.sub_dataset_weights.append(weight)

                total += ds_len
                self.cum_sizes.append(total)

                print(f"[MMEBUnifiedDataset] Loaded {ds_len} samples from {dataset_name}")

            except Exception as e:
                print(f"[MMEBUnifiedDataset] Error loading dataset {dataset_type}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[MMEBUnifiedDataset] Total samples across all datasets: {total}")

        self.sampling_weights: Optional[List[float]] = None
        if total > 0:
            weights: List[float] = []
            total_weight = 0.0
            for ds_idx, dataset in enumerate(self.sub_datasets):
                w = self.sub_dataset_weights[ds_idx]
                n = len(dataset)
                weights.extend([w] * n)
                total_weight += w * n

            if total_weight > 0:
                self.sampling_weights = [w / total_weight for w in weights]
            else:
                self.sampling_weights = weights

    def _get_sub_dataset_index(self, idx: int) -> (int, int):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        import bisect

        ds_idx = bisect.bisect_right(self.cum_sizes, idx)
        prev_cum = 0 if ds_idx == 0 else self.cum_sizes[ds_idx - 1]
        local_idx = idx - prev_cum
        return ds_idx, local_idx

    def _truncate_text(self, text: str, max_length: int = 480) -> str:
        if not text or not self.tokenizer:
            return text
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        return self.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

    def _prepare_messages(
        self,
        data_dict: Dict,
        dataset_name: str = "unknown",
        sample_id: str = "unknown",
    ) -> List[Dict]:
        has_text = "text" in data_dict and data_dict["text"]
        has_image = ("image" in data_dict and data_dict["image"]) or (
            "image_path" in data_dict and data_dict["image_path"]
        )
        has_video = "video" in data_dict and data_dict["video"]

        if has_text:
            data_dict["text"] = self._truncate_text(data_dict["text"])

        try:
            if has_image:
                image_path = data_dict.get("image") or data_dict.get("image_path")
                if (
                    image_path
                    and isinstance(image_path, str)
                    and not os.path.exists(image_path)
                ):
                    raise FileNotFoundError(
                        f"Image file not exists in dataset {dataset_name}, sample {sample_id}: {image_path}"
                    )

            if has_video:
                video_frames = data_dict["video"]
                if isinstance(video_frames, list):
                    for i, frame_path in enumerate(video_frames):
                        if (
                            frame_path
                            and isinstance(frame_path, str)
                            and not os.path.exists(frame_path)
                        ):
                            raise FileNotFoundError(
                                f"Video frame {i} not exists in dataset {dataset_name}, sample {sample_id}: {frame_path}"
                            )
                elif (
                    video_frames
                    and isinstance(video_frames, str)
                    and not os.path.exists(video_frames)
                ):
                    raise FileNotFoundError(
                        f"Video frame not exists in dataset {dataset_name}, sample {sample_id}: {video_frames}"
                    )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Dataset: {dataset_name}, Sample: {sample_id}\n{str(e)}")

        if has_video:
            video_frames = data_dict["video"]

            if isinstance(video_frames, list):
                valid_frames = [frame for frame in video_frames if frame]
            else:
                valid_frames = [video_frames] if video_frames else []

            if not valid_frames:
                raise ValueError(
                    f"Dataset {dataset_name}, sample {sample_id}: Video has no valid frames"
                )

            user_content = []

            if has_text:
                user_content.append({"type": "text", "text": data_dict["text"]})

            video_params = data_dict.get("video_params", {})

            video_content = {
                "video": valid_frames,
                "total_pixels": video_params.get(
                    "total_pixels", VIDEO_TOTAL_PIXELS
                ),
                "min_pixels": video_params.get("min_pixels", VIDEO_MIN_PIXELS),
                "max_pixels": video_params.get("max_pixels", VIDEO_MAX_PIXELS),
                "max_frames": video_params.get("max_frames", FPS_MAX_FRAMES),
                "sample_fps": video_params.get("fps", FPS),
            }

            user_content.append(video_content)

            user_content.append({"type": "text", "text": "Summarize the above content in one word:"})

            message = [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "<emb>"}],
                },
            ]

        elif has_text and has_image:
            image_content = data_dict.get("image")
            if not image_content and "image_path" in data_dict:
                image_content = data_dict["image_path"]

            user_content = [
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": data_dict["text"]},
            ]

            user_content.append({"type": "text", "text": "Summarize the above content in one word:"})

            message = [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "<emb>"}],
                },
            ]

        elif has_text:
            user_content = [{"type": "text", "text": data_dict["text"]}]

            user_content.append({"type": "text", "text": "Summarize the above content in one word:"})

            message = [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "<emb>"}],
                },
            ]

        elif has_image:
            image_content = data_dict.get("image")
            if not image_content and "image_path" in data_dict:
                image_content = data_dict["image_path"]

            user_content = [
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }
            ]

            user_content.append({"type": "text", "text": "Summarize the above content in one word:"})

            message = [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "<emb>"}],
                },
            ]
        else:
            error_msg = (
                f"Dataset {dataset_name}, sample {sample_id}: "
                f"Data dict must contain either text, image, or video. Got: {list(data_dict.keys())}"
            )
            if data_dict:
                for key, value in data_dict.items():
                    if value:
                        error_msg += (
                            f"\n  {key}: {type(value)} "
                            f"(length={len(value) if isinstance(value, (str, list)) else 'N/A'})"
                        )
                    else:
                        error_msg += f"\n  {key}: EMPTY or None"
            else:
                error_msg += "\n  (data_dict is empty)"
            raise ValueError(error_msg)

        return message

    def __len__(self) -> int:
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx = self._get_sub_dataset_index(idx)
        dataset = self.sub_datasets[ds_idx]
        dataset_name = self.sub_dataset_names[ds_idx]
        sample_id = f"{dataset_name}_{local_idx}"

        raw_sample = dataset[local_idx]

        query_dict = raw_sample.get("query", {})
        target_dict = raw_sample.get("target", {})

        if not query_dict or not target_dict:
            print(
                f"[MMEBUnifiedDataset] WARNING: Empty query or target in {dataset_name} sample {local_idx}"
            )

        query_messages = self._prepare_messages(
            query_dict, dataset_name, sample_id
        )
        target_messages = self._prepare_messages(
            target_dict, dataset_name, sample_id
        )

        return {
            "query_messages": query_messages,
            "target_messages": target_messages,
            "dataset": dataset_name,
            "sample_id": sample_id,
        }


class EmbedRLDataCollator:

    def __init__(self, tokenizer, processor, max_length=1024):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

        if hasattr(self.processor, 'tokenizer'):
            emb_token = "<emb>"
            if emb_token not in self.processor.tokenizer.get_vocab():
                self.processor.tokenizer.add_tokens([emb_token])
                print(f"Added {emb_token} to processor.tokenizer")

            tokenizer_emb_id = self.tokenizer.convert_tokens_to_ids(emb_token)
            processor_emb_id = self.processor.tokenizer.convert_tokens_to_ids(emb_token)

            if tokenizer_emb_id != processor_emb_id:
                print(f"WARNING: tokenizer and processor.tokenizer have different <emb> IDs!")
                print(f"  tokenizer: {tokenizer_emb_id}, processor.tokenizer: {processor_emb_id}")
            else:
                print(f"<emb> token synced: ID = {tokenizer_emb_id}")

        self.IMAGE_FACTOR = IMAGE_FACTOR
        self.MIN_PIXELS = MIN_PIXELS
        self.MAX_PIXELS = MAX_PIXELS
        self.MAX_RATIO = 200

        self.VIDEO_MIN_PIXELS = VIDEO_MIN_PIXELS
        self.VIDEO_MAX_PIXELS = VIDEO_MAX_PIXELS
        self.VIDEO_TOTAL_PIXELS = VIDEO_TOTAL_PIXELS

    def _process_messages(self, messages):
        import copy
        messages = copy.deepcopy(messages)

        has_video = False
        has_image = False
        for msg in messages:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if isinstance(content, dict):
                        if 'video' in content:
                            has_video = True
                        elif 'image' in content or 'image_url' in content or content.get('type') == 'image':
                            has_image = True

        try:
            if has_video:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                image_inputs, video_inputs = process_vision_info([messages])

                if video_inputs is not None:
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors='pt',
                        padding='longest'
                    )
                else:
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=None,
                        return_tensors='pt',
                        padding='longest'
                    )

                text_inputs = {
                    'input_ids': inputs['input_ids'][0].tolist(),
                    'attention_mask': inputs['attention_mask'][0].tolist()
                }
                pixel_values = inputs.get('pixel_values', None)
                image_grid_thw = inputs.get('image_grid_thw', None)

            elif has_image:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                image_inputs, video_inputs = process_vision_info([messages])

                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors='pt',
                    padding='longest'
                )

                text_inputs = {
                    'input_ids': inputs['input_ids'][0].tolist(),
                    'attention_mask': inputs['attention_mask'][0].tolist()
                }
                pixel_values = inputs.get('pixel_values', None)
                image_grid_thw = inputs.get('image_grid_thw', None)

            else:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                text_inputs = self.tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=None
                )
                pixel_values = None
                image_grid_thw = None

        except Exception as e:
            print(f"Error processing messages: {e}")
            print(f"Messages: {messages}")
            import traceback
            traceback.print_exc()
            raise

        return text_inputs, pixel_values, image_grid_thw

    def __call__(self, batch):
        batch_size = len(batch)

        all_messages = []
        for item in batch:
            all_messages.append(item['query_messages'])
        for item in batch:
            all_messages.append(item['target_messages'])

        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        from qwen_vl_utils import process_vision_info as official_process_vision_info

        image_inputs, video_inputs, video_kwargs = official_process_vision_info(
            all_messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        if video_inputs is not None or image_inputs is not None:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **video_kwargs,
                do_resize=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        pixel_values = inputs.get('pixel_values', None)
        image_grid_thw = inputs.get('image_grid_thw', None)
        pixel_values_videos = inputs.get('pixel_values_videos', None)
        video_grid_thw = inputs.get('video_grid_thw', None)

        labels = input_ids.clone()

        emb_token_id = self.tokenizer.convert_tokens_to_ids("<emb>")
        for i in range(len(labels)):
            emb_count = (labels[i] == emb_token_id).sum().item()
            if emb_count == 0:
                print(f"CRITICAL: Sample {i} has no <emb> token!")
                print(f"  Text preview: {texts[i][:200]}")
            elif emb_count > 1:
                print(f"WARNING: Sample {i} has {emb_count} <emb> tokens!")

        batch_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        if pixel_values is not None:
            batch_data['pixel_values'] = pixel_values
        if image_grid_thw is not None:
            batch_data['image_grid_thw'] = image_grid_thw
        if pixel_values_videos is not None:
            batch_data['pixel_values_videos'] = pixel_values_videos
        if video_grid_thw is not None:
            batch_data['video_grid_thw'] = video_grid_thw

        return batch_data