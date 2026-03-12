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