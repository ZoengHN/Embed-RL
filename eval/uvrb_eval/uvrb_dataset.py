import os
import json
import random
import base64
from typing import Dict, List, Any, Tuple

from torch.utils.data import Dataset


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    assert os.path.exists(path), f"{path} not found"
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def uniform_sample_indices(num_frames: int, k: int) -> List[int]:
    if num_frames <= k:
        return list(range(num_frames))
    import numpy as np
    idx = np.linspace(0, num_frames - 1, k, dtype=int)
    return idx.tolist()


def load_frames_as_paths(frames_dir: str, num_frames: int = 8) -> Tuple[List[str], int]:
    assert os.path.isdir(frames_dir), f"frames dir {frames_dir} not found"
    files = sorted(
        [f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    assert len(files) > 0, f"no frames found in {frames_dir}"

    idx_list = uniform_sample_indices(len(files), num_frames)
    frame_paths = []
    for i in idx_list:
        frame_path = os.path.join(frames_dir, files[i])
        frame_paths.append(frame_path)
    
    return frame_paths, len(files)


class UVRBBase:
    QUERY_PROMPTS = {
        "MSR-VTT": "Find the clip that corresponds to the described scene in the given video.",
        "DiDeMo": "Find a video that includes the following described scenes.",
        "CRB-G": "Find the video according to the general text description.",
        "CRB-S": "Find the video according to the spatial description.",
        "CRB-T": "Find the video according to the temporal description.",
        "CMRB": "Find the video according to the camera motion description.",
        "DREAM-E": "Find the video according to the text description.",
        "LoVR-TH": "Find the video according to text description about video theme information.",
        "PEV-K": "Find the video according to the text description of a series of keywords.",
        "LoVR-V": "Find the long video according to the long text description.",
        "VDC-D": "Find the video according to the detailed text description.",
        "MS-TI": "Find the video clip that corresponds to the given text and the given image.",
        "MS-TV": "Find the video clip that corresponds to the given text and the given video.",
        "MSRVTT-I2V": "Find the video according to the image.",
        "LoVR-C2V": "Find the original long video according to the short video clip.",
    }

    def __init__(self, max_text_length: int = 512):
        self.max_text_length = max_text_length
    
    def _add_query_prompt(self, text: str, dataset_name: str) -> str:
        prompt = self.QUERY_PROMPTS.get(dataset_name, "")
        if prompt and text:
            return f"{prompt} {text}"
        elif prompt and not text:
            return prompt
        else:
            return text

    def _truncate_text(self, text: str) -> str:
        if text is None:
            return ""
        text = text.strip()
        if len(text) <= self.max_text_length:
            return text
        return text[: self.max_text_length]

    def _prepare_messages(self, data_dict: Dict) -> List[Dict]:
        has_text = "text" in data_dict and data_dict["text"]
        has_image = "image" in data_dict and data_dict["image"]
        has_image_path = "image_path" in data_dict and data_dict["image_path"]
        has_video = "video" in data_dict and data_dict["video"]

        if has_text:
            data_dict["text"] = self._truncate_text(data_dict["text"])

        if not has_image and has_image_path:
            has_image = True
            data_dict["image"] = data_dict["image_path"]

        import sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from collators.qwen2_vision_process import (
            VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS, FPS, FPS_MAX_FRAMES, MIN_PIXELS, MAX_PIXELS
        )

        if has_video:
            video_frames = data_dict["video"]
            if isinstance(video_frames, list):
                valid_frames = [frame for frame in video_frames if frame]
            else:
                valid_frames = [video_frames] if video_frames else []

            if not valid_frames:
                raise ValueError("Video has no valid frames")

            user_content = []

            if has_text:
                user_content.append({"type": "text", "text": data_dict["text"]})

            video_params = data_dict.get("video_params", {})

            video_content = {
                "video": valid_frames,
                "total_pixels": video_params.get("total_pixels", VIDEO_TOTAL_PIXELS),
                "min_pixels": video_params.get("min_pixels", VIDEO_MIN_PIXELS),
                "max_pixels": video_params.get("max_pixels", VIDEO_MAX_PIXELS),
                "max_frames": video_params.get("max_frames", FPS_MAX_FRAMES),
                "sample_fps": video_params.get("fps", FPS)
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
                    "max_pixels": MAX_PIXELS
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
                    "max_pixels": MAX_PIXELS
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
            raise ValueError("Data dict must contain either text, image, or video")

        return message
    
    def _get_video_frames_dir(self, video_file: str, frames_root: str, dataset_name: str) -> str:
        if dataset_name in ["VDC-D", "VCD-O"]:
            parts = video_file.split('/')
            if len(parts) == 2:
                subdir = parts[0]
                video_name = parts[1]
                video_stem = os.path.splitext(video_name)[0]
                frames_dir = os.path.join(frames_root, subdir, video_stem)
            else:
                video_stem = os.path.splitext(video_file)[0]
                frames_dir = os.path.join(frames_root, video_stem)
        elif dataset_name in ["MS-TV", "MS-TI", "LoVR-C2V", "LoVR-TH", "MSRVTT-I2V"]:
            video_stem = os.path.splitext(video_file)[0]
            frames_dir = os.path.join(frames_root, video_stem)
        
        else:
            video_stem = os.path.splitext(video_file)[0]
            frames_dir = os.path.join(frames_root, video_stem)
        
        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(
                f"Frames dir {frames_dir} not found for video {video_file}\n"
                f"Dataset: {dataset_name}, frames_root: {frames_root}"
            )
        return frames_dir


class UVRBQueryDataset(Dataset, UVRBBase):
    def __init__(
        self,
        queries_path: str,
        max_text_length: int = 512,
        dataset_name: str = "CMRB",
        frames_root: str = None,
        num_video_frames: int = 8,
        images_root: str = None,
        use_query_prompt: bool = True,
    ):
        Dataset.__init__(self)
        UVRBBase.__init__(self, max_text_length=max_text_length)

        self.dataset_name = dataset_name
        self.queries = load_jsonl(queries_path)
        self.frames_root = frames_root
        self.num_video_frames = num_video_frames
        self.images_root = images_root
        self.use_query_prompt = use_query_prompt
        self.query_ids: List[str] = [q["id"] for q in self.queries]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx: int) -> Tuple[List[Dict], str]:
        q = self.queries[idx]
        q_text = q.get("text", "")
        q_video = q.get("video", "")
        q_image = q.get("image", "")
        
        data_dict = {}

        if q_text:
            if self.use_query_prompt:
                q_text = self._add_query_prompt(q_text, self.dataset_name)
            data_dict["text"] = q_text
        elif self.use_query_prompt and (q_image or q_video):
            prompt = self._add_query_prompt("", self.dataset_name)
            if prompt:
                data_dict["text"] = prompt

        if q_image:
            if not self.images_root:
                raise ValueError(
                    f"Query has image field but images_root is None. "
                    f"Query: {q.get('id')}, image: {q_image}"
                )
            image_path = os.path.join(self.images_root, q_image)
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Image file {image_path} not found for query {q.get('id')}\n"
                    f"Query image field: {q_image}\n"
                    f"images_root: {self.images_root}"
                )
            data_dict["image_path"] = image_path
        
        if q_video and not q_image:
            if not self.frames_root:
                raise ValueError(
                    f"Query has video field but frames_root is None. "
                    f"Query: {q.get('id')}, video: {q_video}"
                )
            frames_dir = self._get_video_frames_dir(
                q_video, 
                self.frames_root, 
                self.dataset_name
            )
            frame_paths, total_frames = load_frames_as_paths(
                frames_dir, 
                self.num_video_frames
            )
            
            if total_frames > 0:
                sample_fps = self.num_video_frames * 8.0 / total_frames
            else:
                sample_fps = 1.0

            data_dict["video"] = frame_paths
            data_dict["video_params"] = {
                "total_pixels": 768 * 32 * 32 * self.num_video_frames,
                "min_pixels": 128 * 32 * 32,
                "max_pixels": 768 * 32 * 32,
                "max_frames": self.num_video_frames,
                "fps": sample_fps,
            }
        
        messages = self._prepare_messages(data_dict)
        qid = q["id"]
        return messages, qid


class UVRBCandidateDataset(Dataset, UVRBBase):
    def __init__(
        self,
        corpus_path: str,
        frames_root: str,
        num_video_frames: int = 8,
        max_text_length: int = 512,
        dataset_name: str = "CMRB",
    ):
        Dataset.__init__(self)
        UVRBBase.__init__(self, max_text_length=max_text_length)

        self.dataset_name = dataset_name
        self.corpus = load_jsonl(corpus_path)
        self.frames_root = frames_root
        self.num_video_frames = num_video_frames
        self.doc_ids: List[str] = [c["id"] for c in self.corpus]

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx: int) -> Tuple[List[Dict], str]:
        c = self.corpus[idx]
        video_file = c["video"]
        frames_dir = self._get_video_frames_dir(
            video_file, 
            self.frames_root, 
            self.dataset_name
        )
        
        frame_paths, total_frames = load_frames_as_paths(frames_dir, self.num_video_frames)
        
        if total_frames > 0:
            sample_fps = self.num_video_frames * 8.0 / total_frames
        else:
            sample_fps = 1.0

        data_dict = {
            "video": frame_paths,
            "video_params": {
                "total_pixels": 768 * 32 * 32 * self.num_video_frames,
                "min_pixels": 128 * 32 * 32,
                "max_pixels": 768 * 32 * 32,
                "max_frames": self.num_video_frames,
                "fps": sample_fps,
            }
        }

        messages = self._prepare_messages(data_dict)
        return messages, c["id"]