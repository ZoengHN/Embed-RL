import requests
import base64
import json
import os
import logging
import traceback
import warnings
import atexit
import time
import numpy as np
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
import shutil
from threading import Lock
import pandas as pd
from pathlib import Path

DATASET_INSTRUCTION = {
    'Kinetics-700': 'Recognize the category of the video content.',
    'SmthSmthV2': 'What actions or object interactions are being performed by the person in the video?',
    'UCF101': 'What activities or sports are being performed by the person in the video?',
    'HMDB51': 'What actions or objects interactions are the person in the video doing?',
    'Breakfast': 'Recognize the breakfast type that the person is cooking in the video. '
}

cot_tag = "cot"

def build_output_path(base_dir: str, cot_tag: str, dataset_name: str, type_suffix: str) -> str:
    return str(Path(base_dir) / cot_tag / "Video" / f"{dataset_name}_{type_suffix}.json")

BASE_OUTPUT_DIR = "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/"

DATASET_CONFIGS = {
    "SmthSmthV2": {
        "dataset_name": "SmthSmthV2",
        "input_format": "jsonl",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/SmthSmthV2/data/test.jsonl",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_cls/SSv2",
        "video_id_key": "video_id",
        "query_text_key": "pos_text",
        "corpus_text_key": "pos_text",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_classification",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "SmthSmthV2", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "SmthSmthV2", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "HMDB51": {
        "dataset_name": "HMDB51",
        "input_format": "jsonl",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/HMDB51/data/test.jsonl",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_cls/HMDB51",
        "video_id_key": "video_id",
        "query_text_key": "pos_text",
        "corpus_text_key": "pos_text",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_classification",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "HMDB51", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "HMDB51", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "UCF101": {
        "dataset_name": "UCF101",
        "input_format": "jsonl",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/UCF101/data/test.jsonl",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_cls/UCF101",
        "video_id_key": "video_id",
        "query_text_key": "pos_text",
        "corpus_text_key": "pos_text",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_classification",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "UCF101", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "UCF101", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "Kinetics-700": {
        "dataset_name": "Kinetics-700",
        "input_format": "jsonl",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/Kinetics-700/data/test.jsonl",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_cls/K700",
        "video_id_key": "video_id",
        "query_text_key": "pos_text",
        "corpus_text_key": "pos_text",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_classification",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "K700", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "K700", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "Breakfast": {
        "dataset_name": "Breakfast",
        "input_format": "jsonl",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/Breakfast/data/test.jsonl",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_cls/Breakfast",
        "video_id_key": "video_id",
        "query_text_key": "pos_text",
        "corpus_text_key": "pos_text",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_classification",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "Breakfast", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "Breakfast", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "MSR-VTT": {
        "dataset_name": "MSR-VTT",
        "input_format": "json",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/MSR-VTT/msrvtt_test_1k.json",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_ret/MSR-VTT",
        "video_id_key": "video_id",
        "query_text_key": "caption",
        "corpus_text_key": None,
        "query_has_video": False,
        "corpus_has_video": True,
        "task_type": "video_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MSR-VTT", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MSR-VTT", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "MSVD": {
        "dataset_name": "MSVD",
        "input_format": "json",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/MSVD/msvd_test.json",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_ret/MSVD",
        "video_id_key": "video_id",
        "query_text_key": "caption",
        "corpus_text_key": None,
        "query_has_video": False,
        "corpus_has_video": True,
        "task_type": "video_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MSVD", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MSVD", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "DiDeMo": {
        "dataset_name": "DiDeMo",
        "input_format": "json",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/DiDeMo/didemo_test.json",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_ret/DiDeMo",
        "video_id_key": "video",
        "query_text_key": "caption",
        "corpus_text_key": None,
        "query_has_video": False,
        "corpus_has_video": True,
        "task_type": "video_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "DiDeMo", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "DiDeMo", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "YouCook2": {
        "dataset_name": "YouCook2",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/lmms-lab/YouCook2/data/val-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_ret/YouCook2",
        "video_id_key": "id",
        "query_text_key": "sentence",
        "corpus_text_key": None,
        "query_has_video": False,
        "corpus_has_video": True,
        "task_type": "video_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "YouCook2", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "YouCook2", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "VATEX": {
        "dataset_name": "VATEX",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/VATEX/vatex_test/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_ret/VATEX",
        "video_id_key": "videoID",
        "query_text_key": "enCap",
        "corpus_text_key": None,
        "query_has_video": False,
        "corpus_has_video": True,
        "task_type": "video_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "VATEX", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "VATEX", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "QVHighlight": {
        "dataset_name": "QVHighlight",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/QVHighlight/data/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_mret/QVHighlight",
        "video_id_key": "video_path",
        "query_text_key": "query",
        "corpus_text_key": None,
        "query_has_video": True,
        "corpus_has_video": True,
        "task_type": "moment_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "QVHighlight", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "QVHighlight", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": "clip_start"
    },
    "Charades-STA": {
        "dataset_name": "Charades-STA",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/Charades-STA/data/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_mret/Charades-STA",
        "video_id_key": "video_path",
        "query_text_key": "query",
        "corpus_text_key": None,
        "query_has_video": True,
        "corpus_has_video": True,
        "task_type": "moment_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "Charades-STA", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "Charades-STA", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": "clip_start"
    },
    "MomentSeeker": {
        "dataset_name": "MomentSeeker",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/MomentSeeker/data/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_mret/MomentSeeker/video_frames",
        "video_id_key": "input_frames",
        "query_text_key": "query",
        "corpus_text_key": None,
        "query_has_video": True,
        "corpus_has_video": True,
        "task_type": "moment_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MomentSeeker", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MomentSeeker", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "MomentSeeker_1k8": {
        "dataset_name": "MomentSeeker_1k8",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/MomentSeeker_1k8/data/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_mret/MomentSeeker/video_frames",
        "video_id_key": "input_frames",
        "query_text_key": "query",
        "corpus_text_key": None,
        "query_has_video": True,
        "corpus_has_video": True,
        "task_type": "moment_retrieval",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MomentSeeker_1k8", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MomentSeeker_1k8", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": None
    },
    "Video-MME": {
        "dataset_name": "Video-MME",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/Video-MME/videomme/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_qa/Video-MME",
        "video_id_key": "videoID",
        "query_text_key": "question",
        "corpus_text_key": "options",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_qa",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "Video-MME", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "Video-MME", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": "options",
        "unique_id_key": "question_id"
    },
    "NExTQA": {
        "dataset_name": "NExTQA",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/NExTQA/MC/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_qa/NExTQA",
        "video_id_key": "video",
        "query_text_key": "question",
        "corpus_text_key": "options",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_qa",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "NExTQA", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "NExTQA", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": "options",
        "unique_id_key": ["video", "qid"]
    },
    "EgoSchema": {
        "dataset_name": "EgoSchema",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/EgoSchema/Subset/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_qa/EgoSchema",
        "video_id_key": "video_idx",
        "query_text_key": "question",
        "corpus_text_key": "option",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_qa",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "EgoSchema", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "EgoSchema", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": "option"
    },
    "MVBench": {
        "dataset_name": "MVBench",
        "input_format": "json",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/MVBench/json",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_qa/MVBench",
        "video_id_key": "video",
        "query_text_key": "question",
        "corpus_text_key": "candidates",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_qa",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MVBench", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "MVBench", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": "candidates",
        "unique_id_key": ["subset", "video"]
    },
    "ActivityNetQA": {
        "dataset_name": "ActivityNetQA",
        "input_format": "parquet",
        "input_path": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Video/VLM2Vec/ActivityNetQA/data/test-00000-of-00001.parquet",
        "frame_root": "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/video-tasks/frames/video_qa/ActivityNetQA",
        "video_id_key": "video_name",
        "query_text_key": "question",
        "corpus_text_key": "options",
        "query_has_video": True,
        "corpus_has_video": False,
        "task_type": "video_qa",
        "output_query_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "ActivityNetQA", "query_cot"),
        "output_corpus_path": build_output_path(BASE_OUTPUT_DIR, cot_tag, "ActivityNetQA", "corpus_cot"),
        "class_labels": None,
        "corpus_clip_key": "options",
        "unique_id_key": "question_id"
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=22002)
parser.add_argument('--dataset', type=str, nargs='+', required=True)
parser.add_argument('--test-mode', action='store_true', default=False)
parser.add_argument('--test-sample-size', type=int, default=1)
parser.add_argument('--max-concurrent', type=int, default=1)
parser.add_argument('--request-timeout', type=int, nargs=2, default=[10, 60])
parser.add_argument('--retry-times', type=int, default=2)
parser.add_argument('--retry-delay', type=int, default=2)
parser.add_argument('--max-video-frames', type=int, default=8)
parser.add_argument('--backup-interval', type=int, default=10000)
parser.add_argument('--output-root', type=str, default="/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/cot/Video")
args = parser.parse_args()

BASE_URL = f"http://0.0.0.0:{args.port}"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODEL_PATH="Qwen3-VL-8B"

MAX_CONCURRENT = args.max_concurrent
REQUEST_TIMEOUT = tuple(args.request_timeout)
RETRY_TIMES = args.retry_times
RETRY_DELAY = args.retry_delay

TEST_MODE = args.test_mode
TEST_SAMPLE_SIZE = args.test_sample_size
MAX_VIDEO_FRAMES = args.max_video_frames

STREAM_SAVE_LOCK = Lock()
TEMP_SUFFIX = ".tmp"
BACKUP_INTERVAL = args.backup_interval

req_session = None
executor = None

warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

class NumpyAwareEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj.item())
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj.item())
        if isinstance(obj, np.bool_):
            return bool(obj.item())
        return super().default(obj)

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def clean_numpy_type(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.float64, np.float32, np.float16, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: clean_numpy_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_numpy_type(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(clean_numpy_type(item) for item in obj)
    return obj

def load_frames(frame_dir: str) -> List[str]:
    frame_paths = []
    if not os.path.exists(frame_dir):
        logging.warning(f"Frame directory not found: {frame_dir}")
        return frame_paths
    
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
    try:
        frame_names = sorted(os.listdir(frame_dir), key=natural_sort_key)
    except PermissionError:
        logging.error(f"Permission denied: {frame_dir}")
        return frame_paths
    
    for frame_name in frame_names:
        if frame_name.lower().endswith(IMAGE_EXTENSIONS):
            frame_paths.append(os.path.join(frame_dir, frame_name))
    
    return frame_paths

def sample_frames(frames: List[str], num_segments: int) -> List[str]:
    if num_segments == 0 or not frames:
        return []
    if num_segments >= len(frames):
        return frames
    
    duration = len(frames)
    frame_id_array = np.linspace(0, duration-1, num_segments, dtype=int)
    sampled_frames = [frames[idx] for idx in frame_id_array.tolist()]
    
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[-1])
    
    return sampled_frames

def process_video_frames(frame_dir: str, num_frames: int = 8) -> List[str]:
    frames = load_frames(frame_dir)
    return sample_frames(frames, num_frames)

def get_video_frame_dir(video_identifier: str, frame_root: str) -> str:
    video_identifier = str(video_identifier).replace(".mp4", "").replace(".avi", "").replace(".webm", "")
    video_identifier = os.path.basename(video_identifier)
    
    frame_dir = os.path.join(frame_root, video_identifier)
    
    if not os.path.exists(frame_dir):
        alternative_dirs = [
            os.path.join(frame_root, f"videos_{video_identifier}"),
            os.path.join(frame_root, f"{video_identifier}_frames"),
            os.path.join(frame_root, "frames", video_identifier)
        ]
        for alt_dir in alternative_dirs:
            if os.path.exists(alt_dir):
                frame_dir = alt_dir
                break
    
    return frame_dir

def load_dataset(dataset_cfg: Dict) -> List[Dict]:
    input_format = dataset_cfg["input_format"]
    input_path = dataset_cfg["input_path"]
    
    data = []
    try:
        if input_format == "json":
            if os.path.isdir(input_path):
                for json_file in os.listdir(input_path):
                    if json_file.endswith(".json"):
                        with open(os.path.join(input_path, json_file), "r", encoding="utf-8") as f:
                            subset_data = json.load(f)
                            subset_data = clean_numpy_type(subset_data)
                            for item in subset_data:
                                item["subset"] = json_file.replace(".json", "")
                            data.extend(subset_data)
            else:
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data = clean_numpy_type(data)
        
        elif input_format == "parquet":
            df = pd.read_parquet(input_path)
            data = df.to_dict("records")
            data = clean_numpy_type(data)
        
        elif input_format == "jsonl":
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        item = clean_numpy_type(item)
                        data.append(item)
        
        logging.info(f"Loaded dataset: {input_path}, total {len(data)} samples")

        if TEST_MODE:
            original_len = len(data)
            data = data[:TEST_SAMPLE_SIZE]
            logging.info(f"[TEST MODE] Sliced to first {TEST_SAMPLE_SIZE} samples: {original_len} -> {len(data)}")

        dataset_name = dataset_cfg.get("dataset_name", "")
        task_type = dataset_cfg.get("task_type", "")
        if task_type == "video_qa":
            corpus_options_key = dataset_cfg.get("corpus_clip_key", "options")
            for item in data:
                if dataset_name == "ActivityNetQA":
                    cand_names = ["yes", "no"]
                elif dataset_name == "NExTQA":
                    cand_names = [
                        item.get("a0", ""),
                        item.get("a1", ""),
                        item.get("a2", ""),
                        item.get("a3", ""),
                        item.get("a4", "")
                    ]
                elif isinstance(corpus_options_key, list):
                    cand_names = [item.get(k, "") for k in corpus_options_key]
                else:
                    options = item.get(corpus_options_key, [])
                    cand_names = options if isinstance(options, list) else [options]

                item["dataset_infos"] = {
                    "cand_names": cand_names
                }

        return data
    
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        logging.error(traceback.format_exc())
        raise

def init_output_file(output_path: str, dataset_name: str) -> Tuple[str, set]:
    temp_path = output_path + TEMP_SUFFIX
    if TEST_MODE:
        output_dir = os.path.dirname(output_path)
        output_filename = os.path.basename(output_path).replace(".json", f"_test{TEST_SAMPLE_SIZE}.json")
        output_path = os.path.join(output_dir, output_filename)
        temp_path = output_path + TEMP_SUFFIX
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    processed_ids = set()
    if os.path.exists(temp_path):
        try:
            valid_lines = []
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        valid_lines.append(item)
                    except json.JSONDecodeError:
                        continue
            
            for item in valid_lines:
                processed_ids.add(item.get("unique_id", item.get("corpus_id", id(item))))
            logging.info(f"[{dataset_name}] Found checkpoint, {len(processed_ids)} processed items")
        except Exception as e:
            logging.warning(f"[{dataset_name}] Failed to read checkpoint, restarting: {e}")
            backup_path = temp_path + f".bak_{int(time.time())}"
            shutil.move(temp_path, backup_path)
            processed_ids = set()
    
    return temp_path, processed_ids

def stream_save_item(item: Dict, temp_path: str, unique_id: Any, dataset_name: str):
    with STREAM_SAVE_LOCK:
        try:
            item = clean_numpy_type(item)
            item["unique_id"] = unique_id
            
            with open(temp_path, "a", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, cls=NumpyAwareEncoder)
                f.write("\n")
                f.flush()
            
            try:
                line_count = len([l for l in open(temp_path, "r", encoding="utf-8") if l.strip()])
                if line_count % BACKUP_INTERVAL == 0 and os.path.getsize(temp_path) > 0:
                    backup_path = temp_path + f".backup_{int(time.time())}"
                    shutil.copy2(temp_path, backup_path)
            except:
                pass
        
        except Exception as e:
            logging.error(f"[{dataset_name}] Failed to save item {unique_id}: {e}")
            logging.error(traceback.format_exc())
            raise

def finalize_output_file(temp_path: str, output_path: str, dataset_name: str):
    logging.info(f"[{dataset_name}] Merging temp file: {temp_path} -> {output_path}")
    
    processed_items = []
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        processed_items.append(item)
                    except json.JSONDecodeError as e:
                        logging.warning(f"[{dataset_name}] Skipped invalid line: {e}")
                        continue
        except Exception as e:
            logging.error(f"[{dataset_name}] Failed to read temp file: {e}")
            return
    
    if not processed_items:
        logging.warning(f"[{dataset_name}] Empty temp file")
        return
    
    processed_items.sort(key=lambda x: x.get("unique_id", x.get("corpus_id", 999999)))
    for item in processed_items:
        if "unique_id" in item:
            del item["unique_id"]
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_items, f, ensure_ascii=False, indent=2, cls=NumpyAwareEncoder)
        logging.info(f"[{dataset_name}] Final file saved: {output_path} ({len(processed_items)} items)")
        
        backup_path = temp_path + f".final_{int(time.time())}"
        shutil.move(temp_path, backup_path)
    except Exception as e:
        logging.error(f"[{dataset_name}] Failed to save final file: {e}")
        logging.error(traceback.format_exc())

def image_to_base64_url(image_path: str) -> Optional[str]:
    try:
        if not os.path.exists(image_path):
            logging.warning(f"Image not found: {image_path}")
            return None
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        logging.error(f"Failed to encode image {image_path}: {str(e)}")
        return None

def build_request_with_video(frames: List[str], prompt: str, question: str) -> Optional[Dict]:
    base64_urls = []
    for frame_path in frames:
        b64_url = image_to_base64_url(frame_path)
        if b64_url:
            base64_urls.append(b64_url)
        else:
            logging.warning(f"Failed to encode frame: {frame_path}")
    
    if not base64_urls:
        logging.error("All frames failed to encode")
        return None
    
    video_desc = f"{len(base64_urls)} frames sampled at 2.0 FPS"
    full_prompt = prompt.format(video=video_desc, question=question)
    
    user_content = [
        {"type": "text", "text": full_prompt},
        *[{"type": "image_url", "image_url": {"url": b64_url}} for b64_url in base64_urls]
    ]
    
    request_data = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 4096,
        "temperature": 1.0,
        "stream": False
    }
    
    logging.info(f"Built video request - valid frames: {len(base64_urls)}")
    return request_data

def build_request_without_video(prompt: str, question: str) -> Dict:
    full_prompt = prompt.format(question=question)
    return {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}],
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": False
    }

def check_service_health() -> bool:
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        try:
            result = sock.connect_ex(("localhost", args.port))
            return result == 0
        finally:
            sock.close()

def create_requests_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=RETRY_TIMES,
        backoff_factor=RETRY_DELAY,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=MAX_CONCURRENT * 2,
        pool_maxsize=MAX_CONCURRENT * 2
    )
    session.mount("http://", adapter)
    session.headers.update({"Content-Type": "application/json"})
    return session

def call_model(request_data: Dict, item_id: Any, dataset_name: str) -> Optional[str]:
    global req_session
    if req_session is None:
        req_session = create_requests_session()
    
    try:
        if not check_service_health():
            logging.error(f"[{dataset_name}] Item {item_id}: Service unavailable")
            return None
        
        response = req_session.post(
            ENDPOINT,
            json=request_data,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code != 200:
            logging.error(f"[{dataset_name}] Item {item_id}: Request failed {response.status_code}")
            return None
        
        result = response.json()
        if not result.get("choices"):
            logging.error(f"[{dataset_name}] Item {item_id}: No response")
            return None
        
        return result["choices"][0]["message"]["content"]
    
    except requests.exceptions.ConnectionError as e:
        logging.error(f"[{dataset_name}] Item {item_id}: Connection failed {str(e)}")
        time.sleep(RETRY_DELAY)
        try:
            response = req_session.post(ENDPOINT, json=request_data, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return None
        except:
            return None
    except requests.exceptions.Timeout as e:
        logging.error(f"[{dataset_name}] Item {item_id}: Timeout {str(e)}")
        return None
    except Exception as e:
        logging.error(f"[{dataset_name}] Item {item_id}: Failed {str(e)}")
        return None

def generate_query_cot(item: Dict, item_id: Any, dataset_cfg: Dict, dataset_name: str) -> Optional[str]:
    query_text_key = dataset_cfg.get("query_text_key", "text")
    query_has_video = dataset_cfg.get("query_has_video", False)
    task_type = dataset_cfg.get("task_type", "video_qa")

    query_text = item.get(query_text_key, "") if query_text_key else ""
    query_text = clean_numpy_type(query_text)

    if isinstance(query_text, list):
        dataset_name_str = dataset_cfg.get("dataset_name", "")
        if dataset_name_str in ["VATEX", "MSVD"]:
            query_text = str(query_text[0]) if query_text else ""
        else:
            query_text = " ".join([str(x) for x in query_text if x])
    elif not isinstance(query_text, str):
        query_text = str(query_text)

    query_text = query_text.strip()

    dataset_name = dataset_cfg.get("dataset_name", "")
    task_instruction = ""

    if task_type == "video_retrieval":
        if dataset_name == "VATEX":
            task_instruction = f"Select a video that fits the description provided: {query_text}"
        else:
            task_instruction = f"Find a video that contains the following visual content: {query_text}"

    elif task_type == "video_classification":
        if dataset_name == "Kinetics-700":
            task_instruction = "Recognize the category of the video content."
        elif dataset_name == "SmthSmthV2":
            task_instruction = "What actions or object interactions are being performed by the person in the video?"
        elif dataset_name == "UCF101":
            task_instruction = "What activities or sports are being performed by the person in the video?"
        elif dataset_name == "HMDB51":
            task_instruction = "What actions or objects interactions are the person in the video doing?"
        elif dataset_name == "Breakfast":
            task_instruction = "Recognize the breakfast type that the person is cooking in the video."
        else:
            task_instruction = "Recognize the category of the video content."

    elif task_type == "moment_retrieval":
        task_instruction = f"Find the clip that corresponds to the described scene in the given video: {query_text}"

    elif task_type == "video_qa":
        task_instruction = f"Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer. Question: {query_text}"

    request_data = None
    if query_has_video:
        video_identifier = item.get(dataset_cfg.get("video_id_key", "video_id"), "")
        video_identifier = clean_numpy_type(video_identifier)
        frame_root = dataset_cfg.get("frame_root", "")
        dataset_name_str = dataset_cfg.get("dataset_name", "")

        if dataset_name_str.startswith("MomentSeeker"):
            if not video_identifier or str(video_identifier).strip() == "":
                request_data = build_request_without_video(
                    prompt=PROMPT_VIDEO_RETRIEVAL_WITHOUT_ANSWER,
                    question=task_instruction
                )
                return call_model(request_data, item_id, dataset_name)

            elif str(video_identifier).startswith("images/"):
                video_identifier = str(video_identifier).replace("images/", "")
                query_images_root = os.path.join(os.path.dirname(frame_root), "query_images")
                frame_dir = os.path.join(query_images_root, os.path.dirname(video_identifier))
            else:
                video_identifier = video_identifier.split(".mp4")[0].replace("/", "_")
                frame_dir = get_video_frame_dir(video_identifier, frame_root)
        elif dataset_name_str in ["QVHighlight", "Charades-STA"]:
            video_clip_name = os.path.basename(str(video_identifier)).replace(".mp4", "").replace(".avi", "")
            frame_dir = os.path.join(frame_root, video_clip_name, "query")
        elif dataset_name_str == "ActivityNetQA":
            frame_dir = os.path.join(frame_root, f"v_{video_identifier}")
        elif dataset_name_str == "MVBench":
            subset = item.get("subset", "")
            if subset:
                video_base = os.path.splitext(str(video_identifier))[0]
                frame_dir = os.path.join(frame_root, subset, video_base + os.path.splitext(str(video_identifier))[1])
            else:
                frame_dir = get_video_frame_dir(video_identifier, frame_root)
        else:
            frame_dir = get_video_frame_dir(video_identifier, frame_root)

        frame_paths = process_video_frames(frame_dir, MAX_VIDEO_FRAMES)
        if not frame_paths:
            logging.warning(f"[{dataset_name}] Item {item_id}: No valid frames")
            return None

        request_data = build_request_with_video(
            frames=frame_paths,
            prompt=PROMPT_VIDEO_WITHOUT_ANSWER,
            question=task_instruction
        )
    else:
        request_data = build_request_without_video(
            prompt=PROMPT_VIDEO_RETRIEVAL_WITHOUT_ANSWER,
            question=task_instruction
        )
    
    if not request_data:
        return None
    
    return call_model(request_data, item_id, dataset_name)

def _process_single_query_item(item: Dict, item_id: Any, temp_path: str, dataset_cfg: Dict, dataset_name: str):
    logging.info(f"[{dataset_name}] Processing query item: {item_id}")
    new_item = item.copy()
    
    try:
        query_cot = generate_query_cot(item, item_id, dataset_cfg, dataset_name)
        new_item["query_cot"] = query_cot if query_cot else ""
        
        stream_save_item(new_item, temp_path, item_id, dataset_name)
        logging.info(f"[{dataset_name}] Completed query item: {item_id}")
        return new_item
    
    except Exception as e:
        error_msg = f"Query failed: {str(e)}"
        new_item["query_cot"] = ""
        new_item["error"] = error_msg
        logging.error(f"[{dataset_name}] Item {item_id} {error_msg}")
        logging.error(traceback.format_exc())
        try:
            stream_save_item(new_item, temp_path, item_id, dataset_name)
        except:
            pass
        return new_item

def process_query_dataset(dataset_name: str, dataset_cfg: Dict):
    logging.info(f"[{dataset_name}] Starting Query COT generation")
    
    try:
        data = load_dataset(dataset_cfg)
    except Exception as e:
        logging.error(f"[{dataset_name}] Failed to load dataset: {e}")
        return
    
    if not data:
        logging.warning(f"[{dataset_name}] Empty dataset")
        return
    
    output_path = dataset_cfg.get("output_query_path", os.path.join(args.output_root, f"{dataset_name}_query_cot.json"))
    temp_path, processed_ids = init_output_file(output_path, dataset_name)
    
    pending_items = []
    for idx, item in enumerate(data):
        item_id = item.get(dataset_cfg.get("video_id_key", "video_id"), idx)
        item_id = clean_numpy_type(item_id)
        if item_id not in processed_ids:
            pending_items.append((idx, item_id, item))
        else:
            logging.info(f"[{dataset_name}] Query item {item_id} already processed")
    
    if not pending_items:
        logging.info(f"[{dataset_name}] All query items processed")
        finalize_output_file(temp_path, output_path, dataset_name)
        return
    
    logging.info(f"[{dataset_name}] Pending query items: {len(pending_items)}")
    
    global executor
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    futures = []
    
    for idx, item_id, item in pending_items:
        future = executor.submit(
            _process_single_query_item,
            item=item,
            item_id=item_id,
            temp_path=temp_path,
            dataset_cfg=dataset_cfg,
            dataset_name=dataset_name
        )
        futures.append(future)
    
    completed_count = 0
    for future in as_completed(futures):
        try:
            future.result()
            completed_count += 1
            if completed_count % max(1, TEST_SAMPLE_SIZE//2) == 0:
                logging.info(f"[{dataset_name}] Query progress: {completed_count}/{len(pending_items)}")
        except Exception as e:
            logging.error(f"[{dataset_name}] Query task failed: {e}")
    
    finalize_output_file(temp_path, output_path, dataset_name)
    logging.info(f"[{dataset_name}] Query COT generation completed")

def generate_corpus_cot(corpus_item: Any, corpus_has_video: bool, task_instruction: str, corpus_id: Any, dataset_name: str) -> Optional[str]:
    corpus_item = clean_numpy_type(corpus_item)
    
    request_data = None
    if corpus_has_video:
        dataset_cfg = DATASET_CONFIGS[dataset_name]
        frame_root = dataset_cfg.get("frame_root", "")

        if dataset_cfg.get("dataset_name", "").startswith("MomentSeeker"):
            if str(corpus_item).startswith("images/"):
                video_identifier = str(corpus_item).replace("images/", "")
                query_images_root = os.path.join(os.path.dirname(frame_root), "query_images")
                frame_dir = os.path.join(query_images_root, os.path.dirname(video_identifier))
            else:
                video_identifier = str(corpus_item).split(".mp4")[0].replace("/", "_")
                frame_dir = get_video_frame_dir(video_identifier, frame_root)
        elif dataset_cfg.get("dataset_name", "") in ["QVHighlight", "Charades-STA"]:
            frame_dir = os.path.join(frame_root, str(corpus_item))
        else:
            frame_dir = get_video_frame_dir(str(corpus_item), frame_root)

        frame_paths = process_video_frames(frame_dir, MAX_VIDEO_FRAMES)

        if not frame_paths:
            frame_dir = str(corpus_item)
            frame_paths = process_video_frames(frame_dir, MAX_VIDEO_FRAMES)
            if not frame_paths:
                logging.warning(f"[{dataset_name}] Corpus {corpus_id}: No valid frames")
                return None
        
        video_desc = f"{len(frame_paths)} frames sampled at 2.0 FPS"
        full_prompt = PROMPT_POS_VIDEO_RETRIEVAL.format(
            pos_text=task_instruction,
            pos_video_output=video_desc
        )
        
        base64_urls = []
        for frame_path in frame_paths:
            b64_url = image_to_base64_url(frame_path)
            if b64_url:
                base64_urls.append(b64_url)
        
        if not base64_urls:
            logging.error(f"[{dataset_name}] Corpus {corpus_id}: All frames failed to encode")
            return None
        
        user_content = [
            {"type": "text", "text": full_prompt},
            *[{"type": "image_url", "image_url": {"url": b64_url}} for b64_url in base64_urls]
        ]
        
        request_data = {
            "model": MODEL_PATH,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": False
        }
    else:
        corpus_text = str(corpus_item).strip()
        if not corpus_text:
            logging.warning(f"[{dataset_name}] Corpus {corpus_id}: Empty text")
            return None
        
        full_prompt = PROMPT_POS_TEXT.format(pos_text=corpus_text)
        request_data = {
            "model": MODEL_PATH,
            "messages": [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}],
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": False
        }
    
    if not request_data:
        return None
    
    return call_model(request_data, corpus_id, dataset_name)

def get_corpus_items(item: Dict, dataset_cfg: Dict) -> List[Any]:
    task_type = dataset_cfg.get("task_type", "video_qa")
    dataset_name = dataset_cfg.get("dataset_name", "Unknown")
    corpus_items = []

    if task_type == "video_retrieval":
        video_id_key = dataset_cfg.get("video_id_key", "video_id")
        video_id = item.get(video_id_key, "")
        video_id = clean_numpy_type(video_id)

        dataset_name_str = dataset_cfg.get("dataset_name", "")
        if dataset_name_str == "DiDeMo":
            video_id = os.path.splitext(os.path.basename(str(video_id)))[0]

        corpus_items.append(video_id)
    
    elif task_type == "video_classification":
        if dataset_cfg["class_labels"] is None:
            dataset_name_str = dataset_cfg.get("dataset_name", "")

            if dataset_name_str in ["Kinetics-700", "UCF101"]:
                if dataset_name_str in VIDEOCLS_LABEL_MAPPING:
                    dataset_cfg["class_labels"] = VIDEOCLS_LABEL_MAPPING[dataset_name_str]
                    logging.info(f"[{dataset_name}] Using predefined labels: {len(dataset_cfg['class_labels'])}")
                else:
                    logging.warning(f"[{dataset_name}] No predefined labels for {dataset_name_str}")
                    dataset_cfg["class_labels"] = []
            else:
                all_labels = set()
                data = load_dataset(dataset_cfg)
                for d in data:
                    pos_label = d.get("pos_text", "")
                    pos_label = clean_numpy_type(pos_label)
                    if pos_label:
                        all_labels.add(str(pos_label))

                    neg_labels = d.get("neg_text", [])
                    neg_labels = clean_numpy_type(neg_labels)
                    if isinstance(neg_labels, list):
                        for neg_label in neg_labels:
                            if neg_label:
                                all_labels.add(str(neg_label))

                dataset_cfg["class_labels"] = list(all_labels)
                logging.info(f"[{dataset_name}] Extracted unique labels: {len(all_labels)}")
        corpus_items = dataset_cfg["class_labels"]
    
    elif task_type == "moment_retrieval":
        dataset_name_str = dataset_cfg.get("dataset_name", "")

        if dataset_name_str.startswith("MomentSeeker"):
            positive_frames = item.get("positive_frames", [])
            negative_frames = item.get("negative_frames", [])

            positive_frames = clean_numpy_type(positive_frames)
            negative_frames = clean_numpy_type(negative_frames)

            if isinstance(positive_frames, list):
                for frame_dict in positive_frames:
                    if isinstance(frame_dict, dict) and "output_path" in frame_dict:
                        corpus_items.append(frame_dict["output_path"])

            if isinstance(negative_frames, list):
                for frame_dict in negative_frames[:9]:
                    if isinstance(frame_dict, dict) and "output_path" in frame_dict:
                        corpus_items.append(frame_dict["output_path"])

        elif dataset_name_str in ["QVHighlight", "Charades-STA"]:
            video_path = item.get(dataset_cfg.get("video_id_key", "video_path"), "")
            video_clip_name = os.path.basename(str(video_path)).replace(".mp4", "").replace(".avi", "")
            frame_root = dataset_cfg.get("frame_root", "")
            video_clip_dir = os.path.join(frame_root, video_clip_name)

            if os.path.exists(video_clip_dir):
                try:
                    all_subdirs = [d for d in os.listdir(video_clip_dir)
                                   if os.path.isdir(os.path.join(video_clip_dir, d))]
                    corpus_subdirs = [d for d in all_subdirs
                                      if d.startswith("positive") or d.startswith("negative_clip")]
                    corpus_items = [f"{video_clip_name}/{subdir}" for subdir in corpus_subdirs[:10]]
                except Exception as e:
                    logging.warning(f"[{dataset_name}] Failed to read directory {video_clip_dir}: {e}")
            else:
                logging.warning(f"[{dataset_name}] Directory not found: {video_clip_dir}")

        else:
            corpus_clip_key = dataset_cfg.get("corpus_clip_key", "candidates")
            corpus_clips = item.get(corpus_clip_key, [])
            corpus_clips = clean_numpy_type(corpus_clips)

            if isinstance(corpus_clips, list) and corpus_clips:
                for clip in corpus_clips:
                    if clip and str(clip).strip() and not str(clip).startswith("placeholder"):
                        corpus_items.append(clip)
                if len(corpus_items) > 10:
                    corpus_items = corpus_items[:10]
            else:
                logging.warning(f"[{dataset_name}] No valid candidates")
                return []

        if not corpus_items:
            logging.warning(f"[{dataset_name}] No valid clips")
            return []
    
    elif task_type == "video_qa":
        dataset_name_str = dataset_cfg.get("dataset_name", "")

        if dataset_name_str == "ActivityNetQA":
            corpus_items = ["yes", "no"]
        elif dataset_name_str == "NExTQA":
            corpus_items = [
                item.get("a0", ""),
                item.get("a1", ""),
                item.get("a2", ""),
                item.get("a3", ""),
                item.get("a4", "")
            ]
        else:
            corpus_options_key = dataset_cfg.get("corpus_clip_key", "options")
            if corpus_options_key is None:
                corpus_items = dataset_cfg.get("default_corpus", [])
            elif isinstance(corpus_options_key, list):
                corpus_items = [item.get(k, "") for k in corpus_options_key]
            else:
                options = item.get(corpus_options_key, [])
                if isinstance(options, list):
                    corpus_items = options
                else:
                    corpus_items = [options]
    
    valid_corpus_items = []
    for idx, corp in enumerate(corpus_items):
        corp = clean_numpy_type(corp)
        corp_is_valid = False
        if isinstance(corp, (list, tuple)):
            corp_is_valid = any([bool(x) for x in corp])
        elif isinstance(corp, dict):
            corp_is_valid = bool(corp)
        else:
            corp_is_valid = bool(corp)

        if corp_is_valid:
            valid_corpus_items.append(corp)

    return valid_corpus_items

def _process_single_corpus_item(corpus_info: Any, corpus_id: str, temp_path: str, dataset_cfg: Dict, dataset_name: str):
    logging.info(f"[{dataset_name}] Processing corpus item: {corpus_id}")
    
    corpus_item, item_id, corpus_idx, is_positive = corpus_info
    corpus_item = clean_numpy_type(corpus_item)
    
    task_type = dataset_cfg.get("task_type", "video_qa")
    task_instruction = ""

    if task_type == "video_retrieval":
        task_instruction = "Understand the content of the provided video."
    elif task_type == "video_classification":
        task_instruction = f"Analyze the action category '{corpus_item}' and its key characteristics for video classification."
    elif task_type == "moment_retrieval":
        task_instruction = "Understand the content of the provided video."
    elif task_type == "video_qa":
        task_instruction = f"Analyze this answer option '{corpus_item}' and its relevance to the video question."
    
    new_item = {
        "corpus_id": corpus_id,
        "item_id": item_id,
        "clip_index": corpus_idx,
        "is_positive": is_positive,
        "corpus_content": str(corpus_item),
        "corpus_cot": "",
        "error": ""
    }
    
    try:
        corpus_cot = generate_corpus_cot(
            corpus_item=corpus_item,
            corpus_has_video=dataset_cfg.get("corpus_has_video", False),
            task_instruction=task_instruction,
            corpus_id=corpus_id,
            dataset_name=dataset_name
        )
        new_item["corpus_cot"] = corpus_cot if corpus_cot else ""
        
        stream_save_item(new_item, temp_path, corpus_id, dataset_name)
        logging.info(f"[{dataset_name}] Completed corpus item: {corpus_id}")
        return new_item
    
    except Exception as e:
        error_msg = f"Corpus failed: {str(e)}"
        new_item["error"] = error_msg
        logging.error(f"[{dataset_name}] Corpus {corpus_id} {error_msg}")
        logging.error(traceback.format_exc())
        try:
            stream_save_item(new_item, temp_path, corpus_id, dataset_name)
        except:
            pass
        return new_item

def process_corpus_dataset(dataset_name: str, dataset_cfg: Dict):
    logging.info(f"[{dataset_name}] Starting Corpus COT generation")
    
    try:
        data = load_dataset(dataset_cfg)
    except Exception as e:
        logging.error(f"[{dataset_name}] Failed to load dataset: {e}")
        return
    
    if not data:
        logging.warning(f"[{dataset_name}] Empty dataset")
        return
    
    output_path = dataset_cfg.get("output_corpus_path", os.path.join(args.output_root, f"{dataset_name}_corpus_cot.json"))
    temp_path, processed_ids = init_output_file(output_path, dataset_name)
    
    corpus_map = {}
    task_type = dataset_cfg.get("task_type", "video_qa")
    dataset_name_str = dataset_cfg.get("dataset_name", "")

    should_deduplicate = False
    if task_type == "video_classification":
        should_deduplicate = True
    elif dataset_name_str.startswith("MomentSeeker"):
        should_deduplicate = True
    elif task_type == "video_qa":
        should_deduplicate = True

    if should_deduplicate:
        unique_corpus_items = {}
        for data_idx, item in enumerate(data):
            corpus_items = get_corpus_items(item, dataset_cfg)
            for corpus in corpus_items:
                corpus_content = str(corpus).strip()
                if corpus_content and corpus_content not in unique_corpus_items:
                    unique_corpus_items[corpus_content] = corpus

        for corpus_idx, (corpus_content, corpus_item) in enumerate(sorted(unique_corpus_items.items())):
            corpus_unique_id = f"corpus_{corpus_idx}"
            corpus_map[corpus_unique_id] = (corpus_item, corpus_idx, corpus_idx, False)

        original_count = sum(len(get_corpus_items(item, dataset_cfg)) for item in data)
        logging.info(f"[{dataset_name}] Corpus deduplicated: {original_count} -> {len(corpus_map)}")

    else:
        for data_idx, item in enumerate(data):
            unique_id_key = dataset_cfg.get("unique_id_key")
            if unique_id_key:
                if isinstance(unique_id_key, list):
                    id_parts = [str(item.get(key, "")) for key in unique_id_key]
                    item_id = "_".join(id_parts)
                else:
                    item_id = item.get(unique_id_key, data_idx)
            else:
                item_id = item.get(dataset_cfg.get("video_id_key", "video_id"), data_idx)
            item_id = clean_numpy_type(item_id)
            corpus_items = get_corpus_items(item, dataset_cfg)

            for corpus_idx, corpus in enumerate(corpus_items):
                is_positive = (corpus_idx == 0)
                corpus_unique_id = f"{item_id}_clip_{corpus_idx}_{'pos' if is_positive else 'neg'}"

                if corpus_unique_id not in corpus_map:
                    corpus_map[corpus_unique_id] = (corpus, item_id, corpus_idx, is_positive)
    
    pending_corpus = []
    for corpus_id, corpus_info in corpus_map.items():
        if corpus_id not in processed_ids:
            pending_corpus.append((corpus_id, corpus_info))
        else:
            logging.info(f"[{dataset_name}] Corpus {corpus_id} already processed")
    
    if not pending_corpus:
        logging.info(f"[{dataset_name}] All corpus items processed")
        finalize_output_file(temp_path, output_path, dataset_name)
        return
    
    logging.info(f"[{dataset_name}] Pending corpus items: {len(pending_corpus)}")
    
    global executor
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    futures = []
    
    for corpus_id, corpus_info in pending_corpus:
        future = executor.submit(
            _process_single_corpus_item,
            corpus_info=corpus_info,
            corpus_id=corpus_id,
            temp_path=temp_path,
            dataset_cfg=dataset_cfg,
            dataset_name=dataset_name
        )
        futures.append(future)
    
    completed_count = 0
    progress_step = max(1, len(pending_corpus)//10)
    for future in as_completed(futures):
        try:
            future.result()
            completed_count += 1
            if completed_count % progress_step == 0:
                logging.info(f"[{dataset_name}] Corpus progress: {completed_count}/{len(pending_corpus)}")
        except Exception as e:
            logging.error(f"[{dataset_name}] Corpus task failed: {e}")
    
    finalize_output_file(temp_path, output_path, dataset_name)
    logging.info(f"[{dataset_name}] Corpus COT generation completed")

@atexit.register
def cleanup_resources():
    global req_session, executor
    try:
        if req_session:
            req_session.close()
        if 'executor' in globals() and executor:
            executor.shutdown(wait=True)
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker.kill()
        except:
            pass
    except Exception as e:
        logging.warning(f"Cleanup failed: {str(e)}")

def main():
    if not check_service_health():
        logging.error(f"Service unavailable (localhost:{args.port})")
        return
    
    unsupported_datasets = []
    for ds in args.dataset:
        if ds not in DATASET_CONFIGS:
            unsupported_datasets.append(ds)
    if unsupported_datasets:
        logging.error(f"Unsupported datasets: {unsupported_datasets}")
        return
    
    dataset_list_str = "_".join(args.dataset)
    log_filename = f"cot_eval_{dataset_list_str}_{args.port}_testmode_{TEST_SAMPLE_SIZE}.log" if TEST_MODE else f"cot_eval_{dataset_list_str}_{args.port}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - PORT:{args.port} - DATASETS:{dataset_list_str} - TEST_MODE:{TEST_MODE} - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== Global Config ===")
    logger.info(f"Port: {args.port}")
    logger.info(f"Datasets: {args.dataset} ({len(args.dataset)})")
    logger.info(f"Test Mode: {TEST_MODE}")
    if TEST_MODE:
        logger.info(f"Test Sample Size: {TEST_SAMPLE_SIZE}")
    logger.info(f"Max Concurrent: {MAX_CONCURRENT}")
    logger.info(f"Timeout: {REQUEST_TIMEOUT}")
    logger.info(f"Retries: {RETRY_TIMES}")
    logger.info(f"====================")
    
    for idx, dataset_name in enumerate(args.dataset, 1):
        logger.info(f"\nProcessing dataset {idx}/{len(args.dataset)}: {dataset_name}")
        dataset_cfg = DATASET_CONFIGS[dataset_name]
        
        try:
            process_query_dataset(dataset_name, dataset_cfg)
            process_corpus_dataset(dataset_name, dataset_cfg)
            logger.info(f"Completed dataset {idx}/{len(args.dataset)}: {dataset_name}\n")
        
        except Exception as e:
            logger.error(f"Failed dataset {idx}/{len(args.dataset)}: {dataset_name}")
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"All {len(args.dataset)} datasets processed")

if __name__ == "__main__":
    main()