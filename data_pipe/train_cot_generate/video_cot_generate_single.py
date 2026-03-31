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

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=22002)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--test-mode', action='store_true', default=False)
parser.add_argument('--test-sample-size', type=int, default=2)
parser.add_argument('--max-concurrent', type=int, default=2)
parser.add_argument('--request-timeout', type=int, nargs=2, default=[10, 60])
parser.add_argument('--retry-times', type=int, default=2)
parser.add_argument('--retry-delay', type=int, default=2)
parser.add_argument('--max-video-frames', type=int, default=8)
parser.add_argument('--sample-fps', type=float, default=2.0)
parser.add_argument('--backup-interval', type=int, default=20000)
parser.add_argument('--json-root', type=str, default="/ytech_m2v_hdd/data/VR/data/vlm2vec_train/MMEB-train-sample")
parser.add_argument('--mmeb-train-root', type=str, default="/ytech_m2v_hdd/data/VR/data/vlm2vec_train/MMEB-train")
parser.add_argument('--output-root', type=str, default="/ytech_m2v_hdd/data/VR/data/vlm2vec_train/MMEB-train-sample-cot-video")
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODEL_PATH = "/ytech_m2v_hdd/model/Qwen/Qwen3-VL-8B-Instruct"

JSON_ROOT = args.json_root
MMEB_TRAIN_ROOT = args.mmeb_train_root
OUTPUT_ROOT = args.output_root

MAX_CONCURRENT = args.max_concurrent
REQUEST_TIMEOUT = tuple(args.request_timeout)
RETRY_TIMES = args.retry_times
RETRY_DELAY = args.retry_delay

TEST_MODE = args.test_mode
TEST_SAMPLE_SIZE = args.test_sample_size
MAX_VIDEO_FRAMES = args.max_video_frames
SAMPLE_FPS = args.sample_fps

STREAM_SAVE_LOCK = Lock()
TEMP_SUFFIX = ".tmp"
BACKUP_INTERVAL = args.backup_interval

warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

log_filename = f"cot_video_{args.dataset}_{args.port}.log"
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s - PORT:{args.port} - DATASET:{args.dataset} - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def init_stream_file(dataset_name: str) -> tuple:
    output_path = os.path.join(OUTPUT_ROOT, dataset_name)
    temp_path = output_path + TEMP_SUFFIX
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    processed_indices = set()
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "original_index" in item:
                            processed_indices.add(item["original_index"])
            logger.info(f"Checkpoint found, {len(processed_indices)} items processed")
        except Exception as e:
            logger.warning(f"Failed to read checkpoint, restarting: {e}")
            processed_indices = set()
            if os.path.exists(temp_path):
                shutil.move(temp_path, temp_path + f".bak_{int(time.time())}")
    
    return temp_path, processed_indices, output_path

def stream_save_item(item: Dict, temp_path: str, original_index: int):
    with STREAM_SAVE_LOCK:
        try:
            item_with_idx = {"original_index": original_index, **item}
            with open(temp_path, "a", encoding="utf-8") as f:
                json.dump(item_with_idx, f, ensure_ascii=False)
                f.write("\n")
            
            if original_index % BACKUP_INTERVAL == 0 and original_index > 0:
                with open(temp_path, "a", encoding="utf-8") as f:
                    f.flush()
                backup_path = temp_path + f".backup_{int(time.time())}"
                shutil.copy2(temp_path, backup_path)
                logger.debug(f"Item {original_index} backed up to {backup_path}")
                
        except Exception as e:
            logger.error(f"Failed to save item {original_index}: {e}")
            raise

def finalize_stream_file(temp_path: str, output_path: str, total_count: int, dataset_name: str):
    logger.info(f"Merging temp file: {temp_path} -> {output_path}")
    
    processed_items = []
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        processed_items.append(item)
        except Exception as e:
            logger.error(f"Failed to read temp file: {e}")
            return
    
    if not processed_items:
        logger.warning("Temp file empty, no data processed")
        return
    
    processed_items.sort(key=lambda x: x.get("original_index", 999999))
    final_items = []
    for item in processed_items[:total_count]:
        if "original_index" in item:
            del item["original_index"]
        final_items.append(item)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_items, f, ensure_ascii=False, indent=2)
        logger.info(f"Final file saved: {output_path} ({len(final_items)} items)")
        
        if os.path.exists(temp_path):
            backup_path = temp_path + f".final_{int(time.time())}"
            shutil.move(temp_path, backup_path)
            logger.info(f"Temp file backed up to: {backup_path}")
    except Exception as e:
        logger.error(f"Failed to save final file: {e}")

def image_to_base64_url(image_path: str) -> Optional[str]:
    try:
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None
        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        return None

def clean_video_frame_path(raw_path: str) -> str:
    if not raw_path or raw_path == "images/blank.jpg":
        return ""
    if raw_path.startswith("MMEB-train/"):
        raw_path = raw_path[len("MMEB-train/"):]
    full_path = os.path.join(MMEB_TRAIN_ROOT, raw_path)
    full_path = os.path.normpath(full_path).replace("\\", "/")
    return full_path if os.path.exists(full_path) else ""

def clean_question_text(question_text: str) -> str:
    if not question_text:
        return ""
    cleaned = question_text.replace("<video>", "").strip()
    cleaned = cleaned.replace("\n", " ").replace("\r", "")
    cleaned = " ".join(cleaned.split())
    return cleaned

def build_video_request(video_frames: List[str], full_prompt_text: str) -> Optional[Dict]:
    cleaned_frames = []
    for frame_path in video_frames[:MAX_VIDEO_FRAMES]:
        abs_path = clean_video_frame_path(frame_path)
        if abs_path:
            cleaned_frames.append(abs_path)
    
    if not cleaned_frames:
        logger.warning("No valid video frames")
        return None
    
    base64_urls = []
    for idx, frame_path in enumerate(cleaned_frames):
        b64_url = image_to_base64_url(frame_path)
        if b64_url:
            base64_urls.append(b64_url)
        else:
            logger.warning(f"Failed to encode frame: {frame_path}")
    
    if not base64_urls:
        logger.error("All frames encoding failed")
        return None
    
    user_content = [
        {"type": "text", "text": full_prompt_text},
        *[{"type": "image_url", "image_url": {"url": b64_url}} for b64_url in base64_urls]
    ]
    
    request_data = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False
    }
    
    logger.info(f"Video request built - valid frames: {len(base64_urls)}")
    return request_data

def build_text_request(full_prompt_text: str) -> Dict:
    return {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": [{"type": "text", "text": full_prompt_text}]}],
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False
    }

def build_pos_text_request(pos_text: str) -> Dict:
    prompt = PROMPT_POS_TEXT.format(pos_text=pos_text)
    return build_text_request(prompt)

def build_pos_video_retrieval_request(video_frames: List[str], task_question: str) -> Optional[Dict]:
    cleaned_frames = []
    for frame_path in video_frames[:MAX_VIDEO_FRAMES]:
        abs_path = clean_video_frame_path(frame_path)
        if abs_path:
            cleaned_frames.append(abs_path)
    
    if not cleaned_frames:
        logger.warning("POS retrieval: no valid frames")
        return None
    
    base64_urls = []
    for idx, frame_path in enumerate(cleaned_frames):
        b64_url = image_to_base64_url(frame_path)
        if b64_url:
            base64_urls.append(b64_url)
        else:
            logger.warning(f"POS retrieval: failed to encode frame: {frame_path}")
    
    if not base64_urls:
        logger.error("POS retrieval: all frames encoding failed")
        return None
    
    video_desc = f"{len(cleaned_frames)} frames sampled at {SAMPLE_FPS} FPS"
    full_prompt = PROMPT_POS_VIDEO_RETRIEVAL.format(
        pos_video_output=video_desc,
        pos_text=task_question
    )
    
    user_content = [
        {"type": "text", "text": full_prompt},
        *[{"type": "image_url", "image_url": {"url": b64_url}} for b64_url in base64_urls]
    ]
    
    request_data = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False
    }
    
    logger.info(f"POS retrieval request built - valid frames: {len(base64_urls)}")
    return request_data

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

req_session = create_requests_session()

@atexit.register
def cleanup_resources():
    global req_session, executor
    try:
        if req_session:
            req_session.close()
            logger.info("Requests session closed")
        if 'executor' in globals() and executor:
            executor.shutdown(wait=True)
            logger.info("ThreadPool closed")
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker.kill()
        except:
            pass
    except Exception as e:
        logger.warning(f"Cleanup failed: {str(e)}")

def call_model(request_data: Dict, item_idx: int) -> Optional[str]:
    global req_session
    try:
        if not check_service_health():
            logger.error(f"Item {item_idx}: service unavailable (port: {args.port})")
            return None
        
        response = req_session.post(
            ENDPOINT,
            json=request_data,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code != 200:
            error_detail = response.text
            logger.error(f"Item {item_idx}: request failed (status: {response.status_code}) - {error_detail}")
            response.raise_for_status()
        
        result = response.json()
        if not result.get("choices") or len(result["choices"]) == 0:
            logger.error(f"Item {item_idx}: no response")
            return None
        
        return result["choices"][0]["message"]["content"]
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Item {item_idx}: connection failed - {str(e)}")
        time.sleep(RETRY_DELAY)
        try:
            response = req_session.post(ENDPOINT, json=request_data, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except:
            return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Item {item_idx}: request timeout - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Item {item_idx}: request failed - {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def process_single_item_stream(item: Dict, item_idx: int, temp_path: str, is_video_retrieval: bool) -> Dict:
    logger.info(f"Processing item: {item_idx}")
    new_item = item.copy()
    new_item["query_cot"] = ""
    new_item["pos_cot"] = ""
    new_item["error"] = ""
    
    try:
        qry = item.get("qry", {})
        conversations = qry.get("conversations", [])
        
        use_answer = (item_idx + 1) % 2 == 1
        
        if is_video_retrieval:
            video_description = ""
            for conv in conversations:
                if conv.get("from") == "gpt":
                    video_description = conv.get("value", "").strip()
                    break
            
            if not video_description:
                new_item["error"] = "Missing video description"
                stream_save_item(new_item, temp_path, item_idx)
                return new_item
            
            if use_answer:
                pos = item.get("pos", {})
                pos_conversations = pos.get("conversations", [])
                answer = ""
                for conv in pos_conversations:
                    if conv.get("from") == "gpt":
                        answer = conv.get("value", "").strip()
                        break
                full_prompt = PROMPT_VIDEO_RETRIEVAL_WITH_ANSWER.format(
                    question=video_description,
                    answer=answer
                )
            else:
                full_prompt = PROMPT_VIDEO_RETRIEVAL_WITHOUT_ANSWER.format(
                    question=video_description
                )
            
            request_data = build_text_request(full_prompt)
        
        else:
            video_frames = qry.get("video", [])
            question = ""
            for conv in conversations:
                if conv.get("from") == "human":
                    question = clean_question_text(conv.get("value", ""))
                    break
            
            if not question:
                new_item["error"] = "Missing question text"
                stream_save_item(new_item, temp_path, item_idx)
                return new_item
            
            if not video_frames or video_frames == ["images/blank.jpg"]:
                new_item["error"] = "Missing video frames"
                stream_save_item(new_item, temp_path, item_idx)
                return new_item
            
            answer = ""
            if use_answer:
                pos = item.get("pos", {})
                pos_conversations = pos.get("conversations", [])
                for conv in pos_conversations:
                    if conv.get("from") == "gpt":
                        answer = conv.get("value", "").strip()
                        break
            
            video_desc = f"{len(video_frames)} frames sampled at {SAMPLE_FPS} FPS"
            if use_answer:
                full_prompt = PROMPT_VIDEO_WITH_ANSWER.format(
                    video=video_desc,
                    question=question,
                    answer=answer
                )
            else:
                full_prompt = PROMPT_VIDEO_WITHOUT_ANSWER.format(
                    video=video_desc,
                    question=question
                )
            
            request_data = build_video_request(video_frames, full_prompt)
        
        if not request_data:
            new_item["error"] = "Failed to build request"
            stream_save_item(new_item, temp_path, item_idx)
            return new_item
        
        query_cot = call_model(request_data, item_idx)
        new_item["query_cot"] = query_cot if query_cot else ""
        
        pos = item.get("pos", {})
        pos_conversations = pos.get("conversations", [])
        
        if is_video_retrieval:
            pos_video_frames = pos.get("video", [])
            if pos_video_frames and pos_video_frames != ["images/blank.jpg"]:
                pos_task = ""
                for conv in pos_conversations:
                    if conv.get("from") == "human":
                        pos_task = clean_question_text(conv.get("value", ""))
                        break
                
                if pos_task:
                    pos_request_data = build_pos_video_retrieval_request(pos_video_frames, pos_task)
                    if pos_request_data:
                        pos_cot = call_model(pos_request_data, item_idx)
                        new_item["pos_cot"] = pos_cot if pos_cot else ""
        else:
            pos_text = ""
            for conv in pos_conversations:
                if conv.get("from") == "gpt":
                    pos_text = conv.get("value", "").strip()
                    break
            
            if pos_text:
                pos_request_data = build_pos_text_request(pos_text)
                pos_cot = call_model(pos_request_data, item_idx)
                new_item["pos_cot"] = pos_cot if pos_cot else ""
        
        logger.info(f"Completed item: {item_idx}")
        stream_save_item(new_item, temp_path, item_idx)
        return new_item
    
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        new_item["error"] = error_msg
        logger.error(f"Item {item_idx} {error_msg}")
        logger.error(traceback.format_exc())
        stream_save_item(new_item, temp_path, item_idx)
        return new_item

def process_dataset_stream() -> None:
    dataset_name = args.dataset
    input_path = os.path.join(JSON_ROOT, dataset_name)
    logger.info(f"========== Start processing dataset: {dataset_name} (PORT: {args.port}) ==========")
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset {input_path}: {str(e)}")
        raise
    
    if not data:
        logger.warning("Empty dataset")
        return
    
    if TEST_MODE:
        original_len = len(data)
        data = data[:TEST_SAMPLE_SIZE]
        logger.info(f"Test mode: original {original_len}, process {len(data)}")
    
    temp_path, processed_indices, output_path = init_stream_file(dataset_name)
    is_video_retrieval = "video_retrieval" in dataset_name
    
    pending_items = []
    for idx, item in enumerate(data):
        if idx not in processed_indices:
            pending_items.append((idx, item))
        else:
            logger.info(f"Item {idx} already processed, skipped")
    
    if not pending_items:
        logger.info("All items processed")
        finalize_stream_file(temp_path, output_path, len(data), dataset_name)
        return
    
    logger.info(f"Pending items: {len(pending_items)}")
    
    global executor
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    futures = []
    
    for idx, item in pending_items:
        future = executor.submit(
            process_single_item_stream,
            item=item,
            item_idx=idx,
            temp_path=temp_path,
            is_video_retrieval=is_video_retrieval
        )
        futures.append(future)
    
    completed_count = 0
    for future in as_completed(futures):
        try:
            future.result()
            completed_count += 1
            if completed_count % 5 == 0:
                logger.info(f"Progress: {completed_count}/{len(pending_items)}")
        except Exception as e:
            logger.error(f"Task exception: {e}")
    
    finalize_stream_file(temp_path, output_path, len(data), dataset_name)
    logger.info(f"========== Finish processing dataset: {dataset_name} ==========")

def main():
    try:
        if not check_service_health():
            logger.error(f"Service unavailable (localhost:{args.port}), start service first!")
            return
        
        logger.info(f"=== Config ===")
        logger.info(f"Port: {args.port}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Test mode: {TEST_MODE}")
        logger.info(f"Max concurrency: {MAX_CONCURRENT}")
        logger.info(f"Timeout: {REQUEST_TIMEOUT}")
        logger.info(f"Retries: {RETRY_TIMES}")
        logger.info(f"=================")
        
        process_dataset_stream()
        
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        cleanup_resources()

if __name__ == "__main__":
    main()