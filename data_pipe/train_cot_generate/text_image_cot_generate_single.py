import requests
import base64
import json
import os
import logging
import traceback
import warnings
import atexit
import time
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import tempfile
import shutil
from threading import Lock

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

BASE_URL = f"http://localhost:{args.port}"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODEL_PATH = "/ytech_m2v_hdd/model/Qwen/Qwen3-VL-8B-Instruct"

JSON_ROOT = "/ytech_m2v_hdd/data/VR/data/vlm2vec_train/MMEB-train-sample"
IMAGE_ROOT = "/ytech_m2v_hdd/data/VR/data/vlm2vec_train/MMEB-train/images"
OUTPUT_ROOT = "/ytech_m2v_hdd/data/VR/data/vlm2vec_train/MMEB-train-sample-cot"

MAX_CONCURRENT = 8
REQUEST_TIMEOUT = (10, 60)
RETRY_TIMES = 2
RETRY_DELAY = 1

TEST_MODE = args.test
TEST_SAMPLE_SIZE = 16

STREAM_SAVE_LOCK = Lock()
TEMP_SUFFIX = ".tmp"
BACKUP_INTERVAL = 10000

warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

def check_service_health() -> bool:
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        try:
            host = BASE_URL.split("//")[-1].split(":")[0]
            port = int(BASE_URL.split(":")[-1]) if ":" in BASE_URL else 80
            result = sock.connect_ex((host, port))
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

def init_stream_file(dataset_name: str) -> str:
    output_path = os.path.join(OUTPUT_ROOT, dataset_name)
    temp_path = output_path + TEMP_SUFFIX
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    processed_ids = set()
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "index" in item:
                            processed_ids.add(item["index"])
            logger.info(f"Dataset {dataset_name}: Resuming from checkpoint, {len(processed_ids)} items processed")
        except Exception as e:
            logger.warning(f"Failed to read checkpoint, restarting: {e}")
            processed_ids = set()
            if os.path.exists(temp_path):
                shutil.move(temp_path, temp_path + f".bak_{int(time.time())}")
    
    return temp_path, processed_ids, output_path

def stream_save_item(item: Dict, temp_path: str, index: int):
    with STREAM_SAVE_LOCK:
        try:
            item_with_idx = {"index": index, **item}
            
            with open(temp_path, "a", encoding="utf-8") as f:
                json.dump(item_with_idx, f, ensure_ascii=False)
                f.write("\n")
            
            if index % BACKUP_INTERVAL == 0 and index > 0:
                with open(temp_path, "r", encoding="utf-8") as f:
                    f.flush()
                shutil.copy2(temp_path, temp_path + f".backup_{int(time.time())}")
                
        except Exception as e:
            logger.error(f"Failed to save item {index}: {e}")
            raise

def finalize_stream_file(temp_path: str, output_path: str, total_count: int):
    logger.info(f"Merging temp file: {temp_path} -> {output_path}")
    
    processed_items = []
    if os.path.exists(temp_path):
        with open(temp_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        processed_items.append(item)
                    except:
                        continue
    
    processed_items.sort(key=lambda x: x.get("index", 999999))
    final_items = []
    for item in processed_items[:total_count]:
        if "index" in item:
            del item["index"]
        final_items.append(item)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_items, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Final file saved: {output_path} ({len(final_items)} items)")
    
    backup_path = temp_path + f".final_{int(time.time())}"
    shutil.move(temp_path, backup_path)
    logger.info(f"Temp file backed up to: {backup_path}")

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

IMAGE_DATASETS = ["NIGHTS.json", "VisualNews_i2t.json", "DocVQA.json", "ChartQA.json", "HatefulMemes.json",
    "SUN397.json", "ImageNet_1K.json", "Visual7W.json", "CIRR.json", "MSCOCO_i2t.json", "InfographicsVQA.json",
    "A-OKVQA.json", "VOC2007.json", "MSCOCO.json", "N24News.json",
    "OK-VQA.json", "vidore_samples.json", "vidore_samples_1.json", "vidore_samples_2.json"]

log_filename = f"cot_processing_{args.dataset.replace('.json', '')}_port{args.port}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_image_path(raw_path: str, root_path: str) -> str:
    if not raw_path:
        return ""
    if raw_path.strip().lower().startswith("images/"):
        raw_path = raw_path[len("images/"):]
    full_path = os.path.join(root_path, raw_path)
    full_path = os.path.normpath(full_path)
    full_path = full_path.replace("\\/", "/")
    return full_path

def load_json(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON {file_path}: {str(e)}")
        raise

def encode_image(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        logger.warning(f"Image not found, skip encoding: {image_path}")
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        return None

def build_model_request(prompt: str, image_base64: Optional[str] = None) -> Dict:
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": prompt})
    if image_base64:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
    return {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False
    }

def call_model_single(prompt: str, image_path: Optional[str] = None, dataset_name: str = "", item_idx: int = 0) -> Optional[str]:
    global req_session
    try:
        if not check_service_health():
            logger.error(f"Dataset {dataset_name} Item {item_idx}: Service unavailable")
            return None
        
        image_base64 = encode_image(image_path) if image_path else None
        
        request_data = build_model_request(prompt, image_base64)
        
        response = req_session.post(
            ENDPOINT,
            json=request_data,
            timeout=REQUEST_TIMEOUT
        )
        
        response.raise_for_status()
        result = response.json()
        
        if not result.get("choices") or len(result["choices"]) == 0:
            logger.error(f"Dataset {dataset_name} Item {item_idx}: No choices in response")
            return None
        
        return result["choices"][0]["message"]["content"]
    
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Dataset {dataset_name} Item {item_idx}: Connection failed - {str(e)}")
        time.sleep(RETRY_DELAY)
        try:
            response = req_session.post(ENDPOINT, json=request_data, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except:
            return None
    except requests.exceptions.Timeout as e:
        logger.error(f"Dataset {dataset_name} Item {item_idx}: Request timeout - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Dataset {dataset_name} Item {item_idx}: Request failed - {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def process_single_item_stream(item: Dict, item_idx: int, is_image_dataset: bool, dataset_name: str, temp_path: str) -> Dict:
    logger.info(f"Dataset {dataset_name} Processing item: {item_idx}")
    new_item = item.copy()
    new_item["query_cot"] = ""
    new_item["pos_cot"] = ""
    new_item["error"] = ""
    
    try:
        question = item.get("qry", "").strip()
        if not question:
            new_item["error"] = "Empty query"
            logger.warning(f"Dataset {dataset_name} Item {item_idx}: Empty query")
            stream_save_item(new_item, temp_path, item_idx)
            return new_item
        
        use_answer = (item_idx + 1) % 2 == 1
        answer = ""
        if use_answer:
            answer_parts = []
            if item.get("pos_text") and item["pos_text"].strip():
                answer_parts.append(f"Text: {item['pos_text'].strip()}")
            if item.get("pos_image_path") and item["pos_image_path"].strip():
                answer_parts.append(f"Image Path: {item['pos_image_path'].strip()}")
            answer = " | ".join(answer_parts) if answer_parts else ""
        
        if is_image_dataset:
            qry_image_path = item.get("qry_image_path", "").strip()
            full_image_path = clean_image_path(qry_image_path, IMAGE_ROOT)
            if use_answer:
                prompt = PROMPT_IMAGE_WITH_ANSWER.format(question=question, answer=answer)
            else:
                prompt = PROMPT_IMAGE_WITHOUT_ANSWER.format(question=question)
            query_cot = call_model_single(prompt, full_image_path, dataset_name, item_idx)
        else:
            if use_answer:
                prompt = PROMPT_TEXT_WITH_ANSWER.format(question=question, answer=answer)
            else:
                prompt = PROMPT_TEXT_WITHOUT_ANSWER.format(question=question)
            query_cot = call_model_single(prompt, None, dataset_name, item_idx)
        
        new_item["query_cot"] = query_cot if query_cot else ""
        
        pos_text = item.get("pos_text", "").strip()
        pos_image_path = item.get("pos_image_path", "").strip()
        if pos_text or pos_image_path:
            if pos_image_path:
                full_pos_image = clean_image_path(pos_image_path, IMAGE_ROOT)
                pos_image_desc = f"Image path: {full_pos_image}" if os.path.exists(full_pos_image) else "Image not found"
                pos_prompt = PROMPT_POS_IMAGE.format(
                    pos_text=pos_text,
                    pos_image_description=pos_image_desc
                )
                pos_cot = call_model_single(pos_prompt, full_pos_image, dataset_name, item_idx)
            else:
                pos_prompt = PROMPT_POS_TEXT.format(pos_text=pos_text)
                pos_cot = call_model_single(pos_prompt, None, dataset_name, item_idx)
            new_item["pos_cot"] = pos_cot if pos_cot else ""
        
        stream_save_item(new_item, temp_path, item_idx)
        logger.info(f"Dataset {dataset_name} Completed item: {item_idx}")
        return new_item
    
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        new_item["error"] = error_msg
        logger.error(f"Dataset {dataset_name} Item {item_idx} {error_msg}")
        logger.debug(traceback.format_exc())
        stream_save_item(new_item, temp_path, item_idx)
        return new_item

def process_dataset_stream(dataset_name: str) -> None:
    input_path = os.path.join(JSON_ROOT, dataset_name)
    logger.info(f"========== Start processing dataset: {dataset_name} (Port:{args.port}) ==========")
    
    data = load_json(input_path)
    if not data:
        logger.warning(f"Dataset {dataset_name} is empty")
        return
    
    if TEST_MODE:
        original_len = len(data)
        data = data[:TEST_SAMPLE_SIZE]
        logger.info(f"Dataset {dataset_name} Test mode: Original {original_len}, Process {len(data)}")
    
    temp_path, processed_ids, output_path = init_stream_file(dataset_name)
    
    is_image_dataset = dataset_name in IMAGE_DATASETS
    
    pending_data = []
    for idx, item in enumerate(data):
        if idx not in processed_ids:
            pending_data.append((idx, item))
        else:
            logger.info(f"Dataset {dataset_name} Item {idx} already processed, skip")
    
    if not pending_data:
        logger.info(f"Dataset {dataset_name} All items processed")
        finalize_stream_file(temp_path, output_path, len(data))
        return
    
    logger.info(f"Dataset {dataset_name} Pending items: {len(pending_data)}")
    
    global executor
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    futures = []
    
    for idx, item in pending_data:
        future = executor.submit(
            process_single_item_stream,
            item=item,
            item_idx=idx,
            is_image_dataset=is_image_dataset,
            dataset_name=dataset_name,
            temp_path=temp_path
        )
        futures.append(future)
    
    completed_count = 0
    for future in as_completed(futures):
        try:
            future.result()
            completed_count += 1
            logger.info(f"Dataset {dataset_name} Progress: {completed_count}/{len(pending_data)}")
        except Exception as e:
            logger.error(f"Task exception: {e}")
    
    finalize_stream_file(temp_path, output_path, len(data))
    logger.info(f"========== Finish processing dataset: {dataset_name} (Port:{args.port}) ==========\n")

def main():
    try:
        if not check_service_health():
            logger.error(f"Service unavailable ({BASE_URL}), start service first!")
            return
        
        logger.info(f"=== Config ===")
        logger.info(f"Test mode: {TEST_MODE} (Sample size: {TEST_SAMPLE_SIZE})")
        logger.info(f"Concurrency: Max={MAX_CONCURRENT}, Timeout={REQUEST_TIMEOUT}, Retries={RETRY_TIMES}")
        logger.info(f"Image root: {IMAGE_ROOT}")
        logger.info(f"Stream save: Backup interval={BACKUP_INTERVAL}, Temp suffix={TEMP_SUFFIX}")
        logger.info(f"Processing dataset: {args.dataset}, Port: {args.port}")
        
        process_dataset_stream(args.dataset)
        
        logger.info(f"Dataset {args.dataset} (Port:{args.port}) processing completed!")
    
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
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

if __name__ == "__main__":
    main()