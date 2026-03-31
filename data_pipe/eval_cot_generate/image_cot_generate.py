import requests
import base64
import json
import os
import logging
import traceback
import warnings
import atexit
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import shutil
from threading import Lock
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np

BASE_URL = "http://0.0.0.0:22002"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODEL_PATH = "/ytech_m2v_hdd/model/Qwen3-VL-8B"

PARQUET_ROOT = "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/Image/ziyjiang/MMEB_Test_Instruct"
IMAGE_ROOT = "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/TIGER-Lab/MMEB-V2/image-tasks"
OUTPUT_ROOT = "/ytech_m2v_hdd/data/VR/data/vlm2vec_eval/cot/Image"

MAX_CONCURRENT = 8
REQUEST_TIMEOUT = (10, 60)
RETRY_TIMES = 2
RETRY_DELAY = 1

TEST_MODE = False
TEST_SAMPLE_SIZE = 16

STREAM_SAVE_LOCK = Lock()
TEMP_SUFFIX = ".tmp"
BACKUP_INTERVAL = 10000

DATASETS = [
    "A-OKVQA", "CIRR", "ChartQA", "Country211", "DocVQA", "EDIS",
    "FashionIQ", "GQA", "HatefulMemes", "ImageNet-1K", "ImageNet-A",
    "ImageNet-R", "InfographicsVQA", "MSCOCO", "MSCOCO_i2t", "MSCOCO_t2i",
    "N24News", "NIGHTS", "OK-VQA", "OVEN", "ObjectNet", "Place365",
    "RefCOCO", "RefCOCO-Matching", "SUN397", "ScienceQA", "TextVQA",
    "VOC2007", "VisDial", "Visual7W", "Visual7W-Pointing", "VisualNews_i2t",
    "VisualNews_t2i", "VizWiz", "WebQA", "Wiki-SS-NQ"
]

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("cot_processing_mmeb_test_stream.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def init_stream_file(dataset_name: str) -> Tuple[str, set, str]:
    output_path = os.path.join(OUTPUT_ROOT, f"{dataset_name}.json")
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
            logger.info(f"Dataset {dataset_name}: Checkpoint found, {len(processed_ids)} items processed")
        except Exception as e:
            logger.warning(f"Dataset {dataset_name}: Failed to read checkpoint, restarting: {e}")
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

def load_parquet_dataset(dataset_name: str) -> List[Dict]:
    dataset_path = os.path.join(PARQUET_ROOT, dataset_name)
    parquet_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.startswith("test-") and file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        logger.error(f"Dataset {dataset_name}: No parquet files found")
        return []
    
    df_list = []
    for file in sorted(parquet_files):
        try:
            df = pq.read_table(file).to_pandas()
            df_list.append(df)
            logger.info(f"Loaded parquet: {file} (rows: {len(df)})")
        except Exception as e:
            logger.error(f"Failed to read parquet {file}: {e}")
            continue
    
    if not df_list:
        return []
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    if TEST_MODE:
        original_len = len(combined_df)
        combined_df = combined_df.head(TEST_SAMPLE_SIZE)
        logger.info(f"Test mode: {dataset_name} Original {original_len}, Process {len(combined_df)}")
    
    data = combined_df.to_dict("records")
    
    for item in data:
        for key in item:
            na_check = pd.isna(item[key])
            if na_check.any() if isinstance(na_check, (list, np.ndarray, pd.Series)) else na_check:
                item[key] = ""
        
        for key in ["tgt_text", "tgt_img_path"]:
            if key in item:
                val = item[key]
                if isinstance(val, pa.Array):
                    val = val.to_pylist()
                elif isinstance(val, np.ndarray):
                    val = val.tolist()
                if not isinstance(val, list):
                    is_empty = pd.isna(val) or (isinstance(val, (str, list)) and len(val)==0) or (isinstance(val, (np.ndarray, pa.Array)) and val.size==0)
                    val = [val] if not is_empty else []
                item[key] = val
    
    return data

def clean_image_path(raw_path: str, dataset_name: str) -> Optional[str]:
    if not raw_path or raw_path.strip() == "":
        return None
    
    raw_path = raw_path.strip()
    if raw_path.startswith(f"{dataset_name}/"):
        pass
    elif "/" not in raw_path:
        raw_path = f"{dataset_name}/{raw_path}"
    
    full_path = os.path.join(IMAGE_ROOT, raw_path)
    full_path = os.path.normpath(full_path)
    
    if os.path.exists(full_path):
        return full_path
    else:
        logger.warning(f"Image not found: {full_path}")
        return None

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
        "temperature": 1.0,
        "stream": False
    }

def call_model_single(prompt: str, image_path: Optional[str] = None, dataset_name: str = "", item_idx: int = 0) -> Optional[str]:
    global req_session
    try:
        if not check_service_health():
            logger.error(f"Dataset {dataset_name} Item {item_idx}: Service unavailable ({BASE_URL})")
            return None
        
        image_base64 = encode_image(image_path) if image_path else None
        
        request_data = build_model_request(prompt, image_base64)
        logger.debug(f"Dataset {dataset_name} Item {item_idx}: Request sent (image={image_base64 is not None})")
        
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

def collect_corpus_from_dataset(data: List[Dict], dataset_name: str) -> Dict[str, Dict]:
    corpus_dict = {
        "text": {},
        "image": {}
    }

    special_datasets = ["OVEN", "WebQA", "EDIS", "RefCOCO-Matching"]
    is_special = dataset_name in special_datasets

    for idx, item in enumerate(data):
        tgt_inst = item.get("tgt_inst", "").strip()
        tgt_text_list = item.get("tgt_text", [])
        tgt_img_list = item.get("tgt_img_path", [])

        if is_special and tgt_text_list and tgt_img_list:
            min_len = min(len(tgt_text_list), len(tgt_img_list))
            for i in range(min_len):
                text = tgt_text_list[i].strip() if tgt_text_list[i] else ""
                img_path = tgt_img_list[i].strip() if tgt_img_list[i] else ""

                if img_path and img_path not in corpus_dict["image"]:
                    corpus_dict["image"][img_path] = {
                        "inst": tgt_inst,
                        "text": text
                    }
                    logger.debug(f"Dataset {dataset_name} Paired: image={img_path}, text={text[:50] if text else 'empty'}...")
        else:
            if tgt_text_list:
                for text in tgt_text_list:
                    text = text.strip() if text else ""
                    if text and text not in corpus_dict["text"]:
                        corpus_dict["text"][text] = {"inst": tgt_inst}

            if tgt_img_list:
                for img_path in tgt_img_list:
                    img_path = img_path.strip() if img_path else ""
                    if img_path and img_path not in corpus_dict["image"]:
                        corpus_dict["image"][img_path] = {"inst": tgt_inst}

    text_count = len(corpus_dict["text"])
    image_count = len(corpus_dict["image"])
    logger.info(f"Dataset {dataset_name}: Collected {text_count} unique text corpus, {image_count} unique image corpus")
    return corpus_dict

def generate_corpus_cot(corpus_key: str, corpus_item: Dict, corpus_type: str, dataset_name: str) -> Optional[str]:
    try:
        tgt_inst = corpus_item.get("inst", "").strip()
        paired_text = corpus_item.get("text", "").strip()

        if corpus_type == "text":
            pos_text = f"{tgt_inst} {corpus_key}".strip()
            prompt = PROMPT_POS_TEXT.format(pos_text=pos_text)
            cot = call_model_single(prompt, None, dataset_name, 0)
            return cot if cot else ""

        elif corpus_type == "image":
            full_tgt_img_path = clean_image_path(corpus_key, dataset_name)

            if full_tgt_img_path and os.path.exists(full_tgt_img_path):
                if paired_text:
                    question = f"{tgt_inst} {paired_text}".strip()
                    prompt = PROMPT_IMAGE_WITHOUT_ANSWER.format(question=question)
                    logger.debug(f"Dataset {dataset_name} Paired mode: image={corpus_key}, question={question[:100]}...")
                    cot = call_model_single(prompt, full_tgt_img_path, dataset_name, 0)
                    return cot if cot else ""
                else:
                    pos_text = tgt_inst if tgt_inst else "Retrieve the corresponding image."
                    pos_image_desc = f"Image path: {full_tgt_img_path}"
                    prompt = PROMPT_POS_IMAGE.format(
                        pos_text=pos_text,
                        pos_image_description=pos_image_desc
                    )
                    cot = call_model_single(prompt, full_tgt_img_path, dataset_name, 0)
                    return cot if cot else ""
            else:
                logger.warning(f"Image not found: {corpus_key}")
                return ""

        return ""

    except Exception as e:
        logger.error(f"Failed to generate corpus COT {corpus_type}:{corpus_key}: {str(e)}")
        logger.debug(traceback.format_exc())
        return ""

def process_corpus_cot_parallel(corpus_dict: Dict[str, Dict], dataset_name: str) -> Dict[str, Dict[str, str]]:
    total_count = len(corpus_dict["text"]) + len(corpus_dict["image"])
    logger.info(f"Dataset {dataset_name}: Start generating corpus COT, total {total_count} items")

    corpus_cot = {
        "text": {},
        "image": {}
    }

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = []

        for corpus_key, corpus_item in corpus_dict["text"].items():
            future = executor.submit(
                generate_corpus_cot,
                corpus_key,
                corpus_item,
                "text",
                dataset_name
            )
            futures.append((future, "text", corpus_key))

        for corpus_key, corpus_item in corpus_dict["image"].items():
            future = executor.submit(
                generate_corpus_cot,
                corpus_key,
                corpus_item,
                "image",
                dataset_name
            )
            futures.append((future, "image", corpus_key))

        completed_count = 0
        for future, corpus_type, corpus_key in futures:
            try:
                cot = future.result()
                corpus_cot[corpus_type][corpus_key] = cot if cot else ""
                completed_count += 1
                if completed_count % 10 == 0:
                    logger.info(f"Dataset {dataset_name} Corpus COT progress: {completed_count}/{total_count}")
            except Exception as e:
                logger.error(f"Failed to process corpus {corpus_type}:{corpus_key}: {e}")
                corpus_cot[corpus_type][corpus_key] = ""

    logger.info(f"Dataset {dataset_name}: Corpus COT generated, text: {len(corpus_cot['text'])}, image: {len(corpus_cot['image'])}")
    return corpus_cot

def process_single_item_stream(item: Dict, item_idx: int, dataset_name: str, temp_path: str) -> Dict:
    logger.info(f"Dataset {dataset_name} Processing item: {item_idx}")
    new_item = item.copy()
    new_item["query_cot"] = ""
    new_item["error"] = ""

    try:
        qry_inst = item.get("qry_inst", "").strip()
        qry_text = item.get("qry_text", "").strip()
        question = f"{qry_inst} {qry_text}".strip()

        if not question:
            new_item["error"] = "Empty query content"
            logger.warning(f"Dataset {dataset_name} Item {item_idx}: Empty query")
            stream_save_item(new_item, temp_path, item_idx)
            return new_item

        qry_img_path = item.get("qry_img_path", "").strip()
        full_qry_img_path = clean_image_path(qry_img_path, dataset_name)

        if full_qry_img_path:
            prompt = PROMPT_IMAGE_WITHOUT_ANSWER.format(question=question)
            query_cot = call_model_single(prompt, full_qry_img_path, dataset_name, item_idx)
        else:
            prompt = PROMPT_TEXT_WITHOUT_ANSWER.format(question=question)
            query_cot = call_model_single(prompt, None, dataset_name, item_idx)

        new_item["query_cot"] = query_cot if query_cot else ""

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
    logger.info(f"========== Start processing dataset: {dataset_name} ==========")

    data = load_parquet_dataset(dataset_name)
    if not data:
        logger.warning(f"Dataset {dataset_name}: Empty or load failed")
        return

    corpus_dict = collect_corpus_from_dataset(data, dataset_name)
    corpus_cot = process_corpus_cot_parallel(corpus_dict, dataset_name)

    corpus_cot_path = os.path.join(OUTPUT_ROOT, f"{dataset_name}_corpus_cot.json")
    os.makedirs(os.path.dirname(corpus_cot_path), exist_ok=True)
    with open(corpus_cot_path, "w", encoding="utf-8") as f:
        json.dump(corpus_cot, f, ensure_ascii=False, indent=2)
    logger.info(f"Corpus COT saved to: {corpus_cot_path}")

    temp_path, processed_ids, output_path = init_stream_file(dataset_name)

    pending_data = []
    for idx, item in enumerate(data):
        if idx not in processed_ids:
            pending_data.append((idx, item))
        else:
            logger.info(f"Dataset {dataset_name} Item {idx} already processed, skipped")

    if not pending_data:
        logger.info(f"Dataset {dataset_name}: All items processed")
        finalize_stream_file(temp_path, output_path, len(data))
        return

    logger.info(f"Dataset {dataset_name}: Pending items: {len(pending_data)}")

    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    futures = []

    for idx, item in pending_data:
        future = executor.submit(
            process_single_item_stream,
            item=item,
            item_idx=idx,
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

    executor.shutdown(wait=True)
    finalize_stream_file(temp_path, output_path, len(data))
    logger.info(f"========== Finish processing dataset: {dataset_name} ==========\n")

@atexit.register
def cleanup_resources():
    global req_session
    try:
        if req_session:
            req_session.close()
            logger.info("Requests session closed")
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker.kill()
        except:
            pass
    except Exception as e:
        logger.warning(f"Cleanup failed: {str(e)}")

def main():
    try:
        if not check_service_health():
            logger.error(f"Service unavailable ({BASE_URL}), start service first!")
            return
        
        logger.info(f"=== Config ===")
        logger.info(f"Test mode: {TEST_MODE} (Sample size: {TEST_SAMPLE_SIZE})")
        logger.info(f"Concurrency: Max={MAX_CONCURRENT}, Timeout={REQUEST_TIMEOUT}, Retries={RETRY_TIMES}")
        logger.info(f"Paths: Parquet={PARQUET_ROOT}, Image={IMAGE_ROOT}, Output={OUTPUT_ROOT}")
        logger.info(f"Datasets to process: {len(DATASETS)}")
        
        for ds in DATASETS:
            process_dataset_stream(ds)
        
        logger.info("All datasets processed successfully!")
    
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        global req_session
        try:
            if req_session:
                req_session.close()
                logger.info("Requests session closed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    main()