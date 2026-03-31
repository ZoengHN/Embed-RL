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
from typing import Dict, List, Optional, Any, Tuple, Union
import tempfile
import shutil
from threading import Lock
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io

BASE_URL = "http://0.0.0.0:22005"
ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODEL_PATH = "/model/Qwen/Qwen3-VL-8B-Instruct"

VISUALDOC_ROOT = "/data/vlm2vec_eval/VisualDoc"
OUTPUT_ROOT = "/data/vlm2vec_eval/cot/VisDo"
IMAGE_SAVE_DIR = "/data/vlm2vec_eval/VisualDoc/images_cache"

MAX_CONCURRENT = 32
REQUEST_TIMEOUT = (10, 120)
RETRY_TIMES = 2
RETRY_DELAY = 1

TEST_MODE = False
TEST_SAMPLE_SIZE = 4

STREAM_SAVE_LOCK = Lock()
TEMP_SUFFIX = ".tmp"
BACKUP_INTERVAL = 10000

VISUALDOC_DATASETS = {
    "ViDoRe_arxivqa": ("vidore/arxivqa_test_subsampled_beir", None, "test"),
    "ViDoRe_docvqa": ("vidore/docvqa_test_subsampled_beir", None, "test"),
    "ViDoRe_infovqa": ("vidore/infovqa_test_subsampled_beir", None, "test"),
    "ViDoRe_tabfquad": ("vidore/tabfquad_test_subsampled_beir", None, "test"),
    "ViDoRe_tatdqa": ("vidore/tatdqa_test_beir", None, "test"),
    "ViDoRe_shiftproject": ("vidore/shiftproject_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": ("vidore/syntheticDocQA_artificial_intelligence_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_energy": ("vidore/syntheticDocQA_energy_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_government_reports": ("vidore/syntheticDocQA_government_reports_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry": ("vidore/syntheticDocQA_healthcare_industry_test_beir", None, "test"),
    "VisRAG_ArxivQA": ("openbmb/VisRAG-Ret-Test-ArxivQA", None, "train"),
    "VisRAG_ChartQA": ("openbmb/VisRAG-Ret-Test-ChartQA", None, "train"),
    "VisRAG_MP-DocVQA": ("openbmb/VisRAG-Ret-Test-MP-DocVQA", None, "train"),
    "VisRAG_SlideVQA": ("openbmb/VisRAG-Ret-Test-SlideVQA", None, "train"),
    "VisRAG_InfoVQA": ("openbmb/VisRAG-Ret-Test-InfoVQA", None, "train"),
    "VisRAG_PlotQA": ("openbmb/VisRAG-Ret-Test-PlotQA", None, "train"),
    "ViDoSeek-doc": ("VLM2Vec/ViDoSeek", None, "test"),
    "ViDoSeek-page": ("VLM2Vec/ViDoSeek-page-fixed", None, "test"),
    "MMLongBench-doc": ("VLM2Vec/MMLongBench-doc", None, "test"),
    "MMLongBench-page": ("VLM2Vec/MMLongBench-page-fixed", None, "test"),
    "ViDoRe_esg_reports_human_labeled_v2": ("vidore/esg_reports_human_labeled_v2", None, "test"),
    "ViDoRe_biomedical_lectures_v2": ("vidore/biomedical_lectures_v2", "english", "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual": ("vidore/biomedical_lectures_v2", None, "test"),
    "ViDoRe_economics_reports_v2": ("vidore/economics_reports_v2", "english", "test"),
    "ViDoRe_economics_reports_v2_multilingual": ("vidore/economics_reports_v2", None, "test"),
    "ViDoRe_esg_reports_v2": ("vidore/esg_reports_v2", "english", "test"),
    "ViDoRe_esg_reports_v2_multilingual": ("vidore/esg_reports_v2", None, "test"),
}

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("cot_processing.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_serializable_data(data: Any) -> Any:
    if data is None:
        return None
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64,
                          np.unsignedinteger, np.uint8, np.uint16, np.uint32, np.uint64,
                          np.floating, np.float16, np.float32, np.float64)):
        return data.item()
    elif isinstance(data, (list, tuple)):
        return [clean_serializable_data(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_serializable_data(v) for k, v in data.items()}
    elif isinstance(data, set):
        return list(data)
    elif not isinstance(data, (str, int, float, bool)):
        try:
            return str(data)
        except:
            return ""
    else:
        return data

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

def init_stream_file(dataset_name: str, data_type: str) -> Tuple[str, set, str]:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    output_path = os.path.join(OUTPUT_ROOT, f"{dataset_name}_{data_type}.json")
    temp_path = output_path + TEMP_SUFFIX
    
    processed_ids = set()
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        id_key = "query-id" if data_type == "query" else "corpus-id"
                        if id_key in item:
                            processed_ids.add(item[id_key])
        except Exception as e:
            processed_ids = set()
            if os.path.exists(temp_path):
                shutil.move(temp_path, temp_path + f".bak_{int(time.time())}")
    
    return temp_path, processed_ids, output_path

def stream_save_item(item: Dict, temp_path: str, data_type: str, index: int):
    with STREAM_SAVE_LOCK:
        try:
            cleaned_item = clean_serializable_data(item)
            item_with_idx = {"_processing_index": index, **cleaned_item}
            
            with open(temp_path, "a", encoding="utf-8") as f:
                json.dump(item_with_idx, f, ensure_ascii=False)
                f.write("\n")
            
            if index % BACKUP_INTERVAL == 0 and index > 0:
                with open(temp_path, "r", encoding="utf-8") as f:
                    f.flush()
                shutil.copy2(temp_path, temp_path + f".backup_{int(time.time())}")
                
        except Exception as e:
            raise

def finalize_stream_file(temp_path: str, output_path: str, data_type: str, total_count: int):
    processed_items = []
    if os.path.exists(temp_path):
        with open(temp_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        processed_items.append(item)
                    except Exception as e:
                        continue
    
    id_key = "query-id" if data_type == "query" else "corpus-id"
    processed_items.sort(key=lambda x: x.get(id_key, 999999))
    final_items = []
    for item in processed_items[:total_count]:
        if "_processing_index" in item:
            del item["_processing_index"]
        final_items.append(item)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_items, f, ensure_ascii=False, indent=2)
    
    backup_path = temp_path + f".final_{int(time.time())}"
    shutil.move(temp_path, backup_path)

def save_hf_image(image_obj, image_id: str, dataset_name: str) -> Optional[str]:
    try:
        dataset_image_dir = os.path.join(IMAGE_SAVE_DIR, dataset_name)
        os.makedirs(dataset_image_dir, exist_ok=True)
        
        image_filename = f"{image_id}.png"
        image_path = os.path.join(dataset_image_dir, image_filename)
        
        if os.path.exists(image_path):
            return image_path
        
        img = None
        if isinstance(image_obj, Image.Image):
            img = image_obj
        elif isinstance(image_obj, dict):
            if "bytes" in image_obj:
                img_data = image_obj["bytes"]
                img = Image.open(io.BytesIO(img_data))
            elif "data" in image_obj:
                img_data = image_obj["data"]
                if isinstance(img_data, str):
                    img_data = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_data))
            elif "image" in image_obj:
                return save_hf_image(image_obj["image"], image_id, dataset_name)
            else:
                return None
        elif isinstance(image_obj, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_obj))
        else:
            return None
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(image_path, format="PNG", quality=95)
        return image_path
    
    except Exception as e:
        return None

def load_beir_component(dataset_name: str, component: str, split: str) -> List[Dict]:
    dataset_config = VISUALDOC_DATASETS[dataset_name]
    dataset_path = os.path.join(VISUALDOC_ROOT, dataset_config[0])
    
    component_dir = os.path.join(dataset_path, component)
    parquet_files = []
    
    for root, dirs, files in os.walk(component_dir):
        for file in files:
            if file.startswith(f"{split}-") and file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        return []
    
    df_list = []
    for file in sorted(parquet_files):
        try:
            df = pq.read_table(file).to_pandas()
            df_list.append(df)
        except Exception as e:
            continue
    
    if not df_list:
        return []
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    if TEST_MODE:
        combined_df = combined_df.head(TEST_SAMPLE_SIZE)
    
    combined_df = combined_df.fillna("")
    
    if component == "corpus" and "image" in combined_df.columns:
        combined_df["image_path"] = combined_df.apply(
            lambda row: save_hf_image(row["image"], row["corpus-id"], dataset_name) 
            if row["image"] is not None and row["image"] != "" else "",
            axis=1
        )
        combined_df = combined_df.drop(columns=["image"])
    elif component == "queries" and "image" in combined_df.columns:
        combined_df["image_path"] = combined_df.apply(
            lambda row: save_hf_image(row["image"], row["query-id"], dataset_name)
            if row["image"] is not None and row["image"] != "" else "",
            axis=1
        )
        combined_df = combined_df.drop(columns=["image"])
    
    for col in combined_df.columns:
        if combined_df[col].dtype == object:
            combined_df[col] = combined_df[col].apply(lambda x: clean_serializable_data(x))
        elif np.issubdtype(combined_df[col].dtype, np.number):
            combined_df[col] = combined_df[col].apply(
                lambda x: x.item() if (pd.notna(x) and isinstance(x, (np.integer, np.floating))) else x
            )
    
    data = combined_df.to_dict("records")
    data = [clean_serializable_data(item) for item in data]
    
    return data

def encode_image(image_path: str) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
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
        "top_p": 1.0,
        "stream": False
    }

def call_model_single(prompt: str, image_path: Optional[str] = None, dataset_name: str = "", item_type: str = "", item_id: str = "") -> Optional[str]:
    global req_session
    try:
        if not check_service_health():
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
            return None
        
        return result["choices"][0]["message"]["content"]
    
    except requests.exceptions.ConnectionError as e:
        time.sleep(RETRY_DELAY)
        try:
            response = req_session.post(ENDPOINT, json=request_data, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except:
            return None
    except requests.exceptions.Timeout as e:
        return None
    except Exception as e:
        return None

def process_single_query(item: Dict, item_idx: int, dataset_name: str, temp_path: str):
    new_item = item.copy()
    new_item["query_cot"] = ""
    new_item["error"] = ""
    
    for key in ["pos_cot", "pos_corpus_ids"]:
        if key in new_item:
            del new_item[key]
    
    try:
        query_text = item.get("query", "").strip()
        
        if not query_text:
            new_item["error"] = "Empty query content"
            stream_save_item(new_item, temp_path, "query", item_idx)
            return new_item
        
        query_image_path = item.get("image_path", "")
        if query_image_path and os.path.exists(query_image_path):
            prompt = PROMPT_IMAGE_WITHOUT_ANSWER.format(question=query_text)
            query_cot = call_model_single(prompt, query_image_path, dataset_name, "query", item["query-id"])
        else:
            prompt = PROMPT_TEXT_WITHOUT_ANSWER.format(question=query_text)
            query_cot = call_model_single(prompt, None, dataset_name, "query", item["query-id"])
        
        new_item["query_cot"] = query_cot if query_cot else ""
        stream_save_item(new_item, temp_path, "query", item_idx)
        return new_item
    
    except Exception as e:
        new_item["error"] = str(e)
        stream_save_item(new_item, temp_path, "query", item_idx)
        return new_item

def process_query_dataset(dataset_name: str):
    dataset_config = VISUALDOC_DATASETS[dataset_name]
    split = dataset_config[2]
    
    query_data = load_beir_component(dataset_name, "queries", split)
    if not query_data:
        return
    
    temp_path, processed_ids, output_path = init_stream_file(dataset_name, "query")
    
    pending_data = []
    for idx, item in enumerate(query_data):
        if item["query-id"] not in processed_ids:
            pending_data.append((idx, item))
    
    if not pending_data:
        finalize_stream_file(temp_path, output_path, "query", len(query_data))
        return
    
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    futures = []
    
    for idx, item in pending_data:
        future = executor.submit(
            process_single_query,
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
        except Exception as e:
            pass
    
    executor.shutdown(wait=True)
    finalize_stream_file(temp_path, output_path, "query", len(query_data))

def process_single_corpus(item: Dict, item_idx: int, dataset_name: str, temp_path: str):
    new_item = item.copy()
    new_item["corpus_cot"] = ""
    new_item["error"] = ""
    
    try:
        corpus_text = new_item.get("doc-id", "").strip() or new_item.get("company", "").strip() or ""
        corpus_image_path = new_item.get("image_path", "")
        pos_image_desc = f"Image path: {corpus_image_path}" if os.path.exists(corpus_image_path) else "Image not found"
        
        if corpus_image_path and os.path.exists(corpus_image_path):
            prompt = PROMPT_POS_IMAGE.format(pos_text=corpus_text, pos_image_description=pos_image_desc)
            corpus_cot = call_model_single(prompt, corpus_image_path, dataset_name, "corpus", item["corpus-id"])
        else:
            prompt = PROMPT_POS_TEXT.format(pos_text=corpus_text)
            corpus_cot = call_model_single(prompt, None, dataset_name, "corpus", item["corpus-id"])
        
        new_item["corpus_cot"] = corpus_cot if corpus_cot else ""
        stream_save_item(new_item, temp_path, "corpus", item_idx)
        return new_item
    
    except Exception as e:
        new_item["error"] = str(e)
        stream_save_item(new_item, temp_path, "corpus", item_idx)
        return new_item

def process_corpus_dataset(dataset_name: str):
    dataset_config = VISUALDOC_DATASETS[dataset_name]
    split = dataset_config[2]
    
    corpus_data = load_beir_component(dataset_name, "corpus", split)
    if not corpus_data:
        return
    
    temp_path, processed_ids, output_path = init_stream_file(dataset_name, "corpus")
    
    pending_data = []
    for idx, item in enumerate(corpus_data):
        if item["corpus-id"] not in processed_ids:
            pending_data.append((idx, item))
    
    if not pending_data:
        finalize_stream_file(temp_path, output_path, "corpus", len(corpus_data))
        return
    
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    futures = []
    
    for idx, item in pending_data:
        future = executor.submit(
            process_single_corpus,
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
        except Exception as e:
            pass
    
    executor.shutdown(wait=True)
    finalize_stream_file(temp_path, output_path, "corpus", len(corpus_data))

@atexit.register
def cleanup_resources():
    global req_session
    try:
        if req_session:
            req_session.close()
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker.kill()
        except:
            pass
    except Exception as e:
        pass

def main():
    try:
        if not check_service_health():
            return
        
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
        
        for ds_name in VISUALDOC_DATASETS.keys():
            process_query_dataset(ds_name)
            process_corpus_dataset(ds_name)
    
    except Exception as e:
        pass
    finally:
        global req_session
        try:
            if req_session:
                req_session.close()
        except Exception as e:
            pass

if __name__ == "__main__":
    main()