import json
import os
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def load_cot_data(cot_dir: str, dataset_name: str, modality: str = "Image") -> Optional[Dict]:
    query_index = {}
    corpus_index = {"text": {}, "image": {}, "video": {}}

    if modality == "VisDo":
        query_cot_file = os.path.join(cot_dir, modality, f"{dataset_name}_query.json")
        corpus_cot_file = os.path.join(cot_dir, modality, f"{dataset_name}_corpus.json")
    elif modality == "Video":
        query_cot_file = os.path.join(cot_dir, modality, f"{dataset_name}_query_cot.json")
        corpus_cot_file = os.path.join(cot_dir, modality, f"{dataset_name}_corpus_cot.json")
    else:
        query_cot_file = os.path.join(cot_dir, modality, f"{dataset_name}.json")
        corpus_cot_file = os.path.join(cot_dir, modality, f"{dataset_name}_corpus_cot.json")

    if os.path.exists(query_cot_file):
        try:
            with open(query_cot_file, 'r', encoding='utf-8') as f:
                query_data = json.load(f)

            if modality == "VisDo":
                for item in query_data:
                    query_text = item.get('query', '')
                    query_cot = item.get('query_cot', '')
                    if query_text and query_cot:
                        query_index[query_text] = query_cot
            elif modality == "Video":
                for item in query_data:
                    video_id = (item.get('video_id') or item.get('id') or item.get('videoID') or
                               item.get('video') or item.get('video_path') or item.get('input_frames', ''))
                    caption = item.get('caption') or item.get('sentence') or item.get('enCap') or item.get('query') or item.get('question', '')
                    query_cot = item.get('query_cot', '')

                    if query_cot:
                        if video_id:
                            query_index[str(video_id)] = query_cot
                            if '/' in str(video_id) and str(video_id).endswith(('.mp4', '.avi', '.webm')):
                                video_clip_name = os.path.basename(str(video_id)).replace('.mp4', '').replace('.avi', '').replace('.webm', '')
                                query_index[video_clip_name] = query_cot
                        if caption:
                            if isinstance(caption, list):
                                for cap in caption:
                                    if cap:
                                        query_index[str(cap)] = query_cot
                            else:
                                query_index[str(caption)] = query_cot
            else:
                for item in query_data:
                    qry_img_path = item.get('qry_img_path', '')
                    qry_text = item.get('qry_text', '')
                    query_cot = item.get('query_cot', '')

                    if query_cot:
                        if qry_img_path and qry_text:
                            combo_key = (qry_img_path, qry_text)
                            query_index[combo_key] = query_cot
                        if qry_img_path:
                            query_index[qry_img_path] = query_cot
                        if qry_text:
                            query_index[qry_text] = query_cot

            logger.info(f"Loaded query CoT: {len(query_index)} entries")
        except Exception as e:
            logger.error(f"Load query CoT failed: {e}")

    if os.path.exists(corpus_cot_file):
        try:
            with open(corpus_cot_file, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)

            if modality == "VisDo":
                for item in corpus_data:
                    corpus_id = item.get('corpus-id')
                    image_path = item.get('image_path', '')
                    corpus_cot = item.get('corpus_cot', '')
                    if corpus_cot:
                        if corpus_id is not None:
                            corpus_index["image"][str(corpus_id)] = corpus_cot
                        if image_path:
                            corpus_index["image"][image_path] = corpus_cot
            elif modality == "Video":
                for item in corpus_data:
                    corpus_id = item.get('corpus_id', '')
                    item_id = item.get('item_id', '')
                    corpus_content = item.get('corpus_content', '')
                    corpus_cot = item.get('corpus_cot', '')
                    if corpus_cot:
                        if corpus_id:
                            corpus_index["video"][str(corpus_id)] = corpus_cot
                        if item_id:
                            corpus_index["video"][str(item_id)] = corpus_cot
                        if corpus_content:
                            corpus_index["video"][str(corpus_content)] = corpus_cot
                            if '/' in str(corpus_content) and str(corpus_content).endswith('.mp4'):
                                transformed_path = str(corpus_content).replace('/', '_').replace('.mp4', '')
                                corpus_index["video"][transformed_path] = corpus_cot
            else:
                if isinstance(corpus_data, dict):
                    corpus_index["text"] = corpus_data.get("text", {})
                    corpus_index["image"] = corpus_data.get("image", {})

            logger.info(f"Loaded corpus CoT: text={len(corpus_index['text'])}, image={len(corpus_index['image'])}, video={len(corpus_index['video'])}")
        except Exception as e:
            logger.error(f"Load corpus CoT failed: {e}")

    if not any([query_index, corpus_index["text"], corpus_index["image"], corpus_index["video"]]):
        return None

    return {'query_index': query_index, 'corpus_index': corpus_index}


def add_cot_to_dataset(dataset, cot_dict: Optional[Dict], cot_key_field: str = 'query_image', modality: str = "Image"):
    if cot_dict is None:
        return dataset

    query_index = cot_dict.get('query_index', {})
    corpus_index = cot_dict.get('corpus_index', {"text": {}, "image": {}, "video": {}})

    def add_query_cot_fn(example):
        cots = []
        if modality == "VisDo":
            query_texts = example.get('query_text', [])
            query_texts = [query_texts] if not isinstance(query_texts, list) else query_texts
            for query_text in query_texts:
                cot_text = query_index.get(query_text)
                if not cot_text:
                    best_match_key = max((k for k in query_index if isinstance(k, str) and k in query_text), key=len, default=None)
                    cot_text = query_index.get(best_match_key)
                cots.append(cot_text)
        elif modality == "Video":
            query_texts = [example.get('query_text', [])] if not isinstance(example.get('query_text', []), list) else example.get('query_text', [])
            query_images = [example.get('query_image', [])] if not isinstance(example.get('query_image', []), list) else example.get('query_image', [])
            max_len = max(len(query_texts), len(query_images))
            query_texts += [None] * (max_len - len(query_texts))
            query_images += [None] * (max_len - len(query_images))

            for query_text, query_img_info in zip(query_texts, query_images):
                cot_text = query_index.get(query_text) if query_text else None
                if not cot_text and query_img_info and 'paths' in query_img_info:
                    first_path = query_img_info['paths'][0]
                    path_parts = first_path.split('/')
                    if len(path_parts) >=3 and path_parts[-2] in ['query', 'positive_clip', 'negative_clip']:
                        cot_text = query_index.get(path_parts[-3])
                    elif 'video_frames' in first_path:
                        for i, part in enumerate(path_parts):
                            if part == 'video_frames' and i+1 < len(path_parts):
                                cot_text = query_index.get(path_parts[i+1])
                                break
                    elif len(path_parts)>=2:
                        cot_text = query_index.get(path_parts[-2])
                if not cot_text and query_text and "Question: " in query_text:
                    raw_question = query_text.split("Question: ")[-1].split('\n')[0].strip()
                    cot_text = query_index.get(raw_question)
                if not cot_text and query_text:
                    best_match_key = max((k for k in query_index if isinstance(k, str) and k in query_text), key=len, default=None)
                    cot_text = query_index.get(best_match_key)
                cots.append(cot_text)
        else:
            query_texts = [example.get('query_text', [])] if not isinstance(example.get('query_text', []), list) else example.get('query_text', [])
            query_images = [example.get('query_image', [])] if not isinstance(example.get('query_image', []), list) else example.get('query_image', [])
            max_len = max(len(query_texts), len(query_images))
            query_texts += [None] * (max_len - len(query_texts))
            query_images += [None] * (max_len - len(query_images))

            for query_text, img_info in zip(query_texts, query_images):
                cot_text = None
                img_path = img_info['paths'][0] if (img_info and 'paths' in img_info and img_info['paths']) else None

                if img_path and query_text:
                    combo_key = (img_path, query_text)
                    cot_text = query_index.get(combo_key)
                    if not cot_text:
                        img_basename = os.path.basename(img_path)
                        cot_text = query_index.get((img_basename, query_text))
                if not cot_text and img_path:
                    cot_text = query_index.get(img_path)
                    if not cot_text:
                        img_filename = os.path.basename(img_path)
                        cot_text = next((query_index[k] for k in query_index if not isinstance(k, tuple) and img_filename in str(k)), None)
                if not cot_text and query_text:
                    cot_text = query_index.get(query_text)
                    if not cot_text:
                        best_match_key = max((k for k in query_index if isinstance(k, str) and k in query_text), key=len, default=None)
                        cot_text = query_index.get(best_match_key)
                cots.append(cot_text)

        example['query_cot'] = cots if cots else [None]
        return example

    def add_cand_cot_fn(example):
        cots = []
        if modality == "VisDo":
            dataset_info = example.get('dataset_infos', {})
            cand_names = [dataset_info.get('cand_names', dataset_info.get('cand_name', []))] if not isinstance(dataset_info.get('cand_names', []), list) else dataset_info.get('cand_names', [])
            for corpus_id in cand_names:
                cots.append(corpus_index["image"].get(str(corpus_id)))
        elif modality == "Video":
            dataset_info = example.get('dataset_infos', {})
            cand_names = [dataset_info.get('cand_names', dataset_info.get('cand_name', []))] if not isinstance(dataset_info.get('cand_names', []), list) else dataset_info.get('cand_names', [])
            for cand_name in cand_names:
                cot_text = corpus_index["video"].get(str(cand_name))
                if not cot_text and str(cand_name).strip().startswith('('):
                    raw_text = re.sub(r'^\([A-Z]\)\s*', '', str(cand_name))
                    cot_text = corpus_index["video"].get(raw_text)
                if not cot_text and '/' in str(cand_name):
                    path_parts = str(cand_name).split('/')
                    if len(path_parts)>=2:
                        cot_text = corpus_index["video"].get(f"{path_parts[-2]}/{path_parts[-1]}")
                    if not cot_text and 'video_frames' in str(cand_name):
                        for i, part in enumerate(path_parts):
                            if part == 'video_frames' and i+1 < len(path_parts):
                                cot_text = corpus_index["video"].get(path_parts[i+1])
                                break
                cots.append(cot_text)
        else:
            cand_texts = [example.get('cand_text', [])] if not isinstance(example.get('cand_text', []), list) else example.get('cand_text', [])
            cand_images = [example.get('cand_image', [])] if not isinstance(example.get('cand_image', []), list) else example.get('cand_image', [])
            for cand_text, cand_img_info in zip(cand_texts, cand_images):
                cot_text = corpus_index["text"].get(cand_text) if cand_text else None
                if not cot_text and cand_img_info and 'paths' in cand_img_info:
                    cand_img_path = cand_img_info['paths'][0]
                    cot_text = corpus_index["image"].get(cand_img_path)
                    if not cot_text:
                        for p in corpus_index["image"]:
                            if cand_img_path.endswith(p) or p.endswith(cand_img_path):
                                cot_text = corpus_index["image"][p]
                                break
                    if not cot_text:
                        cand_filename = os.path.basename(cand_img_path)
                        for p in corpus_index["image"]:
                            if os.path.basename(p) == cand_filename:
                                cot_text = corpus_index["image"][p]
                                break
                cots.append(cot_text)

        example['cand_cot'] = cots if cots else [None]
        return example

    if 'query' in cot_key_field:
        dataset = dataset.map(add_query_cot_fn)
    else:
        dataset = dataset.map(add_cand_cot_fn)

    return dataset