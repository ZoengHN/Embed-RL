import logging
import PIL
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from transformers.image_utils import ChannelDimension
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from src.utils.basic_utils import print_master

logger = logging.getLogger(__name__)

QWEN3_VL = 'qwen3_vl'

MODEL2BACKBONE = {
    'qwen3_vl': QWEN3_VL,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

VLM_IMAGE_TOKENS = {
    QWEN3_VL: "<|image_pad|>",
}

VLM_VIDEO_TOKENS = {
    QWEN3_VL: "<|video_pad|>",
}

from transformers import Qwen3VLForConditionalGeneration
backbone2model = {
    QWEN3_VL: Qwen3VLForConditionalGeneration,
}

import importlib.util
import sys
import os
_qwen2_vision_process_path = os.path.join(
    os.path.dirname(__file__),
    "vlm_backbone", "qwen3_vl", "qwen2_vision_process.py"
)
_spec = importlib.util.spec_from_file_location("qwen2_vision_process_custom", _qwen2_vision_process_path)
qwen2_vision_process = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qwen2_vision_process)


def load_processor(model_args, data_args=None):
    model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
    print_master(f'Loading processor from: {model_name_or_path}')
    
    from transformers import AutoProcessor, AutoTokenizer, AutoConfig
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )

    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if hasattr(config, 'emb_token_ids') and config.emb_token_ids:
            emb_token_id = config.emb_token_ids[0]
            temp_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            emb_token = temp_tokenizer.convert_ids_to_tokens(emb_token_id)
            if hasattr(processor, 'tokenizer'):
                if emb_token not in processor.tokenizer.get_vocab():
                    processor.tokenizer.add_tokens([emb_token])
        else:
            emb_token = "<emb>"
            if hasattr(processor, 'tokenizer'):
                if emb_token not in processor.tokenizer.get_vocab():
                    processor.tokenizer.add_tokens([emb_token])
    except Exception as e:
        emb_token = "<emb>"
        if hasattr(processor, 'tokenizer'):
            if emb_token not in processor.tokenizer.get_vocab():
                processor.tokenizer.add_tokens([emb_token])

    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        processor.tokenizer = tokenizer
        processor.tokenizer.padding_side = "left"

    if data_args is not None:
        if hasattr(processor, 'image_processor'):
            if data_args.resize_use_processor:
                processor.image_processor.do_resize = True
                if hasattr(processor.image_processor, 'min_pixels'):
                    processor.image_processor.min_pixels = data_args.resize_min_pixels
                if hasattr(processor.image_processor, 'max_pixels'):
                    processor.image_processor.max_pixels = data_args.resize_max_pixels

    return processor


def get_backbone_name(hf_config, model_type=None):
    if model_type is not None:
        setattr(hf_config, 'model_type', model_type)
    assert hf_config.model_type in SUPPORTED_MODELS
    return MODEL2BACKBONE[hf_config.model_type]


def process_cot_with_visual_content(cot_text, images, cot_mode, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS):
    import re
    import json
    from PIL import Image
    if cot_text is None or not cot_text.strip():
        return None
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', cot_text, re.DOTALL)
    if not thinking_match:
        return [{"type": "text", "text": cot_text}]
    thinking_content = thinking_match.group(1)
    thinking_end_pos = thinking_match.end()
    json_match = re.search(r'\{[^{}]*\}', thinking_content, re.DOTALL)
    if not json_match:
        return [{"type": "text", "text": cot_text}]
    try:
        json_data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return [{"type": "text", "text": cot_text}]
    cropped_visuals = []
    if 'bbox_2d' in json_data and images is not None:
        bboxes = json_data['bbox_2d']
        if bboxes and len(bboxes) > 0:
            if isinstance(images, list) and len(images) > 0:
                original_image = images[0]
            else:
                original_image = images
            if isinstance(original_image, Image.Image):
                width, height = original_image.size
                for bbox in bboxes:
                    if len(bbox) >= 4:
                        x1 = int(bbox[0] * width / 1000)
                        y1 = int(bbox[1] * height / 1000)
                        x2 = int(bbox[2] * width / 1000)
                        y2 = int(bbox[3] * height / 1000)
                        x1 = max(0, min(x1, width))
                        y1 = max(0, min(y1, height))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        crop_width = x2 - x1
                        crop_height = y2 - y1
                        MIN_CROP_SIZE = 10
                        MAX_ASPECT_RATIO = 50
                        if crop_width >= MIN_CROP_SIZE and crop_height >= MIN_CROP_SIZE:
                            aspect_ratio = max(crop_width, crop_height) / min(crop_width, crop_height)
                            if aspect_ratio <= MAX_ASPECT_RATIO:
                                try:
                                    cropped_img = original_image.crop((x1, y1, x2, y2))
                                    cropped_visuals.append(cropped_img)
                                except Exception as e:
                                    pass
    elif 'key_frames' in json_data and images is not None:
        key_frames = json_data['key_frames']
        if key_frames and len(key_frames) > 0 and isinstance(images, list):
            for frame_idx in key_frames:
                if 0 <= frame_idx < len(images):
                    cropped_visuals.append(images[frame_idx])
    result = []
    result.append({"type": "text", "text": f"<thinking>{thinking_content}</thinking>\n"})
    for img in cropped_visuals:
        result.append({
            "type": "image",
            "image": img,
            "min_pixels": IMAGE_MIN_PIXELS,
            "max_pixels": IMAGE_MAX_PIXELS,
        })
    remaining_text = cot_text[thinking_end_pos:].strip()
    if remaining_text:
        result.append({"type": "text", "text": remaining_text})
    return result


def Qwen3_VL_process_fn(model_inputs: dict, processor, max_length=None, data_args=None):
    process_vision_info = qwen2_vision_process.process_vision_info
    VIDEO_MIN_PIXELS = qwen2_vision_process.VIDEO_MIN_PIXELS
    VIDEO_MAX_PIXELS = qwen2_vision_process.VIDEO_MAX_PIXELS
    VIDEO_TOTAL_PIXELS = qwen2_vision_process.VIDEO_TOTAL_PIXELS
    IMAGE_MIN_PIXELS = qwen2_vision_process.MIN_PIXELS
    IMAGE_MAX_PIXELS = qwen2_vision_process.MAX_PIXELS
    IMAGE_FACTOR = qwen2_vision_process.IMAGE_FACTOR
    FPS = qwen2_vision_process.FPS
    FPS_MAX_FRAMES = qwen2_vision_process.FPS_MAX_FRAMES
    FPS_MIN_FRAMES = qwen2_vision_process.FPS_MIN_FRAMES

    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    cots = model_inputs.get('cot', [None] * len(texts))

    cot_mode = 'use_mmcot'

    messages_list = []
    for text, images, cot in zip(texts, visual_inputs, cots):
        if text:
            clean_text = text.replace('<|image_pad|>', '').replace('<|video_pad|>', '').strip()
        else:
            clean_text = ""

        user_content = []

        if images is not None:
            if isinstance(images, list) and len(images) > 1:
                is_pil_list = isinstance(images[0], Image.Image)

                if is_pil_list:
                    if clean_text:
                        user_content.append({"type": "text", "text": clean_text})
                    for frame in images:
                        user_content.append({
                            "type": "image",
                            "image": frame,
                            "min_pixels": VIDEO_MIN_PIXELS,
                            "max_pixels": VIDEO_MAX_PIXELS,
                        })
                else:
                    if clean_text:
                        user_content.append({"type": "text", "text": clean_text})
                    video_content = {
                        "video": images,
                        "total_pixels": VIDEO_TOTAL_PIXELS,
                        "min_pixels": VIDEO_MIN_PIXELS,
                        "max_pixels": VIDEO_MAX_PIXELS,
                        "min_frames": FPS_MIN_FRAMES,
                        "max_frames": FPS_MAX_FRAMES,
                        "fps": FPS,
                    }
                    user_content.append(video_content)

                if cot:
                    cot_contents = process_cot_with_visual_content(
                        cot_text=cot,
                        images=images,
                        cot_mode=cot_mode,
                        IMAGE_MIN_PIXELS=VIDEO_MIN_PIXELS,
                        IMAGE_MAX_PIXELS=VIDEO_MAX_PIXELS
                    )
                    if cot_contents:
                        user_content.extend(cot_contents)
                user_content.append({"type": "text", "text": "Summarize the above content in one word:"})

            else:
                if isinstance(images, list) and len(images) == 1:
                    image = images[0]
                else:
                    image = images

                user_content.append({
                    "type": "image",
                    "image": image,
                    "min_pixels": IMAGE_MIN_PIXELS,
                    "max_pixels": IMAGE_MAX_PIXELS,
                })
                if clean_text:
                    user_content.append({"type": "text", "text": clean_text})
                if cot:
                    cot_contents = process_cot_with_visual_content(
                        cot_text=cot,
                        images=image,
                        cot_mode=cot_mode,
                        IMAGE_MIN_PIXELS=IMAGE_MIN_PIXELS,
                        IMAGE_MAX_PIXELS=IMAGE_MAX_PIXELS
                    )
                    if cot_contents:
                        user_content.extend(cot_contents)
                user_content.append({"type": "text", "text": "Summarize the above content in one word:"})
        else:
            if clean_text:
                user_content.append({"type": "text", "text": clean_text})
                if cot:
                    cot_contents = process_cot_with_visual_content(
                        cot_text=cot,
                        images=None,
                        cot_mode=cot_mode,
                        IMAGE_MIN_PIXELS=IMAGE_MIN_PIXELS,
                        IMAGE_MAX_PIXELS=IMAGE_MAX_PIXELS
                    )
                    if cot_contents:
                        user_content.extend(cot_contents)
                user_content.append({"type": "text", "text": "Summarize the above content in one word:"})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>"}]},
        ]
        messages_list.append(messages)

    formatted_texts = []
    for messages in messages_list:
        if hasattr(processor, 'tokenizer'):
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        formatted_texts.append(text)

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages_list, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
    )

    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        if all(m is None for m in video_metadatas):
            video_metadatas = None
            video_kwargs = {}
    else:
        video_metadatas = None
        video_kwargs = {}

    if video_inputs is not None or image_inputs is not None:
        processor_kwargs = {
            "text": formatted_texts, "padding": True, "return_tensors": "pt",
        }
        if image_inputs is not None:
            processor_kwargs["images"] = image_inputs
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs
            if video_metadatas is not None:
                processor_kwargs["video_metadata"] = video_metadatas
                processor_kwargs.update(video_kwargs)
        model_inputs = processor(** processor_kwargs)
    else:
        model_inputs = processor(
            text=formatted_texts, padding=True, truncation=True,
            max_length=max_length if max_length else 8192, return_tensors="pt",
        )

    return {
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'pixel_values': model_inputs.get('pixel_values'),
        'image_grid_thw': model_inputs.get('image_grid_thw'),
        'pixel_values_videos': model_inputs.get('pixel_values_videos'),
        'video_grid_thw': model_inputs.get('video_grid_thw'),
        'texts': texts,
        'images': visual_inputs,
    }


process_vlm_inputs_fns = {
    QWEN3_VL: Qwen3_VL_process_fn,
}