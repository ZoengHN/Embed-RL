# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLM2Vec Dataset for GRPO Training

This dataset loader handles multimodal data for training a VLM to generate
Chain-of-Thought (CoT) reasoning that produces better embeddings for retrieval.

Data format (JSON):
{
    "qry": {
        "conversations": [{"from": "human", "value": "question..."}, ...],
        "video": ["frame1.jpg", "frame2.jpg", ...] or None,
        "image": "path/to/image.jpg" or None
    },
    "pos_text": "positive answer text" (for text+image datasets),
    "pos": {
        "conversations": [{"from": "gpt", "value": "answer..."}, ...],
        "video": ["frame1.jpg", "frame2.jpg", ...]
    } (for video retrieval datasets)
}
"""

import copy
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.getLogger(__name__)


# Prompt templates for CoT generation

PROMPT_TEXT_WITHOUT_ANSWER = """You are a visual reasoning assistant specialized in text-to-image retrieval. Given a text description and a task, analyze the text content to determine the key visual concepts needed for retrieving matching images.

**Rules:**
1.  Keep reasoning concise and grounded in textual evidence. Limit each step to 1-2 sentences.
2.  Base your reasoning solely on the textual content and the task description.
3.  Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.
4.  In your thinking process, you must extract and output key visual concepts from the text description. Use JSON format with key 'text_keywords' to specify the important keywords as a list.
5.  First, think between `<thinking>` and `</thinking>` while output necessary keywords from the text in JSON with key 'text_keywords'. Then, based on the thinking contents, rethink between `<rethink>` and `</rethink>`. Finally, output the answer within `<answer>...</answer>`.
6.  Your thought process should adapt to the task type:
    *   For **caption-based retrieval** (e.g., find image from caption): Extract key visual elements, objects, scenes, and relationships.
    *   For **news retrieval** (e.g., find news image from headline): Identify key people, locations, events, and contextual elements.
    *   For **dialogue-based retrieval** (e.g., find image from conversation): Summarize visual attributes, actions, and scene details.
    *   For **question-based retrieval** (e.g., find factual image): Identify the key concepts and relationships.

**Example :**
QUESTION: Represent the provided dialogue related to an image for the purpose of image retrieval: Q:is this a child or adult A:adult Q:male or female A:male Q:are they inside or outside A:inside Q:are they laying on the floor A:yes, but there is a blanket in between them and the floor Q:is the floor carpeted or wooden A:it is tile Q:what color is the blanket A:red and white Q:what color is the tile A:orange red Q:what breed is the dog A:boxer Q:does the dog look healthy and happy A:yes Q:what color is the dog A:tan
ANSWER:
<thinking>The dialogue describes an adult male lying on a tile floor with a red and white blanket underneath. There's also a tan boxer dog that looks healthy and happy. The scene is indoors. {{"text_keywords": ["adult male", "lying on floor", "tile floor", "red and white blanket", "tan boxer dog", "healthy dog", "indoors"]}}</thinking>
<rethink>The keywords indicate we need an image showing a man lying on a tile floor with a blanket, accompanied by a tan boxer dog in an indoor setting.</rethink>
<answer>An image of an adult male lying on a tile floor with a red and white blanket, accompanied by a tan boxer dog in an indoor environment.</answer>

**Example :**
QUESTION: Based on the provided <|image_1|>, search for a Wikipedia image that answers the question: Which body part is found on both a 1913 D Barber half and a 1914 Barber Quarter?
ANSWER:
<thinking>The question asks about a body part common to two specific coins: the 1913 D Barber half and the 1914 Barber Quarter. These are US coins featuring a design with a head/profile. The body part would likely be the head or face. {{"text_keywords": ["1913 D Barber half", "1914 Barber Quarter", "US coins", "body part", "head", "profile", "coin design"]}}</thinking>
<rethink>The keywords suggest we need an image showing the head/profile design common to both Barber coins.</rethink>
<answer>An image showing the head or profile design featured on both the 1913 D Barber half dollar and the 1914 Barber Quarter.</answer>

Now, process the following input:
TASK/QUESTION: {question}

Follow the specified format for your response. Analyze the text and task/question to determine the appropriate image description."""


PROMPT_IMAGE_WITHOUT_ANSWER = """You are a visual reasoning assistant. Given an image and a task description or question, analyze the image step-by-step to produce the required output. The task may involve image retrieval, classification, question answering, or object identification.

**Rules:**
1.  Keep reasoning concise and grounded in visual evidence. Limit each step to 1-2 sentences.
2.  Base your reasoning solely on the visual content of the image and the task description.
3.  Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.
4.  In your thinking process, you must output coordinates for the key visual element(s) relevant to answering the question (do not use the entire image as a bounding box). Use JSON format with key 'bbox_2d' to specify the bounding box as [x1, y1, x2, y2]. For multiple elements, use a list of bboxes: [[x1, y1, x2, y2], [x1, y1, x2, y2]].
5.  First, think between `<thinking>` and `</thinking>` while output necessary coordinates needed to answer the question in JSON with key 'bbox_2d'. Then, based on the thinking contents and coordinates, rethink between `<rethink>` and `</rethink>`. Finally, output the answer within `<answer>...</answer>`.
6.  Your thought process should adapt to the task:
    *   For **retrieval** (e.g., find similar image): Identify and locate key visual elements that define the match.
    *   For **classification** (e.g., scene, object, domain): Locate distinguishing visual features that belong to the class.
    *   For **question answering**: Locate the visual or textual clues in the image that lead to the answer.
    *   For **object identification/segmentation**: Provide the object's location and boundaries.

**Example :**
QUESTION: What animal is prominently featured in the center of the image?
ANSWER:
<thinking>The image shows a large brown bear standing in a forest clearing at coordinates [220, 150, 380, 320]. The bear is the central subject of the image. {{"bbox_2d": [220, 150, 380, 320]}}</thinking>
<rethink>The coordinates confirm the bear's central position and prominent size in the image.</rethink>
<answer>A brown bear is prominently featured in the center of the image.</answer>

**Example :**
QUESTION: Identify both vehicles visible in the street scene.
ANSWER:
<thinking>The image shows a red car at coordinates [120, 200, 280, 280] and a blue truck at coordinates [320, 180, 480, 300]. Both vehicles are on the same street. {{"bbox_2d": [[120, 200, 280, 280], [320, 180, 480, 300]]}}</thinking>
<rethink>The two bounding boxes correspond to the red car and the blue truck, both clearly visible in the street scene.</rethink>
<answer>The street scene contains a red car and a blue truck.</answer>

Now, process the following input:
TASK/QUESTION: {question}

Analyze the PROVIDED IMAGE (not just the metadata) to complete the task."""

PROMPT_POS_IMAGE = """You are a visual reasoning assistant. Given a task description and a positive image output (with optional text template), analyze the task to determine what the image should contain and locate key regions.

**Rules:**
1. Keep reasoning concise and grounded in the task description. Limit each step to 1-2 sentences.
2. Use the task description to infer what the target image should look like.
3. In your thinking process, you must output coordinates for the key visual element(s) relevant to answering the question. If no question is specified, output the most important element. Use JSON format with key 'bbox_2d' to specify the bounding box as [x1, y1, x2, y2]. For multiple elements, use a list of bboxes: [[x1, y1, x2, y2], [x1, y1, x2, y2]].
4. First, think between `<thinking>` and `</thinking>` while output necessary coordinates in JSON with key 'bbox_2d'. Then, based on the thinking contents, rethink between `<rethink>` and `</rethink>`. Finally, output a brief description of the image within `<answer>...</answer>`.

**Example:**
TASK INFO: Retrieve the corresponding image.
POSITIVE TEXT OUTPUT: An image of a shop featuring teddy bears, with one placed as a door guard and another positioned above it.
ANSWER:
<thinking>The task is to retrieve an image matching the description of a teddy bear shop. The description specifies two key teddy bears: one acting as a "door guard" (likely at the base of a doorway) and another positioned directly above it. To pinpoint these elements, I should place a bounding box around the area where the "door guard" teddy bear is located, as it's a specific positional landmark. {{"bbox_2d": [280, 350, 360, 430]}}</thinking>
<rethink>The bounding box focuses on the described "door guard" teddy bear, which is a critical and specific visual anchor in the scene. The presence of the second bear above it would be inferred from the overall image context.</rethink>
<answer>The image should show a teddy bear shop with a door guard teddy and a neighbor teddy above.</answer>

QUESTION: Locate the sink and the refrigerator in the kitchen image.
ANSWER:
<thinking>The image depicts a modern kitchen. A stainless steel kitchen sink is installed on the left countertop, and a large silver refrigerator stands on the right side. I need to identify both objects. The sink is at coordinates [80, 200, 240, 320], and the refrigerator is at [350, 150, 520, 380]. {{"bbox_2d": [[80, 200, 240, 320], [350, 150, 520, 380]]}}</thinking>
<rethinking>The two bounding boxes correctly correspond to the sink (left) and the refrigerator (right), both clearly visible and distinct in the kitchen layout.</rethinking>
<answer>The sink is on the left countertop, and the refrigerator is on the right side of the kitchen.</answer>


Now, process the following input:
POSITIVE TEXT OUTPUT: {pos_text}
POSITIVE IMAGE OUTPUT: {pos_image_description} 

Analyze the PROVIDED IMAGE (not just the metadata) to complete the task."""


PROMPT_POS_TEXT = """You are a visual reasoning assistant. Given a task description and a positive text output, analyze the text to determine the key concepts that make it the correct output.

**Rules:**
1. Keep reasoning concise and grounded in the textual evidence. Limit each step to 1-2 sentences.
2. Use the task description to understand the context of the output.
3. In your thinking process, you must extract and output key concepts from the text. Use JSON format with key 'text_keywords' to specify the important keywords as a list.
4. First, think between `<thinking>` and `</thinking>` while output necessary keywords from the text in JSON with key 'text_keywords'. Then, based on the thinking contents, rethink between `<rethink>` and `</rethink>`. Finally, output a brief description of the text within `<answer>...</answer>`.

**Example:**
TASK INFO: Retrieve the corresponding text.
POSITIVE TEXT OUTPUT: 1/8/93
ANSWER:
<thinking>The task is to answer a question about a date in a document. The positive text output is "1/8/93", which represents a date. Key concepts include the date format and the specific numbers. {{"text_keywords": ["1/8/93", "date", "January 8, 1993"]}}</thinking>
<rethink>The keywords indicate that the answer is a date in a specific format, which is likely found in a document.</rethink>
<answer>The text output is the date January 8, 1993.</answer>

Now, process the following input:
POSITIVE TEXT OUTPUT: {pos_text}

Follow the specified format for your response."""


PROMPT_VIDEO_WITHOUT_ANSWER = """You are a video reasoning assistant. Given a video sequence (multiple frames) and a task description, analyze the video content step-by-step to produce the required output. The task may involve video captioning, video question answering, or video retrieval.

**Rules:**
1.  Keep reasoning concise and grounded in visual evidence from the video frames. Limit each step to 1-2 sentences.
2.  Base your reasoning solely on the visual content of the video frames and the task description.
3.  Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.
4.  In your thinking process, you must identify and output key frames from the video sequence that are most relevant to answering the question. Use JSON format with key 'key_frames' to specify the frame indices as a list (using 1-based indexing).
5.  First, think between `<thinking>` and `</thinking>` while output necessary key frame indices in JSON with key 'key_frames'. Then, based on the thinking contents, rethink between `<rethink>` and `</rethink>`. Finally, output the answer within `<answer>...</answer>`.
6.  Your thought process should adapt to the task type:
    *   For **video captioning/description**: Identify frames that show main events, transitions, or key moments in the video sequence.
    *   For **video question answering**: Locate frames that contain the visual evidence needed to answer the specific question.
    *   For **video retrieval**: Identify frames that represent the core content or distinguishing features of the video.

**Example :**
QUESTION: Provide a comprehensive analysis of the video's content and themes.
ANSWER:
<thinking>The video shows a cooking process. Frame 1 shows water boiling in a pot. Frame 4 shows a hand adding noodles to the pot. Frame 8 shows children eating the cooked noodles. These frames capture the beginning, middle, and end of the cooking process. {{"key_frames": [1, 4, 8]}}</thinking>
<rethink>The key frames (1, 4, 8) represent the main stages: preparation, cooking, and consumption of the noodles.</rethink>
<answer>The video depicts the process of preparing instant noodles, starting with boiling water, adding and cooking the noodles, and ending with children enjoying the meal.</answer>

**Example :**
QUESTION: How would you describe the design of the architectural structure in the video?
ANSWER:
<thinking>The video shows an unusual building. Frame 3 clearly displays the structure's tilted and angular design. Frame 6 shows another angle of the same structure. {{"key_frames": [3, 6]}}</thinking>
<rethink>Frames 3 and 6 show the architectural structure from different angles, both revealing its non-conventional design.</rethink>
<answer>The architectural structure has a tilted and angular design that deviates from traditional building structures.</answer>

Now, process the following input:
VIDEO FRAMES: {video}
TASK/QUESTION: {question}

Follow the specified format for your response. Analyze the video frames and task/question to determine the appropriate answer."""

PROMPT_VIDEO_RETRIEVAL_WITHOUT_ANSWER = """You are a visual reasoning assistant specialized in text-to-video retrieval. Given a video description and a task, analyze the text content to determine the key visual concepts needed for retrieving matching video clips or keyframes.

**Rules:**
1.  Keep reasoning concise and grounded in textual evidence. Limit each step to 1-2 sentences.
2.  Base your reasoning solely on the textual content and the task description.
3.  Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.
4.  In your thinking process, you must extract and output key visual concepts from the text description. Use JSON format with key 'text_keywords' to specify the important keywords as a list.
5.  First, think between `<thinking>` and `</thinking>` while output necessary keywords from the text in JSON with key 'text_keywords'. Then, based on the thinking contents, rethink between `<rethink>` and `</rethink>`. Finally, output the answer within `<answer>...</answer>`.
6.  Your thought process should adapt to the task type:
    *   For **video-based retrieval** (e.g., find video from description): Extract key visual elements, objects, scenes, actions, temporal sequences, and relationships.
    *   For **scene-based retrieval** (e.g., find video from scene description): Identify key people, locations, events, contextual elements, and temporal progression.
    *   For **action-based retrieval** (e.g., find video from action sequence): Summarize visual attributes, actions, scene details, and temporal dynamics.

**Example :**
QUESTION: Represent the provided video description for the purpose of video retrieval: The video opens with a shot of an individual in an orange safety suit and a white helmet, performing cleaning or maintenance duties on a stairway of the vessel. The railing is yellow, and the flooring consists of metal grates. The surrounding area has a bright green deck, and the water is visible to the side, indicating that the vessel is either docked or stationed at sea. Subsequent scenes shift to different parts of the ship, where other crew members can be seen wearing similar safety gear. The deck is consistently green with yellow railings, and the infrastructure includes various industrial components such as pipes, storage tanks, ladders, and safety signage.
ANSWER:
<thinking>The video description depicts a maritime industrial environment with crew performing maintenance. Key elements include individuals in orange safety suits and white helmets, a ship with yellow railings and metal grate stairway, bright green deck, visible water, and various industrial components like pipes, storage tanks, ladders, and safety signage. {{"text_keywords": ["orange safety suit", "white helmet", "cleaning/maintenance", "ship stairway", "yellow railing", "metal grate flooring", "green deck", "visible water", "crew members", "maritime industrial", "pipes", "storage tanks", "ladders", "safety signage"]}}</thinking>
<rethink>The keywords suggest we need a video showing maintenance activities on an industrial ship, with crew in distinctive safety gear, featuring specific colors (orange, white, yellow, green) and industrial infrastructure.</rethink>
<answer>A video depicting crew members in orange safety suits and white helmets performing maintenance on an industrial ship with green deck, yellow railings, and various industrial components.</answer>

Now, process the following input:
TASK/QUESTION: {question}

Follow the specified format for your response. Analyze the text and task/question to determine the appropriate video description."""


PROMPT_POS_VIDEO_RETRIEVAL = """You are a visual reasoning assistant for video. Given a task description and a set of video frames (provided as image inputs), analyze the video frames to determine the key moments or concepts that make it the correct output for the task.

**Rules:**
1.  Keep reasoning concise and grounded in the VISUAL EVIDENCE from the provided video frames. Limit each step to 1-2 sentences.
2.  Use the task description to understand the context of the output.
3.  In your thinking process, you must identify and output the indices of the most relevant or representative frames from the video sequence (using 1-based indexing). Use JSON format with key 'key_frames' to specify the frame indices as a list.
4.  First, think between `<thinking>` and `</thinking>` while outputting necessary key frame indices in JSON with key 'key_frames'. Then, based on the thinking contents, rethink between `<rethink>` and `</rethink>`. Finally, output a brief analysis of the video content within `<answer>...</answer>`.

**Example:**
TASK INFO: Provide an exhaustive description of the video content.
VIDEO FRAMES: 8 frames at 2.0 FPS (industrial ship maintenance scene)

ANSWER:
<thinking>The video frames show crew members in orange safety suits performing maintenance on a ship's stairway with yellow railings. Frame 1 shows the stairway and safety gear, Frame 3 shows close-up of metal grates, Frame 5 shows green deck and water, Frame 7 shows industrial pipes/tanks. These frames capture the core visual elements. {{"key_frames": [1, 3, 5, 7]}}</thinking>
<rethink>The key frames (1, 3, 5, 7) highlight the main setting (ship stairway) and key details (safety gear, industrial infrastructure, marine environment).</rethink>
<answer>The video shows crew in orange safety suits performing maintenance on a ship with yellow railings, green deck, and industrial components like pipes and storage tanks, with water visible in the background.</answer>

Now, process the following input:
TASK DESCRIPTION: {pos_text}
VIDEO FRAMES: {pos_video_output}

Analyze the PROVIDED VIDEO FRAMES (not just the metadata) to complete the task."""


PROMPT_POS_TEXT = """You are a visual reasoning assistant. Given a task description and a positive text output, analyze the text to determine the key concepts that make it the correct output.

**Rules:**
1. Keep reasoning concise and grounded in the textual evidence. Limit each step to 1-2 sentences.
2. Use the task description to understand the context of the output.
3. In your thinking process, you must extract and output key concepts from the text. Use JSON format with key 'text_keywords' to specify the important keywords as a list.
4. First, think between `<thinking>` and `</thinking>` while output necessary keywords from the text in JSON with key 'text_keywords'. Then, based on the thinking contents, rethink between `<rethink>` and `</rethink>`. Finally, output a brief description of the text within `<answer>...</answer>`.

**Example:**
TASK INFO: Retrieve the corresponding text.
POSITIVE TEXT OUTPUT: 1/8/93
ANSWER:
<thinking>The task is to answer a question about a date in a document. The positive text output is "1/8/93", which represents a date. Key concepts include the date format and the specific numbers. {{"text_keywords": ["1/8/93", "date", "January 8, 1993"]}}</thinking>
<rethink>The keywords indicate that the answer is a date in a specific format, which is likely found in a document.</rethink>
<answer>The text output is the date January 8, 1993.</answer>

Now, process the following input:
POSITIVE TEXT OUTPUT: {pos_text}

Follow the specified format for your response."""


def collate_vlm2vec_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of VLM2Vec samples with padding support.

    Following UME-R1's approach: interleave Query and Pos into a single batch for joint training.

    Input: B samples, each with query and pos data
    Output: 2*B samples interleaved: [qry1, pos1, qry2, pos2, ...]

    This allows:
    1. Agent Loop to rollout both Query and Pos
    2. Reward Model to compute contrastive learning rewards
    3. PPO to train both Query and Pos generation

    Note: pixel_values and image_grid_thw are treated as non-tensors because:
    1. Not all samples may have visual inputs (if image/video loading fails)
    2. They have variable shapes across the batch

    For non-tensor fields, we ensure the array length equals batch_size by filling None for missing values.
    """
    import torch.nn.functional as F

    # Filter out None values (skipped samples from __getitem__ due to overlong prompts)
    data_list = [data for data in data_list if data is not None]

    if len(data_list) == 0:
        raise ValueError("All samples in batch were filtered out (prompts too long)")

    original_batch_size = len(data_list)

    # Interleave Query and Pos: expand to 2*B samples
    interleaved_data_list = []
    for i, data in enumerate(data_list):
        # Extract ground_truth data
        ground_truth = data.get('reward_model', {}).get('ground_truth', {})

        # Query data
        query_item = {
            'input_ids': data['input_ids'],
            'attention_mask': data['attention_mask'],
            'position_ids': data['position_ids'],
            'raw_prompt': data['raw_prompt'],
            'raw_prompt_ids': data['raw_prompt_ids'],
            'multi_modal_data': data.get('multi_modal_data', {}),  # For vLLM vision inference
            'sample_type': 'query',  # Mark as query
            'pair_index': i,  # Track which pair this belongs to
            'data_source': data['data_source'],  # Required by reward manager
            # Ground truth data for query (needed by reward model)
            'query_base_text': ground_truth.get('query_base_text'),
            'query_image_paths': ground_truth.get('query_image_paths'),
            'query_video_paths': ground_truth.get('query_video_paths'),
            'pos_text': ground_truth.get('pos_text'),  # Still need pos info for pairing
            'pos_image_paths': ground_truth.get('pos_image_paths'),
            'pos_video_paths': ground_truth.get('pos_video_paths'),
            # Preserve reward_model structure for reward manager
            'reward_model': data.get('reward_model', {}),
        }
        # Add multimodal fields if present
        if 'pixel_values' in data:
            query_item['pixel_values'] = data['pixel_values']
        if 'image_grid_thw' in data:
            query_item['image_grid_thw'] = data['image_grid_thw']

        # Pos data
        pos_item = {
            'input_ids': data['pos_input_ids'],
            'attention_mask': data['pos_attention_mask'],
            'position_ids': data['pos_position_ids'],
            'raw_prompt': data['pos_raw_prompt'],
            'raw_prompt_ids': data['pos_raw_prompt_ids'],
            'multi_modal_data': data.get('pos_multi_modal_data', {}),  # For vLLM vision inference
            'sample_type': 'pos',  # Mark as pos
            'pair_index': i,  # Track which pair this belongs to
            'data_source': data['data_source'],  # Required by reward manager
            # Ground truth data for pos (needed by reward model)
            'query_base_text': ground_truth.get('query_base_text'),  # Still need query info for pairing
            'query_image_paths': ground_truth.get('query_image_paths'),
            'query_video_paths': ground_truth.get('query_video_paths'),
            'pos_text': ground_truth.get('pos_text'),
            'pos_image_paths': ground_truth.get('pos_image_paths'),
            'pos_video_paths': ground_truth.get('pos_video_paths'),
            # Preserve reward_model structure for reward manager
            'reward_model': data.get('reward_model', {}),
        }
        # Add multimodal fields if present
        if 'pos_pixel_values' in data:
            pos_item['pixel_values'] = data['pos_pixel_values']
        if 'pos_image_grid_thw' in data:
            pos_item['image_grid_thw'] = data['pos_image_grid_thw']

        # Interleave: query, pos, query, pos, ...
        interleaved_data_list.append(query_item)
        interleaved_data_list.append(pos_item)

    # Now batch_size is 2 * original_batch_size
    batch_size = len(interleaved_data_list)
    tensors = defaultdict(list)
    non_tensors = defaultdict(lambda: [None] * batch_size)  # Pre-fill with None

    # Collect all field names first
    all_keys = set()
    for data in interleaved_data_list:
        all_keys.update(data.keys())

    # Initialize non_tensors with None for all fields
    for key in all_keys:
        # Treat multimodal fields as non-tensors
        if key in ['pixel_values', 'image_grid_thw']:
            non_tensors[key] = [None] * batch_size

    # Fill in actual values
    for i, data in enumerate(interleaved_data_list):
        for key, val in data.items():
            # Treat multimodal fields as non-tensors to handle:
            # 1. Variable shapes
            # 2. Missing values (when image/video loading fails)
            if key in ['pixel_values', 'image_grid_thw']:
                non_tensors[key][i] = val
            elif isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = [None] * batch_size
                non_tensors[key][i] = val

    # Handle padding for sequence tensors
    padded_tensors = {}
    for key, val in tensors.items():
        # Pad input_ids (unified for both query and pos)
        if key in ['input_ids']:
            # Pad token sequences to max length in batch
            max_len = max(t.shape[0] for t in val)
            padded = []
            for t in val:
                if t.shape[0] < max_len:
                    # Pad on the left (for prompts)
                    padding = max_len - t.shape[0]
                    padded.append(F.pad(t, (padding, 0), value=0))
                else:
                    padded.append(t)
            padded_tensors[key] = torch.stack(padded, dim=0)

        # Pad attention_mask/position_ids (unified for both query and pos)
        elif key in ['attention_mask', 'position_ids']:
            # Pad attention masks and position_ids
            max_len = max(t.shape[0] for t in val)
            padded = []
            for t in val:
                if t.shape[0] < max_len:
                    # Pad on the left with 0s
                    padding = max_len - t.shape[0]
                    padded.append(F.pad(t, (padding, 0), value=0))
                else:
                    padded.append(t)
            padded_tensors[key] = torch.stack(padded, dim=0)

        else:
            # For other tensors, try to stack directly
            try:
                padded_tensors[key] = torch.stack(val, dim=0)
            except Exception as e:
                # If stacking fails, keep as list in non_tensors
                logger.warning(f"Failed to stack tensor field '{key}': {e}, moving to non_tensors")
                if key not in non_tensors:
                    non_tensors[key] = [None] * batch_size
                for i, v in enumerate(val):
                    non_tensors[key][i] = v

    # Convert non-tensor lists to numpy arrays with dtype=object
    final_non_tensors = {}
    for key, val in non_tensors.items():
        if not isinstance(val, np.ndarray):
            # Ensure it's a list with correct length
            assert len(val) == batch_size, f"Key {key} has length {len(val)} but batch_size is {batch_size}"
            final_non_tensors[key] = np.array(val, dtype=object)
        else:
            final_non_tensors[key] = val

    # Store original batch info for reward calculation
    final_non_tensors['original_batch_size'] = np.array([original_batch_size] * batch_size, dtype=object)

    return {**padded_tensors, **final_non_tensors}


class VLM2VecDataset(Dataset):
    """
    Dataset for VLM2Vec GRPO training.

    Loads multimodal data (text+image or video) and applies appropriate prompts
    for Chain-of-Thought generation that produces better embeddings.

    Args:
        data_files (str or list): Path(s) to JSON file(s).
        tokenizer (PreTrainedTokenizer): Tokenizer for the reasoner model.
        config (DictConfig): Dataset configuration.
        processor (ProcessorMixin): Multimodal processor for images/videos.
        image_root (str): Root directory for image files.
        video_root (str): Root directory for video files.
    """

    # Use custom collate function for this dataset
    collate_fn = staticmethod(collate_vlm2vec_fn)

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: ProcessorMixin,
        max_samples: int = -1,
        image_root: str = "/ytech_m2v_hdd/jianghaonan03/data/VR/data/vlm2vec_train/MMEB-train/images",
        video_root: str = "/ytech_m2v_hdd/jianghaonan03/data/VR/data/vlm2vec_train/MMEB-train/llavahound",
    ):
        # Convert Hydra ListConfig to regular list
        from omegaconf import ListConfig

        if isinstance(data_files, ListConfig):
            data_files = list(data_files)
        elif not isinstance(data_files, (list, tuple)):
            data_files = [data_files]

        self.data_files = data_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples

        self.image_root = Path(image_root)
        self.video_root = Path(video_root)
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 2048)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.max_video_frames = config.get("max_video_frames", 8)
        self.sample_fps = config.get("sample_fps", 2.0)

        # Load data
        self.samples = []
        # NEW: Store dataset indices for each dataset name
        self.dataset_indices = {}  # {dataset_name: [indices]}
        self.dataset_names = []  # List of dataset names in order
        self._load_data()

        # Limit samples if max_samples is specified
        if self.max_samples > 0 and len(self.samples) > self.max_samples:
            logger.info(f"Limiting dataset from {len(self.samples)} to {self.max_samples} samples")
            self.samples = self.samples[:self.max_samples]

    def _load_data(self):
        """Load and parse JSON data files with per-dataset tracking."""
        for data_file in self.data_files:
            logger.info(f"Loading data from {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Determine dataset type from filename
            dataset_name = Path(data_file).stem
            # Only treat as video_retrieval if it's specifically video retrieval task
            # (not caption_retrieval which has video input but text query)
            is_video_retrieval = (
                "video_retrieval" in dataset_name.lower()
                and "caption_retrieval" not in dataset_name.lower()
            )

            # Track starting index for this dataset
            dataset_start_idx = len(self.samples)
            dataset_indices_list = []

            for idx, item in enumerate(data):
                sample = {
                    'data': item,
                    'dataset_name': dataset_name,
                    'is_video_retrieval': is_video_retrieval,
                    'index': len(self.samples),
                }
                self.samples.append(sample)
                dataset_indices_list.append(len(self.samples) - 1)

            # Store dataset indices
            self.dataset_indices[dataset_name] = dataset_indices_list
            self.dataset_names.append(dataset_name)

            logger.info(f"Loaded {len(data)} samples from {dataset_name} (indices: {dataset_start_idx}-{len(self.samples)-1})")

        logger.info(f"Total samples: {len(self.samples)}, datasets: {len(self.dataset_names)}")

    def _clean_question_text(self, text: str) -> str:
        """Clean question text by removing video/image tokens."""
        if not text:
            return ""
        cleaned = text.replace("<video>", "").replace("<image>", "").strip()
        cleaned = cleaned.replace("\n", " ").replace("\r", "")
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _process_video_frames(self, video_frames: list[str], root_dir: Path) -> Optional[list]:
        """Process video frame paths and load images."""
        from verl.utils.dataset.vision_utils import process_image

        if not video_frames or video_frames == ["images/blank.jpg"]:
            return None

        # Clean and resolve paths
        cleaned_frames = []
        for frame_path in video_frames[:self.max_video_frames]:
            # Remove MMEB-train prefix if present
            if frame_path.startswith("MMEB-train/"):
                frame_path = frame_path[len("MMEB-train/"):]

            # Remove duplicate llavahound prefix if present
            # Data paths contain "llavahound/..." but root_dir already ends with "llavahound"
            if frame_path.startswith("llavahound/"):
                frame_path = frame_path[len("llavahound/"):]

            full_path = root_dir / frame_path
            if full_path.exists():
                cleaned_frames.append(str(full_path))
            else:
                logger.warning(f"Video frame not found: {full_path}")

        if not cleaned_frames:
            logger.warning(f"No valid video frames found in {len(video_frames)} frames")
            return None

        # Load images
        images = []
        for frame_path in cleaned_frames:
            try:
                img = process_image({"image": f"file://{frame_path}"}, self.image_patch_size)
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load frame {frame_path}: {e}")
                continue

        return images if images else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            dict with keys:
                - input_ids: Input token IDs for the query
                - attention_mask: Attention mask
                - pixel_values: Pixel values for images (if applicable)
                - image_grid_thw: Image grid info (if applicable)
                - pos_text: Positive text for embedding
                - pos_images: Positive images (if applicable)
                - pos_videos: Positive videos (if applicable)
                - dataset_type: Dataset type identifier
                - raw_data: Original data for debugging
        """
        from verl.utils.dataset.vision_utils import process_image

        sample = self.samples[idx]
        item = sample['data']
        is_video_retrieval = sample['is_video_retrieval']

        # Extract query information - handle both formats
        qry = item.get('qry', {})

        # Determine format: dict (video) vs string (image)
        if isinstance(qry, dict):
            # Video retrieval format: qry is a dict with conversations and video
            conversations = qry.get('conversations', [])

            # Extract question/description from conversations
            question = ""
            for conv in conversations:
                if conv.get('from') == 'gpt':
                    question = self._clean_question_text(conv.get('value', ''))
                    break
                elif conv.get('from') == 'human':
                    question = self._clean_question_text(conv.get('value', ''))

            # For video retrieval, query is text describing the video to find
            if is_video_retrieval:
                prompt_template = PROMPT_VIDEO_RETRIEVAL_WITHOUT_ANSWER
                full_prompt = prompt_template.format(question=question)
                images = None
                videos = None
            else:
                # Video QA: query contains video frames
                video_frames = qry.get('video', [])
                if video_frames and video_frames != ["images/blank.jpg"]:
                    videos = self._process_video_frames(video_frames, self.video_root)
                    images = None
                    video_desc = f"{len(video_frames)} frames sampled at {self.sample_fps} FPS"
                    prompt_template = PROMPT_VIDEO_WITHOUT_ANSWER
                    full_prompt = prompt_template.format(video=video_desc, question=question)
                else:
                    # Fallback to image
                    images = None
                    videos = None
                    prompt_template = PROMPT_IMAGE_WITHOUT_ANSWER
                    full_prompt = prompt_template.format(question=question)

        else:
            # Image format: qry is a string with the question
            question = self._clean_question_text(qry)

            # Load image from qry_image_path
            image_path = item.get('qry_image_path', '')
            if image_path:
                # Remove <|image_1|> token from question
                question = question.replace('<|image_1|>', '').strip()

                # Build full path (relative to image_root/MMEB-train)
                if image_path.startswith('MMEB-train/'):
                    image_path = image_path[len('MMEB-train/'):]

                full_image_path = self.image_root.parent / image_path
                if not full_image_path.exists():
                    # Try without parent
                    full_image_path = self.image_root / image_path

                if full_image_path.exists():
                    images = [process_image({"image": f"file://{full_image_path}"}, self.image_patch_size)]
                else:
                    logger.warning(f"Image not found: {full_image_path}")
                    images = None
            else:
                images = None

            videos = None

            # Choose prompt template based on whether image is available
            if images is not None:
                # Query with image
                prompt_template = PROMPT_IMAGE_WITHOUT_ANSWER
            else:
                # Query with text only (for text-to-image retrieval)
                prompt_template = PROMPT_TEXT_WITHOUT_ANSWER

            full_prompt = prompt_template.format(question=question)

        # Build messages for processor (with image placeholders for tokenization)
        messages_with_images = [
            {
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}]
            }
        ]

        # Build raw_prompt messages for agent loop (MUST include image placeholders for vLLM)
        raw_prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}]
            }
        ]

        # Add images/videos to BOTH messages (for tokenization and vLLM rollout)
        if images:
            for img in images:
                messages_with_images[0]["content"].insert(0, {"type": "image"})
                raw_prompt_messages[0]["content"].insert(0, {"type": "image"})
        elif videos:
            for _ in videos:
                messages_with_images[0]["content"].insert(0, {"type": "image"})  # Videos are treated as images
                raw_prompt_messages[0]["content"].insert(0, {"type": "image"})

        # Tokenize query with smart_resize parameters to control image resolution
        # This significantly reduces token count for high-resolution images/videos
        if images or videos:
            text = self.processor.apply_chat_template(messages_with_images, add_generation_prompt=True, tokenize=False)
            # Add min_pixels/max_pixels for smart resize (from GenVR)
            inputs = self.processor(
                text=[text],
                images=images if images else videos,
                videos=None,  # Qwen3-VL treats video frames as images
                return_tensors="pt",
                padding=False,
                min_pixels=128 * 32 * 32 if not videos else 128 * 32 * 32,  # 131,072
                max_pixels=768 * 32 * 32 if not videos else 300 * 32 * 32,  # Image: 786,432; Video: 307,200
            )
        else:
            text = self.processor.apply_chat_template(messages_with_images, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[text], return_tensors="pt", padding=False)

        # Filter out samples with prompts that are too long (would cause OOM)
        # IMPORTANT: For retrieval tasks, we must keep query-positive pairs intact
        # So we check BOTH query and positive lengths before filtering
        prompt_len = inputs['input_ids'].shape[1]
        if self.filter_overlong_prompts and prompt_len > self.max_prompt_length:
            logger.warning(f"Skipping sample: query prompt length {prompt_len} > {self.max_prompt_length}")
            # Return None to indicate this sample should be skipped
            return None

        # Process positive sample
        pos = item.get('pos', {})

        # Store original query images/videos for reward model
        original_query_images = images
        original_query_videos = videos

        # Also store paths and base text for embedder message construction
        query_image_paths = None
        query_video_paths = None
        query_base_text = ""

        if isinstance(qry, dict):
            # Video format
            conversations = qry.get('conversations', [])

            # For video_retrieval: extract GPT's description text (not human's instruction)
            # For other types: extract the question from human or gpt
            if is_video_retrieval:
                # Video retrieval: query is text describing the video to find
                # Extract GPT's description (the actual video description)
                for conv in conversations:
                    if conv.get('from') == 'gpt':
                        query_base_text = self._clean_question_text(conv.get('value', ''))
                        break
                # Fallback to human's instruction if gpt not found
                if not query_base_text:
                    for conv in conversations:
                        if conv.get('from') == 'human':
                            query_base_text = self._clean_question_text(conv.get('value', ''))
                            break
            else:
                # Video QA or other types: extract question
                for conv in conversations:
                    if conv.get('from') in ['human', 'gpt']:
                        query_base_text = self._clean_question_text(conv.get('value', ''))
                        break

            if is_video_retrieval:
                # Text-only query for video retrieval
                pass
            else:
                # Query has video frames
                video_frames = qry.get('video', [])
                if video_frames and video_frames != ["images/blank.jpg"]:
                    query_video_paths = [str(self.video_root / frame.replace("llavahound/", "")) for frame in video_frames[:self.max_video_frames]]
        else:
            # Image format
            query_base_text = self._clean_question_text(qry)
            image_path = item.get('qry_image_path', '')
            if image_path:
                query_base_text = query_base_text.replace('<|image_1|>', '').strip()
                # Construct full path
                if image_path.startswith('MMEB-train/'):
                    image_path = image_path[len('MMEB-train/'):]
                full_img_path = self.image_root.parent / image_path
                if not full_img_path.exists():
                    full_img_path = self.image_root / image_path
                if full_img_path.exists():
                    query_image_paths = [str(full_img_path)]


        # Also store paths for positive samples
        pos_image_paths = None
        pos_video_paths = None

        # Extract unique identifiers for tracking (video_id, image_id, etc.)
        video_id = None
        image_id = None
        pos_video_id = None
        pos_image_id = None

        # Extract video_id from query video paths if available
        if query_video_paths and len(query_video_paths) > 0:
            # Extract video ID from path like: .../WRgLpPoRQkE/c01_0001.jpeg
            import re
            match = re.search(r'/([^/]+)/[^/]+\.\w+$', query_video_paths[0])
            if match:
                video_id = match.group(1)

        # Extract image_id from query image paths if available
        if query_image_paths and len(query_image_paths) > 0:
            # Use last 2 path segments as identifier
            from pathlib import Path
            path_parts = Path(query_image_paths[0]).parts
            if len(path_parts) >= 2:
                image_id = f"{path_parts[-2]}/{path_parts[-1]}"

        if is_video_retrieval and isinstance(pos, dict):
            # For video retrieval, positive is a video
            pos_conversations = pos.get('conversations', [])
            pos_video_frames = pos.get('video', [])

            if pos_video_frames and pos_video_frames != ["images/blank.jpg"]:
                pos_videos = self._process_video_frames(pos_video_frames, self.video_root)
                pos_images = None
                # Store video paths
                pos_video_paths = [str(self.video_root / frame.replace("llavahound/", "")) for frame in pos_video_frames[:self.max_video_frames]]

                # Extract pos_video_id
                if pos_video_paths and len(pos_video_paths) > 0:
                    import re
                    match = re.search(r'/([^/]+)/[^/]+\.\w+$', pos_video_paths[0])
                    if match:
                        pos_video_id = match.group(1)

                # Get task description from positive conversations
                pos_task = ""
                for conv in pos_conversations:
                    if conv.get('from') == 'human':
                        pos_task = self._clean_question_text(conv.get('value', ''))
                        break

                pos_text = pos_task
            else:
                pos_videos = None
                pos_images = None
                pos_text = ""
        else:
            # For text+image datasets, positive is text
            pos_text = item.get('pos_text', '')
            pos_images_path = item.get('pos_image_path', '')

            # Load positive image if exists
            if pos_images_path:
                if pos_images_path.startswith('MMEB-train/'):
                    pos_images_path = pos_images_path[len('MMEB-train/'):]

                full_pos_image_path = self.image_root.parent / pos_images_path
                if not full_pos_image_path.exists():
                    full_pos_image_path = self.image_root / pos_images_path

                if full_pos_image_path.exists():
                    pos_images = [process_image({"image": f"file://{full_pos_image_path}"}, self.image_patch_size)]
                    pos_image_paths = [str(full_pos_image_path)]  # Store path

                    # Extract pos_image_id
                    from pathlib import Path
                    path_parts = Path(full_pos_image_path).parts
                    if len(path_parts) >= 2:
                        pos_image_id = f"{path_parts[-2]}/{path_parts[-1]}"
                else:
                    pos_images = None
            else:
                pos_images = None

            pos_videos = None

        # ========================================================================
        # Build positive prompt (for reasoner to generate positive CoT)
        # Similar to query, we need to build prompt for positive samples
        # ========================================================================
        if is_video_retrieval and isinstance(pos, dict):
            # For video retrieval: positive has video frames
            if pos_videos:
                # Positive with video - use video retrieval prompt
                pos_full_prompt = PROMPT_POS_VIDEO_RETRIEVAL.format(
                    pos_text=pos_text,
                    pos_video_output=f"{len(pos_video_paths)} video frames" if pos_video_paths else "video"
                )
            else:
                # Fallback to text-only
                pos_full_prompt = PROMPT_POS_TEXT.format(pos_text=pos_text)
        else:
            # For text+image datasets
            if pos_images:
                # Positive with image
                # NOTE: pos_image_description is a placeholder since the actual image is provided as visual input
                pos_full_prompt = PROMPT_POS_IMAGE.format(
                    pos_text=pos_text,
                    pos_image_description="image provided"
                )
            else:
                # Positive text-only
                pos_full_prompt = PROMPT_POS_TEXT.format(pos_text=pos_text)

        # Build positive prompt messages for rollout (MUST include image placeholders for vLLM)
        pos_raw_prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": pos_full_prompt}]
            }
        ]

        # Build positive messages with images for tokenization (training)
        pos_messages_with_images = copy.deepcopy(pos_raw_prompt_messages)

        # Add images/videos to BOTH positive messages (for tokenization and vLLM rollout)
        if pos_images:
            for img in pos_images:
                pos_messages_with_images[0]["content"].insert(0, {"type": "image"})
                pos_raw_prompt_messages[0]["content"].insert(0, {"type": "image"})
        elif pos_videos:
            for _ in pos_videos:
                pos_messages_with_images[0]["content"].insert(0, {"type": "image"})
                pos_raw_prompt_messages[0]["content"].insert(0, {"type": "image"})

        # Tokenize positive prompt with smart_resize to control image resolution
        # This significantly reduces token count for high-resolution images/videos
        if pos_images or pos_videos:
            pos_text_input = self.processor.apply_chat_template(pos_messages_with_images, add_generation_prompt=True, tokenize=False)
            # Add min_pixels/max_pixels for smart resize (same as query)
            pos_inputs = self.processor(
                text=[pos_text_input],
                images=pos_images if pos_images else pos_videos,
                videos=None,  # Qwen3-VL treats video frames as images
                return_tensors="pt",
                padding=False,
                min_pixels=128 * 32 * 32 if not pos_videos else 128 * 32 * 32,  # 131,072
                max_pixels=768 * 32 * 32 if not pos_videos else 300 * 32 * 32,  # Image: 786,432; Video: 307,200
            )
        else:
            pos_text_input = self.processor.apply_chat_template(pos_messages_with_images, add_generation_prompt=True, tokenize=False)
            pos_inputs = self.processor(text=[pos_text_input], return_tensors="pt", padding=False)

        # Filter out samples with positive prompts that are too long
        # This ensures query-positive pair integrity (both must be within limits)
        pos_prompt_len = pos_inputs['input_ids'].shape[1]
        if self.filter_overlong_prompts and pos_prompt_len > self.max_prompt_length:
            logger.warning(f"Skipping sample: positive prompt length {pos_prompt_len} > {self.max_prompt_length}")
            # Return None to skip the entire query-positive pair
            return None

        # Compute positive position_ids
        from verl.utils.model import compute_position_id_with_mask
        pos_position_ids = compute_position_id_with_mask(pos_inputs['attention_mask'])

        # Compute positive raw_prompt_ids for rollout (no truncation)
        pos_raw_prompt_text = self.processor.apply_chat_template(pos_raw_prompt_messages, add_generation_prompt=True, tokenize=False)
        pos_raw_prompt_ids = self.tokenizer.encode(pos_raw_prompt_text, add_special_tokens=False)
        # ========================================================================
        # End positive prompt building
        # ========================================================================

        # Compute position_ids from attention_mask
        from verl.utils.model import compute_position_id_with_mask
        position_ids = compute_position_id_with_mask(inputs['attention_mask'])

        # Compute raw_prompt_ids for rollout (no truncation)
        # raw_prompt_messages now includes image placeholders for vLLM vision inference
        raw_prompt_text = self.processor.apply_chat_template(raw_prompt_messages, add_generation_prompt=True, tokenize=False)
        raw_prompt_ids = self.tokenizer.encode(raw_prompt_text, add_special_tokens=False)

        result = {
            # Query data
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'position_ids': position_ids[0],
            'raw_prompt': raw_prompt_messages,  # Now includes image placeholders for vLLM
            'raw_prompt_ids': raw_prompt_ids,
            # Multimodal data for vLLM agent loop (CRITICAL for vision model inference)
            'multi_modal_data': {
                'image': images if images else (videos if videos else None)
            } if (images or videos) else {},
            # Positive data (for GRPO training)
            'pos_input_ids': pos_inputs['input_ids'][0],
            'pos_attention_mask': pos_inputs['attention_mask'][0],
            'pos_position_ids': pos_position_ids[0],
            'pos_raw_prompt': pos_raw_prompt_messages,  # Now includes image placeholders for vLLM
            'pos_raw_prompt_ids': pos_raw_prompt_ids,
            # Positive multimodal data for vLLM agent loop
            'pos_multi_modal_data': {
                'image': pos_images if pos_images else (pos_videos if pos_videos else None)
            } if (pos_images or pos_videos) else {},
            # Metadata
            'dataset_type': 'video_retrieval' if is_video_retrieval else 'text_image',
            'raw_data': item,
            'dataset_name': sample['dataset_name'],
            'data_source': sample['dataset_name'],  # Add data_source for reward manager
            # Unique identifiers for tracking
            'video_id': video_id,
            'image_id': image_id,
            'pos_video_id': pos_video_id,
            'pos_image_id': pos_image_id,
        }

        # Add multimodal inputs if present (query)
        if 'pixel_values' in inputs:
            result['pixel_values'] = inputs['pixel_values'][0]
        if 'image_grid_thw' in inputs:
            result['image_grid_thw'] = inputs['image_grid_thw'][0]

        # Add multimodal inputs if present (positive)
        if 'pixel_values' in pos_inputs:
            result['pos_pixel_values'] = pos_inputs['pixel_values'][0]
        if 'image_grid_thw' in pos_inputs:
            result['pos_image_grid_thw'] = pos_inputs['image_grid_thw'][0]

        # Store data for reward model in the expected nested structure
        # The reward manager expects: data_item.non_tensor_batch["reward_model"]["ground_truth"]
        result['reward_model'] = {
            'ground_truth': {
                # PIL Images (for backward compatibility, though not used by embedder)
                'original_query_images': original_query_images,
                'original_query_videos': original_query_videos,
                'pos_images': pos_images,
                'pos_videos': pos_videos,
                # Paths and text for embedder message construction (GenVR format)
                'query_image_paths': query_image_paths,
                'query_video_paths': query_video_paths,
                'query_base_text': query_base_text,
                'pos_text': pos_text,
                'pos_image_paths': pos_image_paths,
                'pos_video_paths': pos_video_paths,
                # Metadata
                'dataset_type': 'video_retrieval' if is_video_retrieval else 'text_image',
                'dataset_name': sample['dataset_name'],
            }
        }

        return result


class DatasetGroupedBatchSampler(Sampler):
    """
    Custom sampler that ensures each batch contains samples from the same dataset.

    This sampler groups samples by dataset and yields batches where all samples
    within a batch come from the same dataset. Batches from different datasets
    are interleaved across epochs.

    Args:
        dataset: VLM2VecDataset instance with dataset_indices attribute
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle samples within each dataset
        drop_last: Whether to drop the last incomplete batch for each dataset
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset: VLM2VecDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Get dataset indices from the dataset
        self.dataset_indices = dataset.dataset_indices
        self.dataset_names = dataset.dataset_names

        # Calculate number of batches per dataset
        self.dataset_batches = {}
        self.total_batches = 0
        for dataset_name in self.dataset_names:
            indices = self.dataset_indices[dataset_name]
            num_samples = len(indices)
            if self.drop_last:
                num_batches = num_samples // self.batch_size
            else:
                num_batches = (num_samples + self.batch_size - 1) // self.batch_size
            self.dataset_batches[dataset_name] = num_batches
            self.total_batches += num_batches

        logger.info(f"DatasetGroupedBatchSampler initialized:")
        logger.info(f"  Total batches: {self.total_batches}")
        for dataset_name in self.dataset_names:
            num_samples = len(self.dataset_indices[dataset_name])
            num_batches = self.dataset_batches[dataset_name]
            logger.info(f"  {dataset_name}: {num_samples} samples, {num_batches} batches")

    def __iter__(self):
        # Set random seed for this epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Create batches for each dataset
        all_batches = []

        for dataset_name in self.dataset_names:
            indices = self.dataset_indices[dataset_name].copy()

            # Shuffle indices within dataset if needed
            if self.shuffle:
                # Convert to tensor, shuffle, convert back to list
                indices_tensor = torch.tensor(indices)
                shuffled_indices = indices_tensor[torch.randperm(len(indices_tensor), generator=g)]
                indices = shuffled_indices.tolist()

            # Create batches for this dataset
            dataset_batches = []
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    dataset_batches.append((dataset_name, batch))

            all_batches.extend(dataset_batches)

        # Shuffle batches across datasets (but each batch stays homogeneous)
        if self.shuffle:
            # Shuffle the order of batches
            batch_indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_indices]

        # Yield batches
        for dataset_name, batch in all_batches:
            yield batch

    def __len__(self):
        return self.total_batches

    def set_epoch(self, epoch: int):
        """Set epoch for proper shuffling across epochs."""
        self.epoch = epoch
