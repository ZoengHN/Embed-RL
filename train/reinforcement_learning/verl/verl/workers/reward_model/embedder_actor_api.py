import logging
import requests
import torch
from typing import List, Optional, Dict
from transformers import AutoProcessor
from .qwen2_vision_process import process_vision_info

logger = logging.getLogger(__name__)

MIN_PIXELS = 128 * 32 * 32
MAX_PIXELS = 768 * 32 * 32
VIDEO_MIN_PIXELS = 128 * 32 * 32
VIDEO_MAX_PIXELS = 300 * 32 * 32
VIDEO_TOTAL_PIXELS = 300 * 32 * 32 * 8
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 8
FPS_MAX_FRAMES = 8


class EmbedderActorAPI:
    def __init__(
        self,
        embedding_model_path: str,
        emb_token: str = "<emb>",
        api_url: str = "http://localhost:22003/v1/embeddings",
        model_name: str = "embed-rl",
    ):
        self.embedding_model_path = embedding_model_path
        self.emb_token = emb_token
        self.api_url = api_url
        self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(
            embedding_model_path,
            trust_remote_code=True,
        )

        self.emb_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.emb_token)
        if self.emb_token_id == self.processor.tokenizer.unk_token_id:
            logger.warning(f"Embedding token '{self.emb_token}' not found!")

        logger.info(f"EmbedderActorAPI initialized (token: '{self.emb_token}', ID={self.emb_token_id})")

    def ping(self):
        return "pong"

    def _build_messages(
        self,
        base_text: str,
        cot_text: str,
        image_paths: Optional[List[str]] = None,
        video_paths: Optional[List[str]] = None,
    ) -> List[Dict]:
        user_content = []

        if image_paths is not None and len(image_paths) > 0:
            for img_path in image_paths:
                user_content.append({
                    "type": "image",
                    "image": img_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                })

        if video_paths is not None and len(video_paths) > 0:
            video_content = {
                "video": video_paths,
                "total_pixels": VIDEO_TOTAL_PIXELS,
                "min_pixels": VIDEO_MIN_PIXELS,
                "max_pixels": VIDEO_MAX_PIXELS,
                "max_frames": FPS_MAX_FRAMES,
                "sample_fps": FPS,
            }
            user_content.append(video_content)

        if base_text:
            user_content.append({"type": "text", "text": base_text})

        if cot_text:
            user_content.append({"type": "text", "text": cot_text})

        user_content.append({"type": "text", "text": "Summarize the above content in one word:"})

        messages = [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.emb_token}],
            }
        ]

        return messages

    def generate_embedding(
        self,
        base_text: str,
        cot_text: str = "",
        image_paths: Optional[List[str]] = None,
        video_paths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        try:
            messages = self._build_messages(base_text, cot_text, image_paths, video_paths)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_name,
                "input": text,
                "step_tag_id": self.emb_token_id
            }

            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                embedding_list = result["data"][0]["embedding"]
                embedding = torch.tensor(embedding_list, dtype=torch.float32)
                return embedding
            else:
                logger.error(f"API request failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return torch.zeros(2048, dtype=torch.float32)

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            logger.error(f"Base text: {base_text[:200] if base_text else 'None'}...")
            return torch.zeros(2048, dtype=torch.float32)