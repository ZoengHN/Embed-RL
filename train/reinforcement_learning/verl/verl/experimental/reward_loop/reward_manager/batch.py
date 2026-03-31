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

import inspect
import logging
import ray
import os

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

logger = logging.getLogger(__name__)


# Global embedding share store (singleton)
_EMBEDDING_SHARE_STORE = None
_WORKER_ID_ALLOCATOR = None


@ray.remote
class WorkerIDAllocator:
    """Ray actor to allocate unique worker IDs."""
    def __init__(self):
        self.next_id = 0
        self.worker_ids = {}

    def get_worker_id(self, worker_key):
        """Get or allocate a worker ID for a given worker key."""
        if worker_key not in self.worker_ids:
            self.worker_ids[worker_key] = self.next_id
            self.next_id += 1
        return self.worker_ids[worker_key]


def get_or_create_worker_id_allocator():
    """Get or create the global worker ID allocator."""
    global _WORKER_ID_ALLOCATOR
    if _WORKER_ID_ALLOCATOR is None:
        try:
            _WORKER_ID_ALLOCATOR = WorkerIDAllocator.remote()
            logger.info("[BatchRewardManager] Created WorkerIDAllocator")
        except Exception as e:
            logger.error(f"[BatchRewardManager] Failed to create WorkerIDAllocator: {e}")
            _WORKER_ID_ALLOCATOR = None
    return _WORKER_ID_ALLOCATOR


def get_or_create_embedding_share_store():
    """Get or create the global embedding share store."""
    global _EMBEDDING_SHARE_STORE
    if _EMBEDDING_SHARE_STORE is None:
        try:
            from verl.workers.reward_model.vlm2vec_reward import EmbeddingShareStore
            _EMBEDDING_SHARE_STORE = EmbeddingShareStore.remote()
            logger.info("[BatchRewardManager] Created global EmbeddingShareStore")
        except Exception as e:
            logger.error(f"[BatchRewardManager] Failed to create EmbeddingShareStore: {e}")
            _EMBEDDING_SHARE_STORE = None
    return _EMBEDDING_SHARE_STORE


@register("batch")
class BatchRewardManager(RewardManagerBase):
    """
    Batch Reward Manager for in-batch negative sampling.

    Unlike NaiveRewardManager which processes samples one by one,
    this manager processes the entire batch together to enable
    in-batch negative sampling for contrastive learning.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # NEW: Cross-worker embedding sharing
        # Get unique worker ID from allocator
        try:
            worker_id_allocator = get_or_create_worker_id_allocator()
            if worker_id_allocator is not None:
                # Use Ray worker ID as key
                worker_key = str(ray.get_runtime_context().get_worker_id())
                self.worker_id = ray.get(worker_id_allocator.get_worker_id.remote(worker_key))
                logger.info(f"[BatchRewardManager] Allocated worker_id={self.worker_id} for worker_key={worker_key}")
            else:
                # Fallback: use process ID
                self.worker_id = os.getpid() % 100
                logger.warning(f"[BatchRewardManager] Using fallback worker_id={self.worker_id} (PID % 100)")
        except Exception as e:
            # Fallback: use process ID
            self.worker_id = os.getpid() % 100
            logger.warning(f"[BatchRewardManager] Failed to allocate worker_id, using PID: {self.worker_id}, error: {e}")

        # Get num_workers from config (check both possible locations)
        if hasattr(config, 'reward_model') and hasattr(config.reward_model, 'num_workers'):
            self.num_workers = config.reward_model.num_workers
        elif hasattr(config, 'num_workers'):
            self.num_workers = config.num_workers
        else:
            self.num_workers = config.get('num_workers', 1) if hasattr(config, 'get') else 1

        logger.info(f"[BatchRewardManager] Initializing worker {self.worker_id} with num_workers={self.num_workers}")

        # DISABLED: Cross-worker embedding sharing (not needed when only increasing micro_batch_size)
        # Create or get embedding share store if multiple workers
        # if self.num_workers > 1:
        #     self.embedding_share_store = get_or_create_embedding_share_store()
        #     if self.embedding_share_store is not None:
        #         logger.info(
        #             f"[BatchRewardManager] Worker {self.worker_id}/{self.num_workers} "
        #             f"initialized with cross-worker embedding sharing"
        #         )
        #     else:
        #         logger.warning(f"[BatchRewardManager] Failed to create EmbeddingShareStore, sharing disabled")
        # else:
        #     self.embedding_share_store = None
        #     logger.info(f"[BatchRewardManager] Single worker mode, embedding sharing disabled")

        # Always disable embedding sharing
        self.embedding_share_store = None
        logger.info(f"[BatchRewardManager] Embedding sharing disabled (using single-worker in-batch negatives only)")

    async def run_single(self, data: DataProto) -> dict:
        """
        Process a single data item (for compatibility).

        Note: This falls back to simple similarity computation without in-batch negatives.
        For true in-batch negative sampling, use run_batch instead.
        """
        assert len(data) == 1, "run_single only supports single data item"

        # Fall back to simple processing for single sample
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # Check if compute_score is async and can handle single-sample mode
        # Try to call it with the expected signature for single samples
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        # Check function signature to determine how to call it
        import inspect
        sig = inspect.signature(self.compute_score)
        params = list(sig.parameters.keys())

        # If compute_score expects completions/batch_data (batch mode function),
        # construct the appropriate arguments
        if 'completions' in params and 'batch_data' in params:
            # This is a batch-mode function - construct completions and batch_data
            completions = [[{"content": response_str}]]

            # Build batch_data from non_tensor_batch
            batch_data = {
                "sample_type": [data_item.non_tensor_batch.get("sample_type", "query")],
                "pair_index": [data_item.non_tensor_batch.get("pair_index", 0)],
                "query_base_text": [data_item.non_tensor_batch.get("query_base_text", "")],
                "query_image_paths": [data_item.non_tensor_batch.get("query_image_paths", None)],
                "query_video_paths": [data_item.non_tensor_batch.get("query_video_paths", None)],
                "pos_text": [data_item.non_tensor_batch.get("pos_text", "")],
                "pos_image_paths": [data_item.non_tensor_batch.get("pos_image_paths", None)],
                "pos_video_paths": [data_item.non_tensor_batch.get("pos_video_paths", None)],
                "original_batch_size": [data_item.non_tensor_batch.get("original_batch_size", 1)],
            }

            if self.is_async_reward_score:
                result = await self.compute_score(
                    completions=completions,
                    batch_data=batch_data,
                    return_dict=True,
                    **extra_reward_kwargs,
                )
            else:
                result = await self.loop.run_in_executor(
                    None,
                    lambda: self.compute_score(
                        completions=completions,
                        batch_data=batch_data,
                        return_dict=True,
                        **extra_reward_kwargs,
                    ),
                )

            # Parse batch-mode result
            if isinstance(result, dict) and "reward_tensor" in result:
                score = result["reward_tensor"][0, 0].item()
                reward_extra_info = {}
                for key, values in result.get("reward_extra_info", {}).items():
                    if hasattr(values, '__len__') and len(values) > 0:
                        reward_extra_info[key] = float(values[0]) if hasattr(values[0], '__float__') else values[0]
                return {"reward_score": score, "reward_extra_info": reward_extra_info}
        elif 'completions' not in params and 'batch_data' not in params:
            # This is a single-sample function - use the old approach
            # Import the single-sample version
            from verl.workers.reward_model.vlm2vec_reward import compute_score as single_compute_score

            if inspect.iscoroutinefunction(single_compute_score):
                result = await single_compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                )
            else:
                result = await self.loop.run_in_executor(
                    None,
                    lambda: single_compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        **extra_reward_kwargs,
                    ),
                )
        else:
            # Standard single-sample function
            if self.is_async_reward_score:
                result = await self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                )
            else:
                result = await self.loop.run_in_executor(
                    None,
                    lambda: self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        **extra_reward_kwargs,
                    ),
                )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    async def run_batch(self, data: DataProto) -> list[dict]:
        """
        Process a batch of data items together for in-batch negative sampling.

        This is the key difference from NaiveRewardManager: instead of processing
        samples one by one, we process them as a batch to enable contrastive learning.
        """
        batch_size = len(data)

        # DEBUG: Log batch size to understand what's happening
        logger.warning(f"[BatchRewardManager.run_batch] Received batch_size={batch_size}, worker_id={self.worker_id}")

        # Decode all responses
        response_strs = []
        for i in range(batch_size):
            data_item = data[i]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = await self.loop.run_in_executor(
                None, lambda ids=valid_response_ids: self.tokenizer.decode(ids, skip_special_tokens=True)
            )
            response_strs.append(response_str)

        # Prepare batch data for reward computation
        batch_data = {}

        # Collect all non-tensor fields from the batch
        sample_types = []
        pair_indices = []
        query_base_texts = []
        query_image_paths = []
        query_video_paths = []
        pos_texts = []
        pos_image_paths = []
        pos_video_paths = []
        original_batch_sizes = []

        for i in range(batch_size):
            data_item = data[i]
            non_tensor = data_item.non_tensor_batch

            # Extract fields for in-batch negative sampling
            sample_types.append(non_tensor.get("sample_type", "query"))
            pair_indices.append(non_tensor.get("pair_index", i // 2))
            query_base_texts.append(non_tensor.get("query_base_text", ""))
            query_image_paths.append(non_tensor.get("query_image_paths", None))
            query_video_paths.append(non_tensor.get("query_video_paths", None))
            pos_texts.append(non_tensor.get("pos_text", ""))
            pos_image_paths.append(non_tensor.get("pos_image_paths", None))
            pos_video_paths.append(non_tensor.get("pos_video_paths", None))
            original_batch_sizes.append(non_tensor.get("original_batch_size", batch_size // 2))

        # Build batch_data dict
        batch_data["sample_type"] = sample_types
        batch_data["pair_index"] = pair_indices
        batch_data["query_base_text"] = query_base_texts
        batch_data["query_image_paths"] = query_image_paths
        batch_data["query_video_paths"] = query_video_paths
        batch_data["pos_text"] = pos_texts
        batch_data["pos_image_paths"] = pos_image_paths
        batch_data["pos_video_paths"] = pos_video_paths
        batch_data["original_batch_size"] = original_batch_sizes

        # Format completions for reward model
        completions = [[{"content": response_str}] for response_str in response_strs]

        # Call batch reward function
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        # Extract global_steps from meta_info if available (for logging)
        step = None
        if hasattr(data, 'meta_info') and data.meta_info is not None:
            step = data.meta_info.get('global_steps', None)
        if step is not None:
            extra_reward_kwargs['step'] = step

        # NEW: Add cross-worker embedding sharing parameters
        if self.embedding_share_store is not None:
            extra_reward_kwargs['worker_id'] = self.worker_id
            extra_reward_kwargs['num_workers'] = self.num_workers
            extra_reward_kwargs['embedding_share_store'] = self.embedding_share_store
            logger.debug(
                f"[BatchRewardManager] Worker {self.worker_id} processing batch at step {step} "
                f"with cross-worker sharing enabled"
            )

        if self.is_async_reward_score:
            # Assume compute_score can handle batch mode if it receives completions and batch_data
            result = await self.compute_score(
                completions=completions,
                batch_data=batch_data,
                return_dict=True,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    completions=completions,
                    batch_data=batch_data,
                    return_dict=True,
                    **extra_reward_kwargs,
                ),
            )

        # Parse results
        if isinstance(result, dict) and "reward_tensor" in result:
            # Result is in batch format with reward_tensor and reward_extra_info
            reward_tensor = result["reward_tensor"]  # Shape: (batch_size, 1)
            reward_extra_info = result.get("reward_extra_info", {})

            # Convert to list of dicts
            outputs = []
            for i in range(batch_size):
                reward_score = reward_tensor[i, 0].item()
                extra_info = {}

                # Extract extra info for this sample
                for key, values in reward_extra_info.items():
                    if hasattr(values, '__len__') and len(values) > i:
                        extra_info[key] = float(values[i]) if hasattr(values[i], '__float__') else values[i]

                outputs.append({
                    "reward_score": reward_score,
                    "reward_extra_info": extra_info
                })

            return outputs
        else:
            # Fallback: assume result is a list of scores or dicts
            logger.warning("Batch reward function did not return expected format. Falling back.")
            outputs = []
            for i in range(batch_size):
                if isinstance(result, list):
                    item = result[i]
                else:
                    item = result

                if isinstance(item, dict):
                    outputs.append({
                        "reward_score": item.get("score", 0.0),
                        "reward_extra_info": {k: v for k, v in item.items() if k != "score"}
                    })
                else:
                    outputs.append({
                        "reward_score": float(item),
                        "reward_extra_info": {"acc": float(item)}
                    })

            return outputs
