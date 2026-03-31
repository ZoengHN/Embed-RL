import os
import re
import asyncio
from typing import Any, Optional, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import ray
import requests
import datetime
import concurrent.futures

try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None

_EMBEDDER_ACTOR = None
_EMBEDDER_ACTOR_NAME = "vlm2vec_embedder_actor_singleton"

@ray.remote
class EmbeddingShareStore:
    def __init__(self):
        self.embeddings = {}
        self.ready_workers = {}

    def store(self, worker_id: int, step: int, embeddings: dict):
        key = f"{step}_{worker_id}"
        self.embeddings[key] = embeddings
        if step not in self.ready_workers:
            self.ready_workers[step] = set()
        self.ready_workers[step].add(worker_id)
        return True

    def get_all(self, step: int, num_workers: int):
        all_embs = []
        for worker_id in range(num_workers):
            key = f"{step}_{worker_id}"
            if key in self.embeddings:
                all_embs.append(self.embeddings[key])
        return all_embs

    def is_ready(self, step: int, num_workers: int):
        if step not in self.ready_workers:
            return False
        return len(self.ready_workers[step]) >= num_workers

    def clear(self, step: int):
        keys_to_delete = [k for k in self.embeddings.keys() if k.startswith(f"{step}_")]
        for k in keys_to_delete:
            del self.embeddings[k]
        if step in self.ready_workers:
            del self.ready_workers[step]
        return True

EMBEDDING_MODEL_PATH = "/models/genvr_embedding_model"
EMB_TOKEN = "<emb>"
API_URL = "http://127.0.0.1:22007/v1/embeddings"
MODEL_NAME = "genvr-qwen3vl-embed"
PRM_API_URL = "http://127.0.0.1:22006/v1/chat/completions"
PRM_MODEL_NAME = "/models/Qwen3-VL-8B-Instruct"
PRM_TOP_K = 4
FORMAT_REWARD_WEIGHT = 0.05
ORM_WEIGHT = 0.8
PRM_WEIGHT = 0.2
ENABLE_FORMAT_REWARD = True
ENABLE_ORM = True
ENABLE_PRM = True
ACCURACY_WEIGHT = 0.5
MARGIN_WEIGHT = 0.5
NEGATIVE_TEMPERATURE = 0.5
MIN_PIXELS = 128 * 32 * 32
MAX_PIXELS = 768 * 32 * 32
VIDEO_MIN_PIXELS = 128 * 32 * 32
VIDEO_MAX_PIXELS = 300 * 32 * 32
VIDEO_TOTAL_PIXELS = 300 * 32 * 32 * 8
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 8
FPS_MAX_FRAMES = 8

async def get_or_create_embedder_actor(
    embedding_model_path: str = EMBEDDING_MODEL_PATH,
    emb_token: str = EMB_TOKEN,
    api_url: str = API_URL,
    model_name: str = MODEL_NAME,
) -> ray.actor.ActorHandle:
    global _EMBEDDER_ACTOR
    if _EMBEDDER_ACTOR is None:
        try:
            _EMBEDDER_ACTOR = ray.get_actor(_EMBEDDER_ACTOR_NAME)
        except ValueError:
            from .embedder_actor_api import EmbedderActorAPI
            _EMBEDDER_ACTOR = ray.remote(num_cpus=1, name=_EMBEDDER_ACTOR_NAME)(EmbedderActorAPI).remote(
                embedding_model_path=embedding_model_path,
                emb_token=emb_token,
                api_url=api_url,
                model_name=model_name,
            )
            await _EMBEDDER_ACTOR.ping.remote()
    return _EMBEDDER_ACTOR

class VLM2VecRewardModel:
    def __init__(
        self,
        embedding_model_path: str = EMBEDDING_MODEL_PATH,
        device: Optional[torch.device] = None,
        format_reward_weight: float = FORMAT_REWARD_WEIGHT,
        orm_weight: float = ORM_WEIGHT,
        prm_weight: float = PRM_WEIGHT,
        emb_token: str = EMB_TOKEN,
        api_url: str = API_URL,
        model_name: str = MODEL_NAME,
        prm_api_url: str = PRM_API_URL,
        prm_model_name: str = PRM_MODEL_NAME,
        prm_top_k: int = PRM_TOP_K,
        negative_temperature: float = NEGATIVE_TEMPERATURE,
        accuracy_weight: float = ACCURACY_WEIGHT,
        margin_weight: float = MARGIN_WEIGHT,
        worker_id: Optional[int] = None,
        num_workers: int = 1,
        embedding_share_store: Optional[Any] = None,
    ):
        self.embedding_model_path = embedding_model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.format_reward_weight = format_reward_weight
        self.orm_weight = orm_weight
        self.prm_weight = prm_weight
        self.emb_token = emb_token
        self.api_url = api_url
        self.model_name = model_name
        self.prm_api_url = prm_api_url
        self.prm_model_name = prm_model_name
        self.prm_top_k = prm_top_k
        self.negative_temperature = negative_temperature
        self.accuracy_weight = accuracy_weight
        self.margin_weight = margin_weight
        self.worker_id = worker_id if worker_id is not None else 0
        self.num_workers = num_workers
        self.embedding_share_store = embedding_share_store
        self.enable_cross_worker_sharing = embedding_share_store is not None and num_workers > 1
        self.embedder_actor = None
        self.prm_log_dir = "./prm_logs"
        self.current_step_prm_logs = []
        self.current_step = None
        os.makedirs(self.prm_log_dir, exist_ok=True)

    def _get_timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    async def _ensure_embedder_actor(self):
        if self.embedder_actor is None:
            self.embedder_actor = await get_or_create_embedder_actor(
                embedding_model_path=self.embedding_model_path,
                emb_token=self.emb_token,
                api_url=self.api_url,
                model_name=self.model_name,
            )

    @staticmethod
    def check_format(completion: str) -> float:
        pattern = r"<thinking>.*?</thinking>.*?<rethink>.*?</rethink>.*?<answer>.*?</answer>"
        match = re.search(pattern, completion, re.DOTALL)
        return 1.0 if match else 0.0

    async def call_prm_api(
        self,
        query_text: str,
        query_cot: str,
        candidate_texts: List[str],
        candidate_cots: List[str],
        query_image_paths: Optional[List[str]] = None,
    ) -> int:
        prompt = f"""Given a query reasoning and multiple candidate reasonings, select the most relevant candidate.

Query Reasoning: {query_cot}

Candidates:
"""
        for i, cand_cot in enumerate(candidate_cots, 1):
            prompt += f"""{i}. Candidate Reasoning: {cand_cot}

"""
        prompt += f"""Output ONLY the number (1-{len(candidate_cots)}) of the most relevant candidate. Do not output anything else."""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = requests.post(
                self.prm_api_url,
                json={
                    "model": self.prm_model_name,
                    "messages": messages,
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                output_text = result["choices"][0]["message"]["content"].strip()
                numbers = re.findall(r'\d+', output_text)
                if numbers:
                    predicted_idx = int(numbers[0]) - 1
                    if 0 <= predicted_idx < len(candidate_texts):
                        return predicted_idx
            return -1
        except Exception:
            return -1

    def generate_query_embedding(
        self,
        completion_text: str,
        query_base_text: str,
        query_image_paths: Optional[List[str]] = None,
        query_video_paths: Optional[List[str]] = None,
    ) -> ray.ObjectRef:
        return self.embedder_actor.generate_embedding.remote(
            base_text=query_base_text,
            cot_text=completion_text,
            image_paths=query_image_paths,
            video_paths=query_video_paths,
        )

    def generate_positive_embedding(
        self,
        pos_text: str,
        pos_image_paths: Optional[List[str]] = None,
        pos_video_paths: Optional[List[str]] = None,
    ) -> ray.ObjectRef:
        return self.embedder_actor.generate_embedding.remote(
            base_text=pos_text,
            cot_text="",
            image_paths=pos_image_paths,
            video_paths=pos_video_paths,
        )

    async def compute_in_batch_orm_reward(
        self,
        completions: List[str],
        batch_data: Dict,
    ) -> List[float]:
        await self._ensure_embedder_actor()
        total_batch_size = len(completions)
        original_B = int(batch_data.get('original_batch_size', [total_batch_size // 2])[0])
        sample_types = batch_data.get('sample_type', [])
        pair_indices = batch_data.get('pair_index', list(range(original_B)) * 2)

        query_indices = []
        pos_indices = []
        query_pair_map = {}
        pos_pair_map = {}

        for i in range(total_batch_size):
            pair_idx = pair_indices[i]
            if sample_types[i] == 'query':
                query_indices.append(i)
                query_pair_map[pair_idx] = i
            elif sample_types[i] == 'pos':
                pos_indices.append(i)
                pos_pair_map[pair_idx] = i

        if len(query_indices) != original_B or len(pos_indices) != original_B:
            return [{
                'orm_reward': 0.5,
                'orm_accuracy': 0.0,
                'similarity_margin': 0.0,
                'pos_similarity': 0.0,
                'neg_mean': 0.0,
            } for _ in range(total_batch_size)]

        all_embedding_refs = []
        for idx in query_indices + pos_indices:
            is_query = (idx in query_indices)
            if is_query:
                base_text = batch_data.get('query_base_text', [None] * total_batch_size)[idx] or ""
                image_paths = batch_data.get('query_image_paths', [None] * total_batch_size)[idx]
                video_paths = batch_data.get('query_video_paths', [None] * total_batch_size)[idx]
            else:
                base_text = batch_data.get('pos_text', [None] * total_batch_size)[idx] or ""
                image_paths = batch_data.get('pos_image_paths', [None] * total_batch_size)[idx]
                video_paths = batch_data.get('pos_video_paths', [None] * total_batch_size)[idx]

            embedding_ref = self.embedder_actor.generate_embedding.remote(
                base_text=base_text,
                cot_text=completions[idx],
                image_paths=image_paths,
                video_paths=video_paths,
            )
            all_embedding_refs.append(embedding_ref)

        all_embeddings = [await ref for ref in all_embedding_refs]

        with torch.no_grad():
            query_embeddings = torch.stack(all_embeddings[:original_B])
            positive_embeddings = torch.stack(all_embeddings[original_B:])
            query_embeddings = F.normalize(query_embeddings.float(), p=2, dim=1)
            positive_embeddings = F.normalize(positive_embeddings.float(), p=2, dim=1)

            if self.enable_cross_worker_sharing:
                current_step = batch_data.get('step', [0])[0] if 'step' in batch_data else 0
                local_data = {
                    'query_embeddings': query_embeddings.cpu(),
                    'positive_embeddings': positive_embeddings.cpu(),
                    'original_B': original_B,
                    'query_indices': query_indices,
                    'pos_indices': pos_indices,
                }
                await self.embedding_share_store.store.remote(
                    worker_id=self.worker_id,
                    step=current_step,
                    embeddings=local_data
                )
                max_wait_time = 30
                wait_interval = 0.1
                total_waited = 0
                while total_waited < max_wait_time:
                    is_ready = await self.embedding_share_store.is_ready.remote(
                        step=current_step,
                        num_workers=self.num_workers
                    )
                    if is_ready:
                        break
                    await asyncio.sleep(wait_interval)
                    total_waited += wait_interval

                all_worker_data = await self.embedding_share_store.get_all.remote(
                    step=current_step,
                    num_workers=self.num_workers
                )

                if len(all_worker_data) > 1:
                    all_query_embs = torch.cat([d['query_embeddings'] for d in all_worker_data], dim=0)
                    all_pos_embs = torch.cat([d['positive_embeddings'] for d in all_worker_data], dim=0)
                    query_embeddings = all_query_embs.to(self.device)
                    positive_embeddings = all_pos_embs.to(self.device)
                    original_B = sum(d['original_B'] for d in all_worker_data)
                    query_indices = []
                    pos_indices = []
                    offset = 0
                    for worker_data in all_worker_data:
                        local_query_indices = worker_data['query_indices']
                        local_pos_indices = worker_data['pos_indices']
                        query_indices.extend([idx + offset for idx in range(len(local_query_indices))])
                        pos_indices.extend([idx + offset for idx in range(len(local_pos_indices))])
                        offset += len(local_query_indices) + len(local_pos_indices)

                if current_step > 0:
                    await self.embedding_share_store.clear.remote(step=current_step - 1)

            similarity_matrix = torch.mm(query_embeddings, positive_embeddings.t()).detach()

        query_rewards = []
        pos_rewards = []
        orm_accs = []
        similarity_margins = []
        prm_rewards = []
        prm_accs_binary = []
        pos_similarities = []

        for i in range(original_B):
            similarities = similarity_matrix[i]
            current_query_idx = query_indices[i]
            current_pair_idx = pair_indices[current_query_idx]

            pos_mask = torch.zeros(original_B, dtype=torch.bool, device=similarities.device)
            neg_mask = torch.zeros(original_B, dtype=torch.bool, device=similarities.device)

            for j in range(original_B):
                pos_idx = pos_indices[j]
                pos_pair_idx = pair_indices[pos_idx]
                if pos_pair_idx == current_pair_idx:
                    pos_mask[j] = True
                else:
                    neg_mask[j] = True

            pos_sim_values = similarities[pos_mask]
            neg_sim_values = similarities[neg_mask]

            n_rollouts = pos_mask.sum().item()
            if n_rollouts > 0:
                topk_values, topk_indices = torch.topk(similarities, n_rollouts)
                num_positives_in_topk = pos_mask[topk_indices].sum().item()
                orm_accuracy = num_positives_in_topk / n_rollouts
            else:
                orm_accuracy = 0.0

            pos_mean = pos_sim_values.mean().item() if len(pos_sim_values) > 0 else 0.0
            if len(neg_sim_values) > 0:
                neg_weights = F.softmax(neg_sim_values / self.negative_temperature, dim=0)
                neg_mean = (neg_sim_values * neg_weights).sum().item()
            else:
                neg_mean = 0.0
            similarity_margin = pos_mean - neg_mean
            pos_similarity = similarities[i].item()

            prm_reward = 0.0
            prm_acc_binary = 0.0
            if self.prm_weight > 0 and original_B > 1:
                same_item_indices = []
                for j in range(original_B):
                    pos_idx = pos_indices[j]
                    pos_pair_idx = pair_indices[pos_idx]
                    if pos_pair_idx == current_pair_idx:
                        same_item_indices.append(j)

                pair_idx_to_best = {}
                for j in range(original_B):
                    pos_idx = pos_indices[j]
                    pos_pair_idx = pair_indices[pos_idx]
                    if pos_pair_idx != current_pair_idx:
                        sim = similarities[j].item()
                        if pos_pair_idx not in pair_idx_to_best or sim > pair_idx_to_best[pos_pair_idx][1]:
                            pair_idx_to_best[pos_pair_idx] = (j, sim)

                different_item_indices = []
                different_item_sims = []
                for pos_idx, sim in pair_idx_to_best.values():
                    different_item_indices.append(pos_idx)
                    different_item_sims.append(sim)

                if len(different_item_indices) > 0:
                    sorted_indices = sorted(range(len(different_item_sims)), key=lambda k: different_item_sims[k], reverse=True)
                    top_k = min(self.prm_top_k, len(sorted_indices))
                    selected_negatives = [different_item_indices[sorted_indices[k]] for k in range(top_k)]
                else:
                    selected_negatives = []

                candidate_pos_indices = same_item_indices + selected_negatives
                num_positives = len(same_item_indices)

                if len(selected_negatives) > 0:
                    candidate_texts = []
                    candidate_cots = []
                    for pos_idx in candidate_pos_indices:
                        actual_pos_idx = pos_indices[pos_idx]
                        pos_text = batch_data.get('pos_text', [None] * total_batch_size)[actual_pos_idx] or ""
                        pos_cot = completions[actual_pos_idx]
                        candidate_texts.append(pos_text)
                        candidate_cots.append(pos_cot)

                    query_idx = query_indices[i]
                    query_text = batch_data.get('query_base_text', [None] * total_batch_size)[query_idx] or ""
                    query_cot = completions[query_idx]
                    query_image_paths = batch_data.get('query_image_paths', [None] * total_batch_size)[query_idx]
                    query_full = f"Instruction: {query_text}\n\nReasoning: {query_cot}"

                    try:
                        selected_idx = await self.call_prm_api(
                            query_text=query_text,
                            query_cot=query_full,
                            candidate_texts=candidate_texts,
                            candidate_cots=candidate_cots,
                            query_image_paths=query_image_paths,
                        )
                        if 0 <= selected_idx < num_positives:
                            prm_reward = similarity_margin
                            prm_acc_binary = 1.0
                        elif selected_idx >= num_positives and selected_idx < len(candidate_pos_indices):
                            prm_reward = 0.0
                            prm_acc_binary = 0.0
                    except Exception:
                        pass

            reward = self.accuracy_weight * orm_accuracy * similarity_margin
            query_rewards.append(reward)
            orm_accs.append(orm_accuracy)
            similarity_margins.append(similarity_margin)
            pos_similarities.append(pos_similarity)
            prm_rewards.append(prm_reward)
            prm_accs_binary.append(prm_acc_binary)

        similarity_matrix_T = similarity_matrix.t()
        pos_prm_rewards = []
        pos_prm_accs_binary = []
        pos_orm_accs = []
        pos_similarity_margins = []
        pos_pos_similarities = []

        for i in range(original_B):
            similarities = similarity_matrix_T[i]
            current_pos_idx = pos_indices[i]
            current_pair_idx = pair_indices[current_pos_idx]

            pos_mask = torch.zeros(original_B, dtype=torch.bool, device=similarities.device)
            neg_mask = torch.zeros(original_B, dtype=torch.bool, device=similarities.device)

            for j in range(original_B):
                query_idx = query_indices[j]
                query_pair_idx = pair_indices[query_idx]
                if query_pair_idx == current_pair_idx:
                    pos_mask[j] = True
                else:
                    neg_mask[j] = True

            pos_sim_values = similarities[pos_mask]
            neg_sim_values = similarities[neg_mask]

            n_rollouts = pos_mask.sum().item()
            if n_rollouts > 0:
                topk_values, topk_indices = torch.topk(similarities, n_rollouts)
                num_positives_in_topk = pos_mask[topk_indices].sum().item()
                orm_accuracy = num_positives_in_topk / n_rollouts
            else:
                orm_accuracy = 0.0

            pos_mean = pos_sim_values.mean().item() if len(pos_sim_values) > 0 else 0.0
            if len(neg_sim_values) > 0:
                neg_weights = F.softmax(neg_sim_values / self.negative_temperature, dim=0)
                neg_mean = (neg_sim_values * neg_weights).sum().item()
            else:
                neg_mean = 0.0
            similarity_margin = pos_mean - neg_mean
            pos_similarity = similarities[i].item()

            prm_reward = 0.0
            prm_acc_binary = 0.0
            if self.prm_weight > 0 and original_B > 1:
                same_item_indices = []
                for j in range(original_B):
                    query_idx = query_indices[j]
                    query_pair_idx = pair_indices[query_idx]
                    if query_pair_idx == current_pair_idx:
                        same_item_indices.append(j)

                pair_idx_to_best = {}
                for j in range(original_B):
                    query_idx = query_indices[j]
                    query_pair_idx = pair_indices[query_idx]
                    if query_pair_idx != current_pair_idx:
                        sim = similarities[j].item()
                        if query_pair_idx not in pair_idx_to_best or sim > pair_idx_to_best[query_pair_idx][1]:
                            pair_idx_to_best[query_pair_idx] = (j, sim)

                different_item_indices = []
                different_item_sims = []
                for query_idx, sim in pair_idx_to_best.values():
                    different_item_indices.append(query_idx)
                    different_item_sims.append(sim)

                if len(different_item_indices) > 0:
                    sorted_indices = sorted(range(len(different_item_sims)), key=lambda k: different_item_sims[k], reverse=True)
                    top_k = min(self.prm_top_k, len(sorted_indices))
                    selected_negatives = [different_item_indices[sorted_indices[k]] for k in range(top_k)]
                else:
                    selected_negatives = []

                candidate_query_indices = same_item_indices + selected_negatives
                num_positives = len(same_item_indices)

                if len(selected_negatives) > 0:
                    candidate_texts = []
                    candidate_cots = []
                    for query_idx_in_batch in candidate_query_indices:
                        actual_query_idx = query_indices[query_idx_in_batch]
                        query_text = batch_data.get('query_base_text', [None] * total_batch_size)[actual_query_idx] or ""
                        query_cot = completions[actual_query_idx]
                        candidate_full = f"Instruction: {query_text}\n\nReasoning: {query_cot}"
                        candidate_texts.append(query_text)
                        candidate_cots.append(candidate_full)

                    pos_idx = pos_indices[i]
                    pos_text = batch_data.get('pos_text', [None] * total_batch_size)[pos_idx] or ""
                    pos_cot = completions[pos_idx]
                    pos_full = f"Text: {pos_text}\n\nReasoning: {pos_cot}"

                    try:
                        selected_idx = await self.call_prm_api(
                            query_text=pos_text,
                            query_cot=pos_full,
                            candidate_texts=candidate_texts,
                            candidate_cots=candidate_cots,
                            query_image_paths=None,
                        )
                        if 0 <= selected_idx < num_positives:
                            prm_reward = similarity_margin
                            prm_acc_binary = 1.0
                        elif selected_idx >= num_positives and selected_idx < len(candidate_query_indices):
                            prm_reward = 0.0
                            prm_acc_binary = 0.0
                    except Exception:
                        pass

            reward = self.accuracy_weight * orm_accuracy * similarity_margin
            pos_rewards.append(reward)
            pos_orm_accs.append(orm_accuracy)
            pos_similarity_margins.append(similarity_margin)
            pos_pos_similarities.append(pos_similarity)
            pos_prm_rewards.append(prm_reward)
            pos_prm_accs_binary.append(prm_acc_binary)

        final_rewards = []
        final_prm_rewards = []
        final_orm_accs = []
        final_similarity_margins = []
        final_pos_similarities = []
        final_prm_accs_binary = []

        for i in range(original_B):
            final_rewards.append(query_rewards[i])
            final_rewards.append(pos_rewards[i])
            final_prm_rewards.append(prm_rewards[i])
            final_prm_rewards.append(pos_prm_rewards[i])
            final_orm_accs.append(orm_accs[i])
            final_orm_accs.append(pos_orm_accs[i])
            final_similarity_margins.append(similarity_margins[i])
            final_similarity_margins.append(pos_similarity_margins[i])
            final_pos_similarities.append(pos_similarities[i])
            final_pos_similarities.append(pos_pos_similarities[i])
            final_prm_accs_binary.append(prm_accs_binary[i])
            final_prm_accs_binary.append(pos_prm_accs_binary[i])

        metrics = {
            "rewards": final_rewards,
            "prm_rewards": final_prm_rewards,
            "orm_accuracies": final_orm_accs,
            "similarity_margins": final_similarity_margins,
            "pos_similarities": final_pos_similarities,
            "prm_accuracies_binary": final_prm_accs_binary,
        }

        del query_embeddings, positive_embeddings, similarity_matrix
        del all_embeddings, all_embedding_refs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    async def compute_rewards(
        self,
        completions: List[List[Dict]],
        batch_data: Dict,
        return_dict: bool = False,
        step: Optional[int] = None,
        **kwargs
    ):
        completion_contents = [completion[0]["content"] for completion in completions]
        format_rewards = [self.check_format(content) for content in completion_contents]
        orm_metrics = await self.compute_in_batch_orm_reward(completion_contents, batch_data)

        orm_rewards = orm_metrics["rewards"]
        prm_rewards = orm_metrics["prm_rewards"]
        orm_accuracies = orm_metrics["orm_accuracies"]
        similarity_margins = orm_metrics["similarity_margins"]
        pos_similarities = orm_metrics["pos_similarities"]
        prm_accuracies_binary = orm_metrics["prm_accuracies_binary"]

        total_rewards = [
            self.format_reward_weight * format_r +
            self.orm_weight * orm_r +
            self.prm_weight * prm_r
            for format_r, orm_r, prm_r in zip(format_rewards, orm_rewards, prm_rewards)
        ]

        if return_dict:
            accuracy_x_margin = [orm_accuracies[i] * similarity_margins[i] for i in range(len(orm_accuracies))]
            prm_acc_x_similarity = prm_rewards
            reward_extra_info = {
                "format_reward": np.array(format_rewards),
                "orm_accuracy": np.array(orm_accuracies),
                "similarity_margin": np.array(similarity_margins),
                "pos_similarity": np.array(pos_similarities),
                "prm_accuracy_binary": np.array(prm_accuracies_binary),
                "accuracy_x_margin": np.array(accuracy_x_margin),
                "prm_acc_x_similarity": np.array(prm_acc_x_similarity),
                "orm_reward_unweighted": np.array(orm_rewards),
                "prm_reward_unweighted": np.array(prm_rewards),
            }
            return {
                "reward_tensor": torch.tensor(total_rewards, dtype=torch.float32).unsqueeze(-1).detach(),
                "reward_extra_info": reward_extra_info
            }
        return total_rewards

    async def __call__(self, completions: List[List[Dict]], **kwargs):
        return await self.compute_rewards(completions, **kwargs)

def create_vlm2vec_reward_model(**kwargs) -> VLM2VecRewardModel:
    return VLM2VecRewardModel(**kwargs)

_REWARD_MODEL_CACHE = None
_BATCH_COORDINATOR = {}
_BATCH_COORDINATOR_LOCK = None

class BatchCoordinator:
    def __init__(self, batch_id: str, batch_size: int):
        self.batch_id = batch_id
        self.batch_size = batch_size
        self.samples = {}
        self.results = {}
        self.ready_event = None
        self.lock = None

    async def wait_and_compute(self, sample_idx: int, sample_data: Dict, reward_model: 'VLM2VecRewardModel') -> Dict[str, float]:
        if self.ready_event is None:
            self.ready_event = asyncio.Event()
        if self.lock is None:
            self.lock = asyncio.Lock()

        async with self.lock:
            self.samples[sample_idx] = sample_data
            if len(self.samples) == self.batch_size:
                await self._compute_batch_rewards(reward_model)
                self.ready_event.set()

        await self.ready_event.wait()
        return self.results[sample_idx]

    async def _compute_batch_rewards(self, reward_model: 'VLM2VecRewardModel'):
        sorted_indices = sorted(self.samples.keys())
        sorted_samples = [self.samples[idx] for idx in sorted_indices]
        completions = [[{"content": sample["completion"]}] for sample in sorted_samples]
        batch_data = {k: [sample[k] for sample in sorted_samples] for k in sorted_samples[0].keys()}
        result = await reward_model.compute_rewards(completions=completions, batch_data=batch_data, return_dict=True)
        reward_tensor = result["reward_tensor"]
        reward_extra_info = result.get("reward_extra_info", {})
        for i, idx in enumerate(sorted_indices):
            reward_score = reward_tensor[i, 0].item()
            extra_info = {k: float(v[i]) for k, v in reward_extra_info.items() if hasattr(v, '__len__')}
            self.results[idx] = {"score": reward_score, **extra_info}

async def compute_rewards(**kwargs):
    global _REWARD_MODEL_CACHE
    if _REWARD_MODEL_CACHE is None:
        _REWARD_MODEL_CACHE = create_vlm2vec_reward_model(**kwargs)
    reward_model = _REWARD_MODEL_CACHE
    await reward_model._ensure_embedder_actor()
    return await reward_model.compute_rewards(**kwargs)

def vlm2vec_reward_function(completions: List[List[Dict]], **kwargs) -> List[float]:
    global _REWARD_MODEL_CACHE
    if _REWARD_MODEL_CACHE is None:
        _REWARD_MODEL_CACHE = create_vlm2vec_reward_model()
    return _REWARD_MODEL_CACHE.compute_rewards(completions, **kwargs)

def compute_score(**kwargs):
    completions = kwargs.get('completions')
    batch_data = kwargs.get('batch_data')
    if completions is not None and batch_data is not None:
        solution_strs_extracted = [comp[0]["content"] for comp in completions]
        ground_truths_built = []
        for i in range(len(completions)):
            gt = {k: batch_data[k][i] for k in ['sample_type', 'pair_index', 'query_base_text', 'pos_text', 'original_batch_size'] if k in batch_data}
            ground_truths_built.append(gt)
        results = compute_rewards_batch(
            data_sources=['unknown'] * len(completions),
            solution_strs=solution_strs_extracted,
            ground_truths=ground_truths_built,
            extra_infos=[{}] * len(completions),** kwargs
        )
        return results if kwargs.get('return_dict') else [results["reward_tensor"][i, 0].item() for i in range(len(completions))]
    elif kwargs.get('data_sources') is not None:
        return compute_rewards_batch(**kwargs)
    else:
        return compute_rewards_batch(
            data_sources=[kwargs.get('data_source')],
            solution_strs=[kwargs.get('solution_str')],
            ground_truths=[kwargs.get('ground_truth')],
            extra_infos=[kwargs.get('extra_info')],
            **kwargs
        )[0]

def compute_rewards_batch(**kwargs):
    global _REWARD_MODEL_CACHE
    cache_key = f"worker_{kwargs.get('worker_id')}" if kwargs.get('worker_id') is not None else "default"
    if _REWARD_MODEL_CACHE is None:
        _REWARD_MODEL_CACHE = {}
    if cache_key not in _REWARD_MODEL_CACHE:
        _REWARD_MODEL_CACHE[cache_key] = create_vlm2vec_reward_model(** kwargs)
    reward_model = _REWARD_MODEL_CACHE[cache_key]

    batch_data = {
        'query_base_text': [gt.get('query_base_text', '') for gt in kwargs['ground_truths']],
        'query_image_paths': [gt.get('query_image_paths') for gt in kwargs['ground_truths']],
        'query_video_paths': [gt.get('query_video_paths') for gt in kwargs['ground_truths']],
        'pos_text': [gt.get('pos_text', '') for gt in kwargs['ground_truths']],
        'pos_image_paths': [gt.get('pos_image_paths') for gt in kwargs['ground_truths']],
        'pos_video_paths': [gt.get('pos_video_paths') for gt in kwargs['ground_truths']],
        'sample_type': [gt.get('sample_type', 'query') for gt in kwargs['ground_truths']],
        'pair_index': [gt.get('pair_index', i // 2) for i, gt in enumerate(kwargs['ground_truths'])],
        'original_batch_size': [len(kwargs['ground_truths']) // 2] * len(kwargs['ground_truths']),
        'step': [kwargs.get('step', 0)] * len(kwargs['ground_truths']),
    }
    completions = [[{"content": s}] for s in kwargs['solution_strs']]

    async def _async_compute():
        return await reward_model.compute_rewards(completions=completions, batch_data=batch_data, return_dict=True,** kwargs)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio:
                nest_asyncio.apply()
            result = loop.run_until_complete(_async_compute())
        else:
            result = loop.run_until_complete(_async_compute())
    except RuntimeError:
        result = asyncio.run(_async_compute())
    return result