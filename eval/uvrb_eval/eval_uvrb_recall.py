import os
import json
import argparse
from typing import Dict, List
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from transformers import AutoProcessor, AutoTokenizer
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info


def get_embedding_reps(last_hidden_state, input_ids, embedding_token_id):
    if embedding_token_id is None:
        return last_hidden_state[:, -1, :]

    embedding_idx = (input_ids == embedding_token_id)
    batch_size = last_hidden_state.shape[0]

    embedding_pos = []
    for i in range(batch_size):
        positions = torch.where(embedding_idx[i])[0]
        if len(positions) > 0:
            embedding_pos.append(positions[-1])
        else:
            embedding_pos.append(input_ids.shape[1] - 1)

    embedding_pos = torch.tensor(embedding_pos, device=last_hidden_state.device)

    reps = last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        embedding_pos
    ]

    return reps


def load_jsonl(path: str) -> List[Dict]:
    assert os.path.exists(path), f"{path} not found"
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_qrel_from_instances(instances_path: str) -> Dict[str, List[str]]:
    instances = load_jsonl(instances_path)
    qrel: Dict[str, List[str]] = {}
    for inst in instances:
        qid = inst["qid"]
        pos_list = inst.get("pos", [])
        if not pos_list:
            continue
        if qid not in qrel:
            qrel[qid] = []
        qrel[qid].extend(pos_list)
    print(f"Loaded {len(qrel)} queries from {instances_path}")
    avg_pos = sum(len(v) for v in qrel.values()) / max(1, len(qrel))
    print(f"Avg #pos per query: {avg_pos:.2f}")
    return qrel


def compute_recall_at_k(relevant_docs: List[str],
                        retrieved_docs: List[str],
                        k: int) -> float:
    if not relevant_docs:
        return 0.0
    top_k = set(retrieved_docs[:k])
    rel_set = set(relevant_docs)
    return 1.0 if rel_set.intersection(top_k) else 0.0


class UVRBQueryCollator:
    def __init__(self, processor: AutoProcessor, max_length: int = 8192):
        self.processor = processor
        self.max_length = max_length

        if hasattr(self.processor, 'tokenizer'):
            emb_token = "<emb>"
            if emb_token in self.processor.tokenizer.get_vocab():
                emb_token_id = self.processor.tokenizer.convert_tokens_to_ids(emb_token)
                print(f"✓ UVRBQueryCollator: processor.tokenizer has {emb_token} (ID: {emb_token_id})")
            else:
                print(f"⚠️  WARNING: UVRBQueryCollator: processor.tokenizer missing {emb_token}")

    def __call__(self, batch) -> Dict:
        messages_list, qid_list = zip(*batch)

        texts = []
        for messages in messages_list:
            if hasattr(self.processor, 'tokenizer'):
                text = self.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            texts.append(text)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages_list,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        if video_inputs is not None or image_inputs is not None:
            model_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **video_kwargs,
                do_resize=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            model_inputs = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        model_inputs["qids"] = list(qid_list)
        return model_inputs


class UVRBCandidateCollator:
    def __init__(self, processor: AutoProcessor, max_length: int = 8192):
        self.processor = processor
        self.max_length = max_length

        if hasattr(self.processor, 'tokenizer'):
            emb_token = "<emb>"
            if emb_token in self.processor.tokenizer.get_vocab():
                emb_token_id = self.processor.tokenizer.convert_tokens_to_ids(emb_token)
                print(f"✓ UVRBCandidateCollator: processor.tokenizer has {emb_token} (ID: {emb_token_id})")
            else:
                print(f"⚠️  WARNING: UVRBCandidateCollator: processor.tokenizer missing {emb_token}")

    def __call__(self, batch) -> Dict:
        messages_list, did_list = zip(*batch)

        texts = []
        for messages in messages_list:
            if hasattr(self.processor, 'tokenizer'):
                text = self.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            texts.append(text)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages_list,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True
        )

        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        if video_inputs is not None or image_inputs is not None:
            model_inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **video_kwargs,
                do_resize=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            model_inputs = self.processor(
                text=texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        model_inputs["dids"] = list(did_list)
        return model_inputs


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg["dataset_name"]
    dataset_root = cfg["dataset_root"]

    queries_path = os.path.join(dataset_root, cfg["queries_file"])
    instances_path = os.path.join(dataset_root, cfg["instances_file"])
    corpus_path = os.path.join(dataset_root, cfg["corpus_file"])
    frames_root = os.path.join(dataset_root, cfg["frames_dir"])
    
    images_root = None
    if "images_dir" in cfg:
        images_root = os.path.join(dataset_root, cfg["images_dir"])

    num_video_frames = cfg.get("num_video_frames", 8)
    max_text_length = cfg.get("max_text_length", 512)
    batch_size = cfg.get("batch_size", 16)
    num_workers = cfg.get("num_workers", 8)
    max_length = cfg.get("max_length", 8192)
    use_query_prompt = cfg.get("use_query_prompt", True)

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using max_length: {max_length}")
    print(f"Using query prompt: {use_query_prompt}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uvrb_eval import UVRBQueryDataset, UVRBCandidateDataset

    query_dataset = UVRBQueryDataset(
        queries_path=queries_path,
        max_text_length=max_text_length,
        dataset_name=dataset_name,
        frames_root=frames_root,
        num_video_frames=num_video_frames,
        images_root=images_root,
        use_query_prompt=use_query_prompt,
    )

    cand_dataset = UVRBCandidateDataset(
        corpus_path=corpus_path,
        frames_root=frames_root,
        num_video_frames=num_video_frames,
        max_text_length=max_text_length,
        dataset_name=dataset_name,
    )

    from transformers import Qwen3VLForConditionalGeneration

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    is_main = accelerator.is_main_process

    print(f"Loading model from: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    if hasattr(model.config, 'emb_token_ids') and model.config.emb_token_ids:
        emb_token_id = model.config.emb_token_ids[0]
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
        emb_token = tokenizer.convert_ids_to_tokens(emb_token_id)

        print(f"Found <emb> token in model config: {emb_token} (ID: {emb_token_id})")

        if hasattr(processor, 'tokenizer'):
            if emb_token not in processor.tokenizer.get_vocab():
                processor.tokenizer.add_tokens([emb_token])
                print(f"Added {emb_token} to processor.tokenizer for evaluation")
            else:
                print(f"processor.tokenizer already has {emb_token}")
    else:
        print("⚠️  Warning: emb_token_ids not found in model config, using default <emb>")
        emb_token = "<emb>"
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
        if emb_token in tokenizer.get_vocab():
            if hasattr(processor, 'tokenizer'):
                if emb_token not in processor.tokenizer.get_vocab():
                    processor.tokenizer.add_tokens([emb_token])
                    print(f"Added {emb_token} to processor.tokenizer")

    query_collator = UVRBQueryCollator(processor, max_length=max_length)
    cand_collator = UVRBCandidateCollator(processor, max_length=max_length)

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=query_collator,
    )
    cand_loader = DataLoader(
        cand_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cand_collator,
    )

    query_loader, cand_loader, model = accelerator.prepare(
        query_loader, cand_loader, model
    )
    model.eval()

    all_cand_embeds: List[torch.Tensor] = []
    all_doc_ids: List[str] = []

    emb_token_id = None
    if hasattr(model.config, 'emb_token_ids') and model.config.emb_token_ids:
        emb_token_id = model.config.emb_token_ids[0]
        print(f"Using <emb> token ID: {emb_token_id}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=True)
        emb_token = "<emb>"
        if emb_token in tokenizer.get_vocab():
            emb_token_id = tokenizer.convert_tokens_to_ids(emb_token)
            print(f"Using <emb> token ID: {emb_token_id}")
        else:
            print("⚠️  Warning: <emb> token not found, using last token position")

    with torch.no_grad():
        for batch in tqdm(cand_loader, disable=not is_main, desc="Encode candidates"):
            batch_dids = batch.pop("dids")

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model.model(**batch, return_dict=True)
            last_hidden_state = outputs.last_hidden_state

            cand_embed = get_embedding_reps(
                last_hidden_state,
                batch['input_ids'],
                emb_token_id
            )

            cand_embed = F.normalize(cand_embed, dim=-1)
            cand_embed = accelerator.gather_for_metrics(cand_embed)
            all_cand_embeds.append(cand_embed.cpu())

            from accelerate import utils as accelerate_utils
            batch_dids = accelerate_utils.gather_object(batch_dids)[:len(cand_embed)]
            all_doc_ids.extend(batch_dids)

    all_cand_embeds = torch.cat(all_cand_embeds, dim=0)

    all_query_embeds: List[torch.Tensor] = []
    all_query_ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(query_loader, disable=not is_main, desc="Encode queries"):
            batch_qids = batch.pop("qids")

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model.model(**batch, return_dict=True)
            last_hidden_state = outputs.last_hidden_state

            query_embed = get_embedding_reps(
                last_hidden_state,
                batch['input_ids'],
                emb_token_id
            )

            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            all_query_embeds.append(query_embed.cpu())

            from accelerate import utils as accelerate_utils
            batch_qids = accelerate_utils.gather_object(batch_qids)[:len(query_embed)]
            all_query_ids.extend(batch_qids)

    all_query_embeds = torch.cat(all_query_embeds, dim=0)

    if not is_main:
        return

    scores = all_query_embeds @ all_cand_embeds.T
    topk = 50
    topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1)

    doc_ids_list = list(all_doc_ids)
    query_ids_list = list(all_query_ids)

    retrieved_doc_ids_per_query: List[List[str]] = []
    for row in topk_indices.tolist():
        retrieved_doc_ids_per_query.append([doc_ids_list[i] for i in row])

    qrel = load_qrel_from_instances(instances_path)

    k_list = [1, 5, 10, 50]
    res = {f"recall_{k}": [] for k in k_list}

    for qid, retrieved_docs in zip(query_ids_list, retrieved_doc_ids_per_query):
        rel_docs = qrel.get(qid, [])
        for k in k_list:
            r = compute_recall_at_k(rel_docs, retrieved_docs, k)
            res[f"recall_{k}"].append(r)

    print(f"===== Dataset: {dataset_name}  Model: {args.model_id} =====")
    print(f"Max length used: {max_length}")
    for k in k_list:
        mean_recall = sum(res[f"recall_{k}"]) / max(1, len(res[f"recall_{k}"]))
        print(f"Recall@{k}: {mean_recall:.4f}")

    model_name = os.path.basename(args.model_id.rstrip("/"))
    save_prefix = f"{dataset_name}_{model_name}"

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, f"{save_prefix}_query_ids.json"), "w") as f:
        json.dump(query_ids_list, f, indent=2)
    with open(os.path.join(save_dir, f"{save_prefix}_doc_ids.json"), "w") as f:
        json.dump(doc_ids_list, f, indent=2)
    with open(os.path.join(save_dir, f"{save_prefix}_retrieved_doc_ids.json"), "w") as f:
        json.dump(retrieved_doc_ids_per_query, f, indent=2)

    torch.save(
        all_query_embeds,
        os.path.join(save_dir, f"{save_prefix}_query_embeds.pt"),
    )
    torch.save(
        all_cand_embeds,
        os.path.join(save_dir, f"{save_prefix}_cand_embeds.pt"),
    )

    with open(os.path.join(save_dir, f"{save_prefix}_results.txt"), "w") as f:
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"model: {args.model_id}\n")
        f.write(f"max_length: {max_length}\n")
        for k in k_list:
            mean_recall = sum(res[f"recall_{k}"]) / max(1, len(res[f"recall_{k}"]))
            f.write(f"Recall@{k}: {mean_recall:.6f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="yaml config for one dataset"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="local Qwen3-VL model path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results_uvrb",
    )

    args = parser.parse_args()
    main(args)