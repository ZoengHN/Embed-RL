import os
import sys
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler

current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Optional, List, Dict, Union
import yaml

from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoTokenizer,
    AutoProcessor,
    Qwen3VLForConditionalGeneration
)

from dataset.mmeb_unified_dataset import MMEBUnifiedDataset
from dataset.sampler import InterleavedSubBatchSampler
from utils.utils import rank0_print, find_all_linear_names
from torch.utils.data import DataLoader

import deepspeed
from transformers.integrations.deepspeed import HfDeepSpeedConfig

def setup_logging(output_dir, rank=0):
    if rank != 0:
        return
    
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    file_handler = RotatingFileHandler(
        log_file,
        mode='a',
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    class PrintToLog:
        def __init__(self, logger):
            self.logger = logger
            self.stdout = sys.stdout
            
        def write(self, message):
            if message.strip():
                self.logger.info(message.strip())
            self.stdout.write(message)
            
        def flush(self):
            self.stdout.flush()
    
    sys.stdout = PrintToLog(root_logger)
    sys.stderr = PrintToLog(root_logger)
    
    return root_logger

logger = logging.getLogger(__name__)

EMB_TOKEN_ID = None

try:
    import torch.distributed.nn
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def gather_features(
        query_features,
        target_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_query_features = hvd.allgather(query_features)
            all_target_features = hvd.allgather(target_features)
        else:
            with torch.no_grad():
                all_query_features = hvd.allgather(query_features)
                all_target_features = hvd.allgather(target_features)
            if not local_loss:
                gathered_query_features = list(all_query_features.chunk(world_size, dim=0))
                gathered_target_features = list(all_target_features.chunk(world_size, dim=0))
                gathered_query_features[rank] = query_features
                gathered_target_features[rank] = target_features
                all_query_features = torch.cat(gathered_query_features, dim=0)
                all_target_features = torch.cat(gathered_target_features, dim=0)
    else:
        if gather_with_grad:
            all_query_features = torch.cat(torch.distributed.nn.all_gather(query_features), dim=0)
            all_target_features = torch.cat(torch.distributed.nn.all_gather(target_features), dim=0)
        else:
            gathered_query_features = [torch.zeros_like(query_features) for _ in range(world_size)]
            gathered_target_features = [torch.zeros_like(target_features) for _ in range(world_size)]
            dist.all_gather(gathered_query_features, query_features)
            dist.all_gather(gathered_target_features, target_features)
            if not local_loss:
                gathered_query_features[rank] = query_features
                gathered_target_features[rank] = target_features
            all_query_features = torch.cat(gathered_query_features, dim=0)
            all_target_features = torch.cat(gathered_target_features, dim=0)

    return all_query_features, all_target_features

class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, query_features, target_features, logit_scale):
        if self.world_size > 1:
            all_query_features, all_target_features = gather_features(
                query_features, target_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_query = logit_scale * query_features @ all_target_features.T
                logits_per_target = logit_scale * target_features @ all_query_features.T
            else:
                logits_per_query = logit_scale * all_query_features @ all_target_features.T
                logits_per_target = logits_per_query.T
        else:
            logits_per_query = logit_scale * query_features @ target_features.T
            logits_per_target = logit_scale * target_features @ query_features.T

        return logits_per_query, logits_per_target

    def forward(self, query_features, target_features, logit_scale, output_dict=False):
        device = query_features.device
        logits_per_query, logits_per_target = self.get_logits(query_features, target_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_query.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_query, labels) +
            F.cross_entropy(logits_per_target, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

def get_embedding_reps(last_hidden_state, input_ids, embedding_token_id):
    if embedding_token_id is None:
        logger.warning("embedding_token_id is None, returning the representation of the last token")
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

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_local_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path to the model"}
    )
    emb_token: str = field(
        default="<emb>",
        metadata={"help": "Special token for embedding extraction"}
    )

@dataclass
class DataArguments:
    data_config: str = field(
        default="./data_config.yaml",
        metadata={"help": "Path to data configuration file"}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class HFTrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum model sequence length"}
    )
    train_vision_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to train vision encoder"}
    )
    train_vision_projector: bool = field(
        default=False, 
        metadata={"help": "Whether to train vision projector"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA"}
    )
    q_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA"}
    )
    lora_r: int = field(
        default=128,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=256,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints."}
    )
    save_safetensors: bool = field(
        default=False,
        metadata={"help": "Use safetensors saving and loading for state dicts."}
    )
    ddp_find_unused_parameters: bool = field(
        default=True,
        metadata={"help": "Whether to find unused parameters in DDP."}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to deepspeed config file"}
    )
    num_sub_batches_per_batch: int = field(
        default=4,
        metadata={"help": "Number of sub-batches per global batch"}
    )
    max_video_sub_batches_per_batch: int = field(
        default=1,
        metadata={"help": "Maximum number of video sub-batches per global batch"}
    )
    mini_batch_size: int = field(
        default=8,
        metadata={"help": "Mini-batch size for per-step model forward (avoids OOM, adjust by GPU memory)"}
    )
    resume_from_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the checkpoint directory to resume training from (e.g. ./checkpoints/xxx/checkpoint-250)"}
    )

class GenVRTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.clip_loss_fct = ClipLoss(
            local_loss=True,
            gather_with_grad=True,
            rank=rank,
            world_size=world_size
        )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_sampler = InterleavedSubBatchSampler(
            dataset=train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sub_batch_size=None,
            num_sub_batches_per_batch=self.args.num_sub_batches_per_batch,
            max_video_sub_batches_per_batch=self.args.max_video_sub_batches_per_batch,
            probabilities=None,
            seed=self.args.seed if hasattr(self.args, 'seed') else 42,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
        )

        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global EMB_TOKEN_ID

        try:
            batch_size = inputs['input_ids'].shape[0]

            if batch_size % 2 != 0:
                raise ValueError(f"Batch size must be even (got {batch_size}), as it contains query-positive pairs")

            half_batch = batch_size // 2

            if hasattr(model, 'module'):
                unwrapped_model = model.module
            else:
                unwrapped_model = model

            if hasattr(unwrapped_model, 'base_model'):
                unwrapped_model = unwrapped_model.base_model

            if hasattr(unwrapped_model, 'model'):
                qwen3vl_model = unwrapped_model.model
            else:
                qwen3vl_model = unwrapped_model

            image_nums, video_nums = qwen3vl_model._get_image_nums_and_video_nums(
                inputs['input_ids'], inputs_embeds=None
            )

            image_grid_thw = inputs.get('image_grid_thw', None)
            pixel_values = inputs.get('pixel_values', None)

            if image_grid_thw is not None and image_grid_thw.numel() > 0:
                has_image_per_sample = (image_nums > 0).long()

                sample_image_cumsum = torch.cat([
                    torch.zeros(1, dtype=torch.long, device=has_image_per_sample.device),
                    has_image_per_sample.cumsum(0)
                ])

                pixels_per_image_grid = image_grid_thw.prod(-1)
                image_pixel_cumsum = torch.cat([
                    torch.zeros(1, dtype=torch.long, device=pixels_per_image_grid.device),
                    pixels_per_image_grid.cumsum(0)
                ])
            else:
                sample_image_cumsum = None
                image_pixel_cumsum = None

            video_grid_thw = inputs.get('video_grid_thw', None)
            pixel_values_videos = inputs.get('pixel_values_videos', None)

            if video_grid_thw is not None and video_grid_thw.numel() > 0:
                has_video_per_sample = (video_nums > 0).long()

                sample_video_cumsum = torch.cat([
                    torch.zeros(1, dtype=torch.long, device=has_video_per_sample.device),
                    has_video_per_sample.cumsum(0)
                ])

                pixels_per_video_grid = video_grid_thw.prod(-1)
                video_pixel_cumsum = torch.cat([
                    torch.zeros(1, dtype=torch.long, device=pixels_per_video_grid.device),
                    pixels_per_video_grid.cumsum(0)
                ])
            else:
                sample_video_cumsum = None
                video_pixel_cumsum = None

            mini_batch_size = self.args.mini_batch_size

            all_hidden_states = []

            for b_start in range(0, batch_size, mini_batch_size):
                b_end = min(b_start + mini_batch_size, batch_size)

                mb_input_ids = inputs['input_ids'][b_start:b_end]
                mb_attention_mask = inputs['attention_mask'][b_start:b_end] if 'attention_mask' in inputs else None

                mb_pixel_values_videos = None
                mb_video_grid_thw = None

                if pixel_values_videos is not None and video_grid_thw is not None:
                    num_videos_mb = int(has_video_per_sample[b_start:b_end].sum().item())

                    if num_videos_mb > 0:
                        video_start_idx = int(sample_video_cumsum[b_start].item())
                        video_end_idx = int(sample_video_cumsum[b_end].item())
                        mb_video_grid_thw = video_grid_thw[video_start_idx:video_end_idx]

                        video_pixel_start = int(video_pixel_cumsum[video_start_idx].item())
                        video_pixel_end = int(video_pixel_cumsum[video_end_idx].item())
                        mb_pixel_values_videos = pixel_values_videos[video_pixel_start:video_pixel_end]

                mb_pixel_values = None
                mb_image_grid_thw = None

                if pixel_values is not None and image_grid_thw is not None:
                    num_images_mb = int(has_image_per_sample[b_start:b_end].sum().item())

                    if num_images_mb > 0:
                        image_start_idx = int(sample_image_cumsum[b_start].item())
                        image_end_idx = int(sample_image_cumsum[b_end].item())
                        mb_image_grid_thw = image_grid_thw[image_start_idx:image_end_idx]

                        image_pixel_start = int(image_pixel_cumsum[image_start_idx].item())
                        image_pixel_end = int(image_pixel_cumsum[image_end_idx].item())
                        mb_pixel_values = pixel_values[image_pixel_start:image_pixel_end]

                mb_inputs = {
                    'input_ids': mb_input_ids,
                    'return_dict': True,
                }
                if mb_attention_mask is not None:
                    mb_inputs['attention_mask'] = mb_attention_mask
                if mb_pixel_values_videos is not None:
                    mb_inputs['pixel_values_videos'] = mb_pixel_values_videos
                if mb_video_grid_thw is not None:
                    mb_inputs['video_grid_thw'] = mb_video_grid_thw
                if mb_pixel_values is not None:
                    mb_inputs['pixel_values'] = mb_pixel_values
                if mb_image_grid_thw is not None:
                    mb_inputs['image_grid_thw'] = mb_image_grid_thw

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    mb_outputs = unwrapped_model.model.model(**mb_inputs)

                if hasattr(mb_outputs, 'last_hidden_state') and mb_outputs.last_hidden_state is not None:
                    all_hidden_states.append(mb_outputs.last_hidden_state)
                else:
                    raise ValueError(f"Model must return last_hidden_state, got output type: {type(mb_outputs)}")

            all_hidden = torch.cat(all_hidden_states, dim=0)

            qry_hidden = all_hidden[:half_batch]
            pos_hidden = all_hidden[half_batch:]

            qry_input_ids = inputs['input_ids'][:half_batch]
            pos_input_ids = inputs['input_ids'][half_batch:]

            qry_reps = F.normalize(
                get_embedding_reps(qry_hidden, qry_input_ids, EMB_TOKEN_ID),
                p=2, dim=-1
            )
            pos_reps = F.normalize(
                get_embedding_reps(pos_hidden, pos_input_ids, EMB_TOKEN_ID),
                p=2, dim=-1
            )

            contrastive_loss = self.clip_loss_fct(qry_reps, pos_reps, logit_scale=50)

            total_loss = contrastive_loss

            self.log({
                "loss": total_loss.item(),
                "contrastive_loss": contrastive_loss.item(),
            })

            if return_outputs:
                outputs = {
                    "loss": total_loss,
                    "contrastive_loss": contrastive_loss,
                    "logits": None,
                    "hidden_states": (qry_hidden,) if qry_hidden is not None else None
                }
                return total_loss, outputs
            else:
                return total_loss

        except Exception as e:
            logger.error(f"Error in compute_loss: {str(e)}", exc_info=True)
            raise e

def get_model_vocab_size(model):
    if hasattr(model.config, 'vocab_size'):
        return model.config.vocab_size
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'vocab_size'):
        return model.config.text_config.vocab_size
    elif hasattr(model.config, 'llm_config') and hasattr(model.config.llm_config, 'vocab_size'):
        return model.config.llm_config.vocab_size
    else:
        if hasattr(model, 'get_input_embeddings'):
            embedding_layer = model.get_input_embeddings()
            if hasattr(embedding_layer, 'num_embeddings'):
                return embedding_layer.num_embeddings
        if hasattr(model.config, 'to_dict'):
            config_dict = model.config.to_dict()
            if 'vocab_size' in config_dict:
                return config_dict['vocab_size']
            elif 'text_config' in config_dict and 'vocab_size' in config_dict['text_config']:
                return config_dict['text_config']['vocab_size']
        
        rank0_print("Warning: Could not determine model vocab size. Using tokenizer length as fallback.")
        return None

def train():
    try:
        parser = HfArgumentParser((ModelArguments, DataArguments, HFTrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        rank = int(os.environ.get("LOCAL_RANK", 0))
        setup_logging(training_args.output_dir, rank)
        
        if training_args.deepspeed is not None:
            training_args.hf_deepspeed_config = HfDeepSpeedConfig(training_args.deepspeed)
        
        rank0_print(f"Output dir: {training_args.output_dir}")
        
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        args_dir = output_dir / "arguments"
        args_dir.mkdir(parents=True, exist_ok=True)
        
        device_map = None
        if training_args.q_lora:
            device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        
        bnb_config = None
        if training_args.use_lora and training_args.q_lora:
            from transformers import BitsAndBytesConfig
            rank0_print("Using QLoRA quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        rank0_print("Loading model and processor...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
        
        global EMB_TOKEN_ID

        if model_args.emb_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([model_args.emb_token])
            rank0_print(f"Added special token to tokenizer: {model_args.emb_token}")

        if hasattr(processor, 'tokenizer'):
            if model_args.emb_token not in processor.tokenizer.get_vocab():
                processor.tokenizer.add_tokens([model_args.emb_token])
                rank0_print(f"Added special token to processor.tokenizer: {model_args.emb_token}")

        emb_token_id = tokenizer.convert_tokens_to_ids(model_args.emb_token)
        EMB_TOKEN_ID = emb_token_id

        if hasattr(processor, 'tokenizer'):
            proc_emb_token_id = processor.tokenizer.convert_tokens_to_ids(model_args.emb_token)
            rank0_print(f"Tokenizer <emb> ID: {emb_token_id}")
            rank0_print(f"Processor tokenizer <emb> ID: {proc_emb_token_id}")
        
        rank0_print("Loading Qwen3VLForConditionalGeneration (native model)...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if training_args.use_flash_attn else None
        )

        model_vocab_size = get_model_vocab_size(model)

        if model_vocab_size is not None and len(tokenizer) > model_vocab_size:
            rank0_print(f"Resizing model embeddings from {model_vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        else:
            rank0_print(f"Tokenizer length: {len(tokenizer)}, Model vocab size: {model_vocab_size}")
            rank0_print("No need to resize model embeddings.")

        model.config.emb_token_ids = [emb_token_id]
        rank0_print(f"Set model.config.emb_token_ids = [{emb_token_id}]")

        tokenizer.model_max_length = training_args.model_max_length
        
        if training_args.gradient_checkpointing:
            rank0_print("Enabling gradient checkpointing...")
            model.enable_input_require_grads()
        
        if not training_args.train_vision_encoder:
            rank0_print("Freezing vision encoder...")
            if hasattr(model, 'vision_model'):
                for param in model.vision_model.parameters():
                    param.requires_grad = False
        
        if not training_args.train_vision_projector:
            rank0_print("Freezing vision projector...")
            if hasattr(model, 'multi_modal_projector'):
                for param in model.multi_modal_projector.parameters():
                    param.requires_grad = False
        
        if training_args.use_lora:
            rank0_print("Setting up LoRA...")
            
            model_name = model_args.model_name_or_path.lower()
            if "qwen3-vl-2b" in model_name:
                model_family_id = "qwen3-vl-2b"
            elif "qwen3-vl-4b" in model_name:
                model_family_id = "qwen3-vl-4b"
            elif "qwen3-vl-8b" in model_name:
                model_family_id = "qwen3-vl-8b"
            else:
                model_family_id = "qwen3-vl-8b"
            
            vision_encoder_keys = ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"]
            vision_projector_keys = ["visual.merger"]
            llm_keys = ["model"]
            
            named_modules = {n: m for n, m in model.named_modules()}
            lora_modules = []
            full_modules = []

            if training_args.train_vision_encoder:
                rank0_print("Vision encoder will be fully trained...")
                full_modules.extend(vision_encoder_keys)
            
            if training_args.use_lora:
                rank0_print("LoRA for LLM enabled...")
                lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
            else:
                rank0_print("LLM will be fully trained...")
                full_modules.extend(llm_keys)
            
            if training_args.train_vision_projector:
                rank0_print("Vision projector will be fully trained...")
                full_modules.extend(vision_projector_keys)
            
            rank0_print(f"LoRA modules: {lora_modules}")
            rank0_print(f"Full modules: {full_modules}")
            
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=lora_modules,
                modules_to_save=full_modules,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            if training_args.q_lora:
                model = prepare_model_for_kbit_training(
                    model, 
                    use_gradient_checkpointing=training_args.gradient_checkpointing
                )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        with open(data_args.data_config, 'r') as f:
            data_configs = yaml.safe_load(f)

        rank0_print("Loading MMEB datasets...")
        
        train_dataset = MMEBUnifiedDataset(
            data_configs=data_configs['train_datasets'],
            tokenizer=tokenizer,
            image_processor=processor,
            max_length=training_args.model_max_length,
            split="train"
        )
        
        data_collator = GenVRDataCollator(
            tokenizer=tokenizer,
            processor=processor,
            max_length=training_args.model_max_length
        )
        
        if training_args.gradient_checkpointing:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        if training_args.save_strategy is None:
            training_args.save_strategy = "steps"
        
        rank0_print(f"Save strategy: {training_args.save_strategy}")
        rank0_print(f"Save steps: {training_args.save_steps}")
        rank0_print(f"Save total limit: {training_args.save_total_limit}")
        
        trainer = GenVRTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer
        )
        
        rank0_print("Starting training...")
        if training_args.resume_from_checkpoint_path is not None:
            rank0_print(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint_path}")
            trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint_path)
        else:
            trainer.train()
        
        rank0_print("Saving final model...")
        trainer.save_model()
        trainer.save_state()
        
        tokenizer.save_pretrained(training_args.output_dir)
        
        rank0_print(f"Training completed successfully! All logs saved to {training_args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    train()