from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from src.arguments import ModelArguments, TrainingArguments
from src.model.processor import QWEN3_VL, backbone2model, print_master

import transformers.modeling_utils
if not hasattr(transformers.modeling_utils, "ALL_PARALLEL_STYLES") or transformers.modeling_utils.ALL_PARALLEL_STYLES is None:
    transformers.modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_input(self, input):
        model_inputs = {
            'input_ids': input.get('input_ids'),
            'attention_mask': input.get('attention_mask'),
            'return_dict': True,
        }

        if input.get('pixel_values') is not None:
            model_inputs['pixel_values'] = input['pixel_values']
        if input.get('image_grid_thw') is not None:
            model_inputs['image_grid_thw'] = input['image_grid_thw']
        if input.get('pixel_values_videos') is not None:
            model_inputs['pixel_values_videos'] = input['pixel_values_videos']
        if input.get('video_grid_thw') is not None:
            model_inputs['video_grid_thw'] = input['video_grid_thw']

        if hasattr(self.encoder, 'model'):
            outputs = self.encoder.model(**model_inputs)
        else:
            outputs = self.encoder(**model_inputs)

        last_hidden_state = outputs.last_hidden_state
        emb_token_id = None
        if hasattr(self.encoder.config, 'emb_token_ids') and self.encoder.config.emb_token_ids:
            emb_token_id = self.encoder.config.emb_token_ids[0]

        embed_features = self._get_embedding_reps(
            last_hidden_state,
            model_inputs['input_ids'],
            emb_token_id
        )
        embed_features = F.normalize(embed_features, p=2, dim=-1)
        return embed_features

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                eos_indices = attention_mask.sum(dim=1) - 1
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def _get_embedding_reps(self, last_hidden_state, input_ids, embedding_token_id):
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

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        config._attn_implementation = "flash_attention_2"
        config.vision_config._attn_implementation = "flash_attention_2"
        config.use_cache = False
        config.padding_side = "left"

        base_model = backbone2model[QWEN3_VL].from_pretrained(
            model_args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        if model_args.lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_backbone = QWEN3_VL
        setattr(model_args, 'model_backbone', model_backbone)

        config._attn_implementation = "flash_attention_2"
        config.vision_config._attn_implementation = "flash_attention_2"
        config.use_cache = False
        config.padding_side = "left"

        base_model = backbone2model[QWEN3_VL].from_pretrained(
            model_args.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            config=config
        )

        if model_args.lora:
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        model.model_backbone = model_args.model_backbone
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None
        tgt_reps = self.encode_input(tgt) if tgt else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size

        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))