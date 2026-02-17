import os
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_embedding_reps(last_hidden_state, input_ids, embedding_token_id):
    if embedding_token_id is None:
        return last_hidden_state[:, -1, :]

    embedding_idx = (input_ids == embedding_token_id)
    batch_size = last_hidden_state.shape[0]
    embedding_pos = []
    for i in range(batch_size):
        positions = torch.where(embedding_idx[i])[0]
        embedding_pos.append(positions[-1] if len(positions) > 0 else input_ids.shape[1] - 1)
    embedding_pos = torch.tensor(embedding_pos, device=last_hidden_state.device)

    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), embedding_pos]
    return reps

def prepare_single_message(text, image_path, max_text_length=512):
    text = text.strip()[:max_text_length] if text else ""

    user_content = []
    if image_path and os.path.exists(image_path):
        user_content.append({
            "type": "image",
            "image": image_path,
            "min_pixels": 128 * 32 * 32,
            "max_pixels": 768 * 32 * 32
        })
    if text:
        user_content.append({"type": "text", "text": text})
    user_content.append({"type": "text", "text": "Summarize the above content in one word:"})

    return [{
        "role": "user",
        "content": user_content,
    }, {
        "role": "assistant",
        "content": [{"type": "text", "text": "<emb>"}],
    }]

def get_embedding(model, processor, emb_token_id, device, text, image_path):
    messages = prepare_single_message(text, image_path)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
    )
    model_inputs = processor(
        text=[processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)],
        images=image_inputs if image_inputs else None,
        videos=video_inputs,
        video_metadata=None,
        **video_kwargs,
        do_resize=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model.model(**model_inputs, return_dict=True)
    embed = get_embedding_reps(outputs.last_hidden_state, model_inputs["input_ids"], emb_token_id)
    return F.normalize(embed, dim=-1).squeeze(0)

def calculate_query_pos_similarity(model_path, sample_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
    
    emb_token = "<emb>"
    if emb_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_tokens([emb_token])
    emb_token_id = processor.tokenizer.convert_tokens_to_ids(emb_token)

    query_embed = get_embedding(
        model, processor, emb_token_id, device,
        text=sample_dict["qry"],
        image_path=sample_dict["qry_image_path"]
    )
    pos_embed = get_embedding(
        model, processor, emb_token_id, device,
        text=sample_dict["pos_text"],
        image_path=sample_dict["pos_image_path"]
    )

    similarity = torch.dot(query_embed, pos_embed).item()
    return similarity

if __name__ == "__main__":
    MODEL_PATH = "./ckpt/Embed-RL-4B"

    test_sample = {
        "qry": "Identify a similar everyday image based on.",
        "qry_image_path": "./eval/image/query.jpg",
        "pos_text": "Represent the given image.",
        "pos_image_path": "./eval/image/target.jpg",
    }
    
    similarity = calculate_query_pos_similarity(MODEL_PATH, test_sample)
    print(f"Query-Pos Similarity: {similarity:.6f}")