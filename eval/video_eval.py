import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

VIDEO_MIN_PIXELS = 128 * 32 * 32 
VIDEO_MAX_PIXELS = 300 * 32 * 32
VIDEO_TOTAL_PIXELS = 300 * 32 * 32 * 8
FPS = 2.0
FPS_MAX_FRAMES = 8

def get_embedding_reps(last_hidden_state, input_ids, embedding_token_id):
    embedding_idx = (input_ids == embedding_token_id)
    batch_size = last_hidden_state.shape[0]
    embedding_pos = []
    for i in range(batch_size):
        positions = torch.where(embedding_idx[i])[0]
        embedding_pos.append(positions[-1])
    embedding_pos = torch.tensor(embedding_pos, device=last_hidden_state.device)
    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), embedding_pos]
    return reps

def prepare_single_message(text, video_paths=None, video_params=None, max_text_length=512):
    text = text.strip()[:max_text_length]
    user_content = [{"type": "text", "text": text}]
    
    if video_paths and len(video_paths) > 0:
        video_params = video_params or {}
        default_params = {
            "total_pixels": VIDEO_TOTAL_PIXELS,
            "min_pixels": VIDEO_MIN_PIXELS,
            "max_pixels": VIDEO_MAX_PIXELS,
            "max_frames": FPS_MAX_FRAMES,
            "sample_fps": FPS,
        }
        default_params.update(video_params)
        user_content.append({
            "video": video_paths,
            "total_pixels": default_params["total_pixels"],
            "min_pixels": default_params["min_pixels"],
            "max_pixels": default_params["max_pixels"],
            "max_frames": default_params["max_frames"],
            "sample_fps": default_params["sample_fps"]
        })
    
    user_content.append({"type": "text", "text": "Summarize the above content in one word:"})
    
    return [{
        "role": "user",
        "content": user_content,
    }, {
        "role": "assistant",
        "content": [{"type": "text", "text": "<emb>"}],
    }]

def get_embedding(model, processor, emb_token_id, device, text, video_paths=None, video_params=None):
    messages = prepare_single_message(text, video_paths, video_params)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info([messages])
    
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        outputs = model.model(**model_inputs, return_dict=True)
    embed = get_embedding_reps(outputs.last_hidden_state, model_inputs["input_ids"], emb_token_id)
    return F.normalize(embed, dim=-1).squeeze(0)

def calculate_query_pos_similarity(model_path, sample_dict):
    device = torch.device("cuda")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True
    )
    
    emb_token = "<emb>"
    processor.tokenizer.add_tokens([emb_token])
    model.resize_token_embeddings(len(processor.tokenizer))
    emb_token_id = processor.tokenizer.convert_tokens_to_ids(emb_token)
    
    query_embed = get_embedding(
        model, processor, emb_token_id, device,
        text=sample_dict["qry_text"],
        video_paths=sample_dict["qry_video_paths"],
        video_params=sample_dict.get("video_params")
    )
    pos_embed = get_embedding(
        model, processor, emb_token_id, device,
        text=sample_dict["pos_text"],
        video_paths=None,
        video_params=None
    )
    
    similarity = torch.dot(query_embed, pos_embed).item()
    torch.cuda.empty_cache()
    return similarity

if __name__ == "__main__":
    MODEL_PATH = "./ckpt/Embed-RL-2B"
    test_sample = {
        "qry_text": "What is the dominant type of plant showcased in the video?",
        "qry_video_paths": [
            "./eval/video/1.jpeg",
            "./eval/video/2.jpeg",
            "./eval/video/3.jpeg",
            "./eval/video/4.jpeg",
            "./eval/video/5.jpeg",
            "./eval/video/6.jpeg",
            "./eval/video/7.jpeg",
            "./eval/video/8.jpeg"
        ],
        "video_params": {
            "total_pixels": VIDEO_TOTAL_PIXELS,
            "min_pixels": VIDEO_MIN_PIXELS,
            "max_pixels": VIDEO_MAX_PIXELS,
            "max_frames": FPS_MAX_FRAMES,
            "fps": FPS,
        },
        "pos_text": "The most dominant type of plant in the video is a collection of purple flowering plants with vibrant green foliage."
    }
    
    similarity = calculate_query_pos_similarity(MODEL_PATH, test_sample)
    print(f"Query-Target cosine similarity: {similarity:.6f}")