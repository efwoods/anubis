import logging
logger = logging.getLogger(__name__)

# AUDIO PROCESSING

from functools import lru_cache

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

@lru_cache(maxsize=1)
def get_whisper_pipeline():
    """Load and cache the Whisper model pipeline"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        dtype=dtype,
        device=device,
    )
    
    return pipe

