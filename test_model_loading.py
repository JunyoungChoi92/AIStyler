#!/usr/bin/env python3
"""Quick test to check model loading"""

import torch
from unsloth import FastLanguageModel

print("Testing model loading...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    # Try to load the checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="checkpoint",
        max_seq_length=8192,
        load_in_4bit=True,
        load_in_8bit=False,
    )
    print("✓ Model loaded successfully from checkpoint")
    
    # Test tokenizer
    test_text = "Hello world"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"✓ Tokenizer works: '{test_text}' -> {len(tokens['input_ids'][0])} tokens")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nTrying to load base model instead...")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
            max_seq_length=8192,
            load_in_4bit=True,
            load_in_8bit=False,
        )
        print("✓ Base model would need to be downloaded")
    except Exception as e2:
        print(f"✗ Error: {e2}")