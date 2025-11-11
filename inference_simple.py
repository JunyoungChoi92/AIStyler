#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple inference without full style guide rules in prompt
"""

import os
import json
import re
import torch
from unsloth import FastLanguageModel

def load_model(checkpoint_path="checkpoint"):
    """Load model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=8192,
        load_in_4bit=True,
        load_in_8bit=False,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    print("✓ Model loaded successfully")
    return model, tokenizer


def create_simple_prompt(article_text):
    """
    Create a simplified inference prompt WITHOUT full style guide rules
    """
    system_content = """You are a Korea Times style guide expert. 

Your task is to:
1. Correct the given article text according to Korea Times style guidelines
2. Identify and list all style guide violations

The text is structured with XML-like tags:
- [TITLE]...[/TITLE]: Article headline
- [BODY]...[/BODY]: Article main content  
- [CAPTION]...[/CAPTION]: Image/photo caption

Please output in JSON format:
{
  "corrected_text": "...",  // Corrected text with tags preserved
  "violations": [
    {
      "rule_id": "...",
      "component_type": "title|body|caption",
      "violation_type": "...",
      "original_text": "...",
      "violated_text": "...",
      "description": "..."
    }
  ]
}"""

    user_content = f"Correct this text:\n\n{article_text}"
    
    return system_content, user_content


def run_inference(model, tokenizer, article_text,
                  temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=2048):
    """Run inference on article text"""
    
    # Create prompt
    system_content, user_content = create_simple_prompt(article_text)
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Non-thinking mode
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Generate
    print("\nGenerating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response


def parse_json_response(response):
    """Extract and parse JSON from model response"""
    # Try to find JSON block
    json_match = re.search(r'\{[\s\S]*\}', response)
    
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            return result
        except json.JSONDecodeError as e:
            print(f"Warning: JSON parsing failed: {e}")
            return {"raw_response": response, "error": "JSON parsing failed"}
    
    return {"raw_response": response, "error": "No JSON found"}


def main():
    print("="*70)
    print("Korea Times Style Correction - Simple Prompt Test")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model("checkpoint")
    
    # Load test article
    with open('examples/sample_article.txt', 'r', encoding='utf-8') as f:
        article = f.read()
    
    print(f"\nProcessing: examples/sample_article.txt")
    
    # Run inference
    response = run_inference(
        model, tokenizer, article,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=2048
    )
    
    # Parse response
    result = parse_json_response(response)
    
    # Save to output file
    with open('test_result_simple.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved to: test_result_simple.json")
    
    # Display summary
    if "violations" in result:
        print(f"\n✓ Found {len(result['violations'])} violations")
        for v in result['violations']:
            print(f"  - {v['rule_id']} ({v['component_type']}): {v['violation_type']}")
    else:
        print(f"\n⚠ JSON parsing issues - check test_result_simple.json for raw output")


if __name__ == "__main__":
    main()