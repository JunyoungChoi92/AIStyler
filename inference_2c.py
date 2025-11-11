#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korea Times Style Correction Model - Inference Script
Checkpoint: stage_2C_mixed_review/checkpoint-466

Usage:
    python inference.py --input "your_article.txt" --output "corrected.json"
    python inference.py --interactive  # Interactive mode
"""

import os
import json
import re
import argparse
from pathlib import Path
import torch
from unsloth import FastLanguageModel


# ============================================================================
# Style Guide Loading
# ============================================================================

def load_style_guides(style_guide_file="style_guides/complete_style_guides_extracted.txt"):
    """스타일가이드 파일에서 컴포넌트별 규칙 추출"""
    with open(style_guide_file, 'r', encoding='utf-8') as f:
        content = f.read()

    rules_by_component = {
        "title": {},
        "body": {},
        "caption": {}
    }

    # Title 규칙 (T01-T11 또는 H01-H11)
    title_section = re.search(r'## 1\. (?:헤드라인|타이틀|제목) 스타일가이드.*?(?=## 2\.)', content, re.DOTALL)
    if title_section:
        for i in range(1, 20):
            for prefix in ['H', 'T']:
                rule_id = f"{prefix}{i:02d}"
                pattern = f"### Style Guide {rule_id}\n- (.*?)(?=\n(?:Correct|###|##))"
                match = re.search(pattern, title_section.group(0), re.DOTALL)
                if match:
                    rules_by_component["title"][rule_id] = match.group(1).strip()

    # Body 규칙 (A01-A39)
    body_section = re.search(r'## 2\. 기사 본문 스타일가이드.*?(?=## 3\.)', content, re.DOTALL)
    if body_section:
        for i in range(1, 50):
            rule_id = f"A{i:02d}"
            pattern = f"### Style Guide {rule_id}\n- (.*?)(?=\n(?:Correct|###|##))"
            match = re.search(pattern, body_section.group(0), re.DOTALL)
            if match:
                rules_by_component["body"][rule_id] = match.group(1).strip()

    # Caption 규칙 (C01-C33)
    caption_section = re.search(r'## 3\. 캡션 스타일가이드.*', content, re.DOTALL)
    if caption_section:
        for i in range(1, 50):
            rule_id = f"C{i:02d}"
            pattern = rf"### Style Guide {rule_id}\n- (.*?)(?=\n(?:Correct|###|##|\Z))"
            match = re.search(pattern, caption_section.group(0), re.DOTALL)
            if match:
                rules_by_component["caption"][rule_id] = match.group(1).strip()

    return rules_by_component


def format_rules_compact(component_type, rules_dict):
    """컴포넌트별 규칙을 compact 형식으로 포맷"""
    component_rules = rules_dict.get(component_type, {})

    if not component_rules:
        return "No rules available."

    formatted = []
    for rule_id, description in component_rules.items():
        summary = description.split('.')[0] + '.'
        formatted.append(f"{rule_id}: {summary}")

    return "\n".join(formatted)


# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path="checkpoint", base_model="unsloth/Qwen3-8B-unsloth-bnb-4bit"):
    """
    Load model from checkpoint

    If checkpoint_path contains adapter files, it will load the fine-tuned model.
    Otherwise, it will download the base model from Hugging Face.
    """
    print(f"Loading model from: {checkpoint_path}")

    # Check if checkpoint exists locally
    if os.path.exists(checkpoint_path) and os.path.exists(f"{checkpoint_path}/adapter_config.json"):
        model_name = checkpoint_path
        print("✓ Using local checkpoint")
    else:
        model_name = base_model
        print(f"✓ Downloading base model: {base_model}")
        print("  (This will be cached for future use)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
        load_in_4bit=True,
        load_in_8bit=False,
    )

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    print("✓ Model loaded successfully")
    return model, tokenizer


# ============================================================================
# Inference
# ============================================================================

def create_inference_prompt(article_text, rules_by_component):
    """
    Create inference prompt with ALL style guide rules
    (Train-Inference consistency: includes all 83 rules)
    """
    # Format ALL rules for all components
    all_rules = []
    for comp in ['title', 'body', 'caption']:
        rules_text = format_rules_compact(comp, rules_by_component)
        all_rules.append(f"**{comp.upper()} RULES:**\n{rules_text}")

    system_content = f"""You are a Korea Times style guide expert. Follow these steps:

**COMPONENT-SPECIFIC STYLE GUIDES:**

{chr(10).join(all_rules)}

**INPUT FORMAT:**
The text is structured with XML-like tags to clearly separate components:
- [TITLE]...[/TITLE]: Article headline/title
- [BODY]...[/BODY]: Article main content
- [CAPTION]...[/CAPTION]: Image/photo caption

**CRITICAL INSTRUCTIONS:**
1. **Apply rules ONLY to the matching component:**
   - TITLE rules (T01-T11, H01-H11) → Apply ONLY within [TITLE]...[/TITLE]
   - BODY rules (A01-A39) → Apply ONLY within [BODY]...[/BODY]
   - CAPTION rules (C01-C33) → Apply ONLY within [CAPTION]...[/CAPTION]

2. **Process each component separately:**
   - Read each tagged section independently
   - Check against that component's specific rules
   - Do NOT apply title rules to body, or body rules to caption, etc.

3. **Preserve component tags in output:**
   - Keep [TITLE]...[/TITLE], [BODY]...[/BODY], [CAPTION]...[/CAPTION] structure
   - Only fix the content inside tags

**OUTPUT SCHEMA:**
{{
  "corrected_text": "...",  // Must maintain [TITLE], [BODY], [CAPTION] tags
  "violations": [
    {{
      "rule_id": "...",
      "component_type": "title|body|caption",
      "violation_type": "...",
      "original_text": "...",
      "violated_text": "...",
      "description": "..."
    }}
  ]
}}"""

    user_content = f"Correct this text:\n\n{article_text}"

    return system_content, user_content


def run_inference(model, tokenizer, article_text, rules_by_component,
                  temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=2048):
    """Run inference on article text"""

    # Create prompt
    system_content, user_content = create_inference_prompt(article_text, rules_by_component)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Non-thinking mode (consistent with training)
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Korea Times Style Correction Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_2c_466",
                       help="Path to checkpoint directory")
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen3-8B-unsloth-bnb-4bit",
                       help="Base model to use if checkpoint not found")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)

    args = parser.parse_args()

    print("="*70)
    print("Korea Times Style Correction - Inference")
    print("Checkpoint: stage_2C_mixed_review/checkpoint-466")
    print("="*70)

    # Load style guides
    print("\nLoading style guides...")
    rules_by_component = load_style_guides()
    print(f"✓ Loaded {len(rules_by_component['title'])} title rules")
    print(f"✓ Loaded {len(rules_by_component['body'])} body rules")
    print(f"✓ Loaded {len(rules_by_component['caption'])} caption rules")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.checkpoint, args.base_model)

    # Interactive mode
    if args.interactive:
        print("\n" + "="*70)
        print("Interactive Mode")
        print("Enter article text (use [TITLE], [BODY], [CAPTION] tags)")
        print("Type 'quit' or 'exit' to stop")
        print("="*70)

        while True:
            print("\n" + "-"*70)
            article = input("\nEnter article text:\n")

            if article.lower() in ['quit', 'exit']:
                break

            response = run_inference(
                model, tokenizer, article, rules_by_component,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_tokens
            )

            result = parse_json_response(response)

            print("\n" + "="*70)
            print("RESULT:")
            print("="*70)
            print(json.dumps(result, indent=2, ensure_ascii=False))

    # File mode
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            article = f.read()

        print(f"\nProcessing: {args.input}")

        response = run_inference(
            model, tokenizer, article, rules_by_component,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens
        )

        result = parse_json_response(response)

        # Save to output file
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✓ Results saved to: {args.output}")
        else:
            print("\n" + "="*70)
            print("RESULT:")
            print("="*70)
            print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        print("\nError: Provide --input or --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()
