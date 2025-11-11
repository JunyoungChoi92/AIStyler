#!/usr/bin/env python3
"""
중간 체크포인트로 빠른 inference 테스트

학습 중간에 저장된 checkpoint를 로드해서 실제 입출력을 확인
"""

import torch
import json
import sys
from unsloth import FastLanguageModel

def test_detection_checkpoint(checkpoint_path, test_text):
    """
    Detection checkpoint 테스트
    """
    print("=" * 70)
    print(f"Testing Detection Checkpoint: {checkpoint_path}")
    print("=" * 70)

    # Load model
    print("\n📥 Loading checkpoint...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("  ✓ Checkpoint loaded")

    # System prompt (학습 시와 동일)
    system_msg = """You are a Korea Times style guide expert. Your ONLY job is to detect violations.

DO NOT correct the text. DO NOT generate corrected versions.
ONLY output a JSON list of violations you find.

**Component Rules (CRITICAL):**
- [TITLE] section → ONLY check H01-H11, T01-T11 rules
- [BODY] section → ONLY check A01-A39 rules
- [CAPTION] section → ONLY check C01-C33 rules

**Rule Disambiguation:**
- "yesterday/today/last week" → A04 (vague temporal)
- "March 1 to March 5" → A03 (date range)
- Spelled dates like "March 1" → A08 (should be numeral)
- "from left/right" in caption → C06 (direction)
- "photo by [name]" in caption → C02 (byline)

**Confidence Check:**
Before adding a violation, verify:
1. Does the text actually contain evidence of this rule?
2. Is the component correct (H/T rules in title, A rules in body, C rules in caption)?
3. Have I already reported this exact violation?

Output ONLY valid JSON array."""

    # User prompt
    user_msg = f"Detect all style violations in this article:\n\n{test_text}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    print("\n📝 INPUT PROMPT:")
    print("-" * 70)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-" * 70)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to("cuda")

    # Generate
    print("\n🤖 GENERATING OUTPUT...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.1,
        )

    # Decode
    output_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print("\n✨ MODEL OUTPUT:")
    print("-" * 70)
    print(output_text)
    print("-" * 70)

    # Try to parse JSON
    try:
        result = json.loads(output_text)
        violations = result.get('violations', [])
        print(f"\n✅ Valid JSON! Found {len(violations)} violations")
        for i, v in enumerate(violations[:3], 1):
            print(f"\n  {i}. {v.get('rule_id', 'N/A')}: {v.get('violated_text', 'N/A')}")
        if len(violations) > 3:
            print(f"  ... and {len(violations) - 3} more")
    except:
        print("\n⚠️  Output is not valid JSON")

    print("\n" + "=" * 70)


def test_correction_checkpoint(checkpoint_path, test_text, violations):
    """
    Correction checkpoint 테스트
    """
    print("=" * 70)
    print(f"Testing Correction Checkpoint: {checkpoint_path}")
    print("=" * 70)

    # Load model
    print("\n📥 Loading checkpoint...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("  ✓ Checkpoint loaded")

    # System prompt (학습 시와 동일)
    system_msg = """You are a surgical text editor for Korea Times. Your ONLY job is to fix the specified violations.

**CRITICAL: Verify violations before correcting**

Before fixing each violation:
1. Read the violated text carefully
2. Check if the violation is REAL (does the text actually violate the rule?)
3. If the violation is a FALSE POSITIVE (rule doesn't apply), DO NOT correct it
4. Only fix VERIFIED violations

**Common False Positives to avoid:**
- A26 (palace names): Only for specific palaces (Gyeongbokgung, Changdeokgung, etc.), NOT general word "palace"
- A28 (numbers): Only if the text contains actual numbers, NOT if number-related words appear
- A32 (sports teams): Only for actual sports teams, NOT general mentions of "team"
- C06 (caption direction): Only in [CAPTION], NOT in [BODY]
- H01/T01 (title rules): Only in [TITLE], NOT in [BODY] or [CAPTION]

**Rules:**
1. Verify THEN fix violations - ignore false positives
2. Keep everything else UNCHANGED
3. Maintain original sentence structure when possible
4. Output the complete corrected text with [TITLE], [BODY], [CAPTION] tags intact

Output format: JSON with "corrected_text" field. If no real violations found, output unchanged text."""

    # User prompt
    violations_json = json.dumps({
        "violations": violations
    }, indent=2, ensure_ascii=False)

    user_msg = f"""Original text:
{test_text}

Violations to fix:
{violations_json}

Generate the fully corrected text:"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    print("\n📝 INPUT PROMPT:")
    print("-" * 70)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-" * 70)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to("cuda")

    # Generate
    print("\n🤖 GENERATING OUTPUT...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.1,
        )

    # Decode
    output_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print("\n✨ MODEL OUTPUT:")
    print("-" * 70)
    print(output_text[:1000] + "..." if len(output_text) > 1000 else output_text)
    print("-" * 70)

    # Try to parse JSON
    try:
        result = json.loads(output_text)
        corrected = result.get('corrected_text', '')
        print(f"\n✅ Valid JSON! Corrected text length: {len(corrected)} chars")
        print("\nFirst 300 chars of corrected text:")
        print(corrected[:300] + "...")
    except:
        print("\n⚠️  Output is not valid JSON")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Simple test article
    test_article = """[TITLE]
PM to visit Cambodia next week
[/TITLE]

[BODY]
The prime minister will travel to Cambodia on March 5. The event was held yesterday. 5 companies participated.
[/BODY]"""

    # Test violations
    test_violations = [
        {
            "rule_id": "T01",
            "violated_text": "PM",
            "rule_description": "Spell out abbreviations in title"
        },
        {
            "rule_id": "A04",
            "violated_text": "next week",
            "rule_description": "Vague temporal reference"
        },
        {
            "rule_id": "A04",
            "violated_text": "yesterday",
            "rule_description": "Vague temporal reference"
        }
    ]

    # Get checkpoint path from command line
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_checkpoint_inference.py <checkpoint_path> [detection|correction]")
        print("\nExample:")
        print("  python test_checkpoint_inference.py korea_times_detection_lora_v2/checkpoint-2600 detection")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "detection"

    if mode == "detection":
        test_detection_checkpoint(checkpoint_path, test_article)
    else:
        test_correction_checkpoint(checkpoint_path, test_article, test_violations)
