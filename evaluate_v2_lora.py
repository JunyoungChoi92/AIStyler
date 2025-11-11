#!/usr/bin/env python3
"""
Korea Times V2 LoRA Evaluation Script

학습 시 사용한 프롬프트와 정확히 동일한 조건으로 모델 평가

Features:
- Detection LoRA 평가 (Rule F1, Precision, Recall)
- Correction LoRA 평가 (Exact Match, Similarity)
- 학습 프롬프트와 100% 동일한 형식 사용
"""

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple
from unsloth import FastLanguageModel
from difflib import SequenceMatcher


# ============================================================================
# TRAINING PROMPTS (from prepare_detection_dataset.py and prepare_correction_dataset.py)
# ============================================================================

def create_detection_system_prompt():
    """
    Detection 전용 System Prompt (학습 시 사용한 것과 정확히 동일)

    From: prepare_detection_dataset.py
    """
    return """You are a Korea Times style guide expert. Your ONLY job is to detect violations.

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


def create_correction_system_prompt():
    """
    Correction 전용 System Prompt (학습 시 사용한 것과 정확히 동일)

    From: prepare_correction_dataset.py
    """
    return """You are a surgical text editor for Korea Times. Your ONLY job is to fix the specified violations.

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


# ============================================================================
# MODEL LOADING
# ============================================================================

class V2LoRAEvaluator:
    """
    V2 LoRA 평가기 (학습 조건과 동일)
    """

    def __init__(
        self,
        detection_lora_path: str,
        correction_lora_path: str,
        device: str = "cuda"
    ):
        self.device = device

        print("=" * 70)
        print("Loading V2 LoRA Models")
        print("=" * 70)

        # Load Detection LoRA
        print("\n📥 Loading Detection LoRA...")
        self.detection_model, self.detection_tokenizer = self._load_lora(detection_lora_path)
        FastLanguageModel.for_inference(self.detection_model)
        print("  ✓ Detection LoRA loaded")

        # Load Correction LoRA
        print("\n📥 Loading Correction LoRA...")
        self.correction_model, self.correction_tokenizer = self._load_lora(correction_lora_path)
        FastLanguageModel.for_inference(self.correction_model)
        print("  ✓ Correction LoRA loaded")

        print("\n✅ All models loaded successfully")

    def _load_lora(self, lora_path: str):
        """Load LoRA adapter"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        return model, tokenizer


    # ========================================================================
    # DETECTION INFERENCE (학습 조건과 동일)
    # ========================================================================

    def detect_violations(self, article_text: str) -> List[Dict]:
        """
        Detection inference (학습 시와 동일한 프롬프트)

        Args:
            article_text: Article with [TITLE], [BODY], [CAPTION] tags

        Returns:
            List of violations
        """
        # System message (학습 시와 동일)
        system_msg = create_detection_system_prompt()

        # User message (학습 시와 동일)
        user_msg = f"Detect all style violations in this article:\n\n{article_text}"

        # Messages
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        # Apply chat template (학습 시와 동일)
        prompt = self.detection_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # ⭐ Non-thinking mode
        )

        # Tokenize
        inputs = self.detection_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        # Generate (학습 시와 동일한 설정)
        with torch.no_grad():
            outputs = self.detection_model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                do_sample=True,
                repetition_penalty=1.1,
            )

        # Decode
        output_text = self.detection_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Parse JSON
        try:
            result = json.loads(output_text)
            violations = result.get('violations', [])
        except json.JSONDecodeError:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    violations = result.get('violations', [])
                except:
                    violations = []
            else:
                violations = []

        return violations


    # ========================================================================
    # CORRECTION INFERENCE (학습 조건과 동일)
    # ========================================================================

    def correct_violations(self, article_text: str, violations: List[Dict]) -> str:
        """
        Correction inference (학습 시와 동일한 프롬프트)

        Args:
            article_text: Original article text
            violations: List of violations to fix

        Returns:
            Corrected text
        """
        # System message (학습 시와 동일)
        system_msg = create_correction_system_prompt()

        # User message (학습 시와 동일한 형식)
        violations_json = json.dumps({
            "violations": violations
        }, indent=2, ensure_ascii=False)

        user_msg = f"""Original text:
{article_text}

Violations to fix:
{violations_json}

Generate the fully corrected text:"""

        # Messages
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        # Apply chat template (학습 시와 동일)
        prompt = self.correction_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # ⭐ Non-thinking mode
        )

        # Tokenize
        inputs = self.correction_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        # Generate (학습 시와 동일한 설정)
        with torch.no_grad():
            outputs = self.correction_model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                do_sample=True,
                repetition_penalty=1.1,
            )

        # Decode
        output_text = self.correction_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Parse JSON
        try:
            result = json.loads(output_text)
            corrected_text = result.get('corrected_text', article_text)
        except json.JSONDecodeError:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    corrected_text = result.get('corrected_text', article_text)
                except:
                    corrected_text = article_text
            else:
                corrected_text = output_text.strip()

        return corrected_text


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_detection_metrics(predictions: List[List[Dict]],
                                ground_truth: List[List[Dict]]) -> Dict:
    """
    Calculate detection metrics

    Args:
        predictions: List of predicted violations per article
        ground_truth: List of true violations per article

    Returns:
        Dict with precision, recall, f1
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    rule_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for pred_violations, true_violations in zip(predictions, ground_truth):
        # Extract rule IDs
        pred_rules = set(v.get('rule_id', '') for v in pred_violations)
        true_rules = set(v.get('rule_id', '') for v in true_violations)

        # Calculate TP, FP, FN
        tp = len(pred_rules & true_rules)
        fp = len(pred_rules - true_rules)
        fn = len(true_rules - pred_rules)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-rule stats
        for rule in pred_rules & true_rules:
            rule_stats[rule]['tp'] += 1
        for rule in pred_rules - true_rules:
            rule_stats[rule]['fp'] += 1
        for rule in true_rules - pred_rules:
            rule_stats[rule]['fn'] += 1

    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-rule F1
    rule_f1 = {}
    for rule, stats in rule_stats.items():
        p = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        r = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        rule_f1[rule] = {'precision': p, 'recall': r, 'f1': f}

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'rule_f1': rule_f1
    }


def calculate_correction_metrics(predictions: List[str],
                                 ground_truth: List[str]) -> Dict:
    """
    Calculate correction metrics

    Args:
        predictions: List of predicted corrected texts
        ground_truth: List of true corrected texts

    Returns:
        Dict with exact_match, similarity
    """
    exact_matches = 0
    similarities = []

    for pred, true in zip(predictions, ground_truth):
        # Exact match
        if pred.strip() == true.strip():
            exact_matches += 1

        # Similarity
        similarity = SequenceMatcher(None, pred, true).ratio()
        similarities.append(similarity)

    return {
        'exact_match_rate': exact_matches / len(predictions) if predictions else 0,
        'avg_similarity': sum(similarities) / len(similarities) if similarities else 0,
        'exact_matches': exact_matches,
        'total': len(predictions)
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_v2_lora(
    detection_lora_path: str,
    correction_lora_path: str,
    test_data_path: str,
    output_path: str = None,
    max_samples: int = None
):
    """
    Main evaluation function
    """
    print("=" * 70)
    print("Korea Times V2 LoRA Evaluation")
    print("=" * 70)

    # Load test data
    print(f"\n📥 Loading test data from {test_data_path}...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"  ✓ Loaded {len(test_data)} test samples")

    # Initialize evaluator
    evaluator = V2LoRAEvaluator(
        detection_lora_path=detection_lora_path,
        correction_lora_path=correction_lora_path
    )

    # Evaluation
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)

    detection_predictions = []
    detection_ground_truth = []
    correction_predictions = []
    correction_ground_truth = []

    results = []

    for sample in tqdm(test_data, desc="Evaluating"):
        article_id = sample.get('article_id', 'unknown')
        violated_text = sample.get('violated_text', '')
        true_violations = sample.get('violations', [])
        true_corrected = sample.get('corrected_text', violated_text)

        # Detection
        pred_violations = evaluator.detect_violations(violated_text)
        detection_predictions.append(pred_violations)
        detection_ground_truth.append(true_violations)

        # Correction
        pred_corrected = evaluator.correct_violations(violated_text, pred_violations)
        correction_predictions.append(pred_corrected)
        correction_ground_truth.append(true_corrected)

        # Store result
        results.append({
            'article_id': article_id,
            'violated_text': violated_text,
            'true_violations': true_violations,
            'pred_violations': pred_violations,
            'true_corrected': true_corrected,
            'pred_corrected': pred_corrected
        })

    # Calculate metrics
    print("\n📊 Calculating metrics...")

    detection_metrics = calculate_detection_metrics(detection_predictions, detection_ground_truth)
    correction_metrics = calculate_correction_metrics(correction_predictions, correction_ground_truth)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n🔍 Detection LoRA Performance:")
    print(f"  Precision: {detection_metrics['precision']:.4f}")
    print(f"  Recall:    {detection_metrics['recall']:.4f}")
    print(f"  F1 Score:  {detection_metrics['f1']:.4f}")
    print(f"  TP: {detection_metrics['total_tp']}, FP: {detection_metrics['total_fp']}, FN: {detection_metrics['total_fn']}")

    print("\n✏️  Correction LoRA Performance:")
    print(f"  Exact Match Rate: {correction_metrics['exact_match_rate']:.4f}")
    print(f"  Avg Similarity:   {correction_metrics['avg_similarity']:.4f}")
    print(f"  Exact Matches: {correction_metrics['exact_matches']} / {correction_metrics['total']}")

    # Top/Bottom rules
    print("\n📈 Top 5 Rules by F1:")
    sorted_rules = sorted(detection_metrics['rule_f1'].items(),
                         key=lambda x: x[1]['f1'], reverse=True)
    for rule, stats in sorted_rules[:5]:
        print(f"  {rule}: F1={stats['f1']:.3f}, P={stats['precision']:.3f}, R={stats['recall']:.3f}")

    print("\n📉 Bottom 5 Rules by F1:")
    for rule, stats in sorted_rules[-5:]:
        print(f"  {rule}: F1={stats['f1']:.3f}, P={stats['precision']:.3f}, R={stats['recall']:.3f}")

    # Save results
    if output_path:
        print(f"\n💾 Saving results to {output_path}...")
        output_data = {
            'detection_metrics': detection_metrics,
            'correction_metrics': correction_metrics,
            'results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved to {output_path}")

    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)

    return detection_metrics, correction_metrics, results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Korea Times V2 LoRA")

    parser.add_argument(
        '--detection-lora',
        type=str,
        default='./korea_times_detection_lora_v2_final',
        help='Path to detection LoRA'
    )

    parser.add_argument(
        '--correction-lora',
        type=str,
        default='./korea_times_correction_lora_v2_final',
        help='Path to correction LoRA'
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default='stage2_2000/stage2_balanced_val.jsonl',
        help='Path to test data'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results_v2.json',
        help='Output path for results'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluate_v2_lora(
        detection_lora_path=args.detection_lora,
        correction_lora_path=args.correction_lora,
        test_data_path=args.test_data,
        output_path=args.output,
        max_samples=args.max_samples
    )
