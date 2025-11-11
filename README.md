# Korea Times AI Styler - Inference Package 

---

## 📦 패키지 내용

이 패키지는 추론(inference) 전용입니다. 3개의 학습된 모델과 추론 스크립트가 포함되어 있습니다.
단, 이 폴더의 모델은 학습 및 평가 과정에서 얻은 로라 어댑터로, 학습 등이 (ChatGPT를 사용한 데이터 증강의 여러 한계 문제로 인하여)완료되지 않은 상태에서 중간 보고 목적으로 보고드리는 것입니다. 
충분한 학습데이터(1만여건 기사 이상)가 모이면, 별도로 제공된 학습 스크립트를 통해 로컬 모델을 훈련하고 성과를 확인하실 수 있습니다. 
이하 모델 및 추론 코드에 대한 설명입니다. 

### 포함된 모델 (3개, 1.05GB)
- checkpoint_2c_466/ (518MB) - 2C 통합 모델 (빠른 단일 추론)
- detection_checkpoint_3300/ (267MB) - Detection 전용 모델
- correction_checkpoint_3200/ (267MB) - Correction 전용 모델

### 추론 스크립트 (6개)
- `inference_2c.py` - 2C 모델 추론 (통합)
- `inference_simple.py` - 간단한 추론
- `evaluate_v2_lora.py` - 2-Stage 추론 (Detection + Correction)
- `test_checkpoint_inference.py` - 체크포인트 테스트
- `test_model_loading.py` - 모델 로딩 테스트
- `quick_start.sh` - 빠른 시작

### 데이터 및 예제
- `test_samples_10.jsonl` - 테스트 샘플 10개
- `test_samples_50.jsonl` - 테스트 샘플 50개
- `examples/` - 예제 기사
- `style_guides/` - 83개 스타일 가이드 규칙

---

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install torch transformers unsloth peft bitsandbytes trl
```

### 2. 2C 모델 추론 (빠른 방법)

#### Interactive 모드
```bash
python inference_2c.py --interactive
```

입력 예시:
```
[TITLE]South Korea to invest W1.5 trillion in AI[/TITLE]
[BODY]The government announced plans yesterday.[/BODY]
```

#### 파일 모드
```bash
python inference_2c.py \
    --input examples/sample_article.txt \
    --output result.json
```

### 3. 2-Stage 추론 (더 정확한 방법)

```bash
python evaluate_v2_lora.py \
    --detection-lora detection_checkpoint_3300 \
    --correction-lora correction_checkpoint_3200 \
    --test-data test_samples_10.jsonl \
    --max-samples 10
```

---
## 📚 입력 형식

모든 모델은 XML 태그 형식을 사용합니다:

```
[TITLE]기사 제목[/TITLE]
[BODY]기사 본문 내용...[/BODY]
[CAPTION]사진 설명[/CAPTION]
```

- 모든 태그는 선택 사항 (최소 1개 필요)
- 여러 문단 가능
- 영문 기사 전용

---

## 📊 출력 형식

모든 추론은 JSON 형식으로 출력됩니다:

```json
{
  "corrected_text": "[TITLE]Corrected headline[/TITLE][BODY]...[/BODY]",
  "violations": [
    {
      "rule_id": "H03",
      "component_type": "title",
      "violation_type": "capitalization",
      "original_text": "south korea",
      "violated_text": "south korea",
      "description": "Use title case for headlines"
    }
  ]
}
```

---

## 🔧 고급 사용법

### 2C 파라미터 조정

```bash
python inference_2c.py \
    --input article.txt \
    --output result.json \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-tokens 2048
```

파라미터 설명:
- `--temperature`: 0.1-1.0 (낮을수록 일관성↑, 높을수록 창의성↑)
- `--top-p`: 0.1-1.0 (Nucleus sampling)
- `--max-tokens`: 512-4096 (출력 최대 길이)

### 2-Stage 배치 처리

```bash
# 50개 샘플 모두 처리
python evaluate_v2_lora.py \
    --detection-lora detection_checkpoint_3300 \
    --correction-lora correction_checkpoint_3200 \
    --test-data test_samples_50.jsonl \
    --max-samples 50
```

---

## ⚙️ 시스템 요구사항

### 최소 사양
- GPU: NVIDIA GPU with 8GB VRAM
- RAM: 16GB
- 저장공간: 2GB
- CUDA: 11.8 이상

### 권장 사양
- GPU: NVIDIA RTX 3090 / A100 (24GB VRAM)
- RAM: 32GB
- 저장공간: 5GB
- CUDA: 12.1 이상

---

## 🐛 문제 해결

### CUDA Out of Memory

2C 모델:
```bash
# max-tokens 줄이기
python inference_2c.py --max-tokens 1024 --input article.txt
```

2-Stage 모델:
- 샘플 수 줄이기 (`--max-samples 1`)
- 다른 GPU 프로세스 종료
- 배치 사이즈 확인

### 모델 로딩 실패

```bash
# 모델 파일 확인
ls -lh checkpoint_2c_466/adapter_model.safetensors  # 334MB여야 함
ls -lh detection_checkpoint_3300/adapter_model.safetensors  # 167MB
ls -lh correction_checkpoint_3200/adapter_model.safetensors  # 167MB

# 모델 로딩 테스트
python test_model_loading.py
```

### JSON 파싱 오류

모델이 잘못된 JSON을 출력할 경우:
- `--temperature` 낮추기 (0.3-0.5)
- `--top-p` 낮추기 (0.7-0.8)
- 입력 형식 확인 (태그 올바른지)

---

## 📖 스타일 가이드 규칙

### Title Rules (11개: H01-H11)
- H01: 헤드라인 대문자 규칙
- H02: 날짜 형식
- H03: 숫자 표기
- ... (총 11개)

### Body Rules (39개: A01-A39)
- A01: 문단 구조
- A02: 인용문 형식
- A03: 약어 사용
- ... (총 39개)

### Caption Rules (33개: C01-C33)
- C01: 캡션 시제
- C02: 사진 설명 형식
- C03: 위치 표기
- ... (총 33개)

---

## 🔍 성능 참고

### 2C 모델 (checkpoint-466)
- 학습 진행: 25% (1 epoch)
- 학습 데이터: Korea Times articles
- 특징: 빠르고 균형잡힌 성능

### Detection 모델 (checkpoint-3300)
- 학습 진행: 81% (2.4/3 epochs)
- 특징: 위반 탐지 특화

### Correction 모델 (checkpoint-3200)
- 학습 진행: 61% (2.4/4 epochs)
- 특징: 교정 생성 특화

---

## 📝 사용 예제

### 예제 1: 단일 기사 빠른 검토

```bash
# 2C 모델로 빠르게 검토
python inference_2c.py --interactive

# 입력
[TITLE]PM visits cambodia next week[/TITLE]
[BODY]The event was held yesterday. 5 companies participated.[/BODY]

# 출력: JSON with violations and corrections
```

### 예제 2: 중요 기사 정밀 검토

```bash
# 2-Stage로 정확하게 검토
python evaluate_v2_lora.py \
    --detection-lora detection_checkpoint_3300 \
    --correction-lora correction_checkpoint_3200 \
    --test-data my_important_article.jsonl \
    --max-samples 1
```

### 예제 3: 배치 처리

```bash
# 50개 기사 한번에 처리
python evaluate_v2_lora.py \
    --detection-lora detection_checkpoint_3300 \
    --correction-lora correction_checkpoint_3200 \
    --test-data test_samples_50.jsonl \
    --max-samples 50
```

---

## ❓ FAQ

### Q1: 학습은 가능한가요?
A: 아니요. 이 패키지는 추론 전용입니다. 

### Q2: 2C와 2-Stage 중 어떤 것을 써야 하나요?
A:
- 빠른 검토 -> 2C
- 정확한 최종 검토 -> 2-Stage
- 실시간 편집 -> 2C
- 출판 전 검증 -> 2-Stage

### Q3: GPU 없이 사용 가능한가요?
A: 이론적으로 가능하지만 매우 느립니다 (100배 이상). GPU 권장합니다.

### Q4: 한국어 기사도 되나요?
A: 아니요. 영문 기사 전용입니다.

### Q5: 모델 성능은 어떤가요?
A:
- 2C: 빠르고 균형잡힌 성능
- 2-Stage: 성능이 조금 더 좋음

### Q6: 다른 언론사에서도 사용 가능한가요?
A: 모델은 Korea Times 스타일에 특화되어 있습니다. 

### Q7: API로 제공되나요?
A: 현재는 CLI만 제공됩니다. API 필요 시 Flask/FastAPI로 감싸서 사용하세요.