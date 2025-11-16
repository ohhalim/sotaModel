# 파인튜닝 가이드

> 사전학습된 모델을 나만의 데이터로 커스터마이징하기

## 🎯 목표

이 가이드에서는:
- **사전학습 모델 활용**: Music Informer, ImprovNet 등
- **LoRA, QLoRA**: 효율적 파인튜닝
- **커스텀 데이터**: 특정 작곡가, 스타일, 악기
- **Few-shot Learning**: 적은 데이터로 학습

---

## 📂 디렉토리 구조

```
08_finetuning/
├── README.md                      # 이 파일
├── 01_lora/
│   ├── lora_wrapper.py            # LoRA 구현
│   ├── qlora.py                   # QLoRA (양자화)
│   └── README.md
├── 02_data_collection/
│   ├── scrape_midis.py            # MIDI 수집 도구
│   ├── filter_quality.py          # 품질 필터링
│   └── README.md
├── 03_finetuning_scripts/
│   ├── finetune_music_informer.py # Music Informer 파인튜닝
│   ├── finetune_improvnet.py      # ImprovNet 파인튜닝
│   ├── config.yaml                # 설정 파일
│   └── README.md
└── 04_evaluation/
    ├── evaluate.py                # 파인튜닝 모델 평가
    └── README.md
```

---

## 🚀 빠른 시작

### 1. 사전학습 모델 다운로드

```bash
# HuggingFace에서 다운로드 (가상)
from transformers import AutoModel

model = AutoModel.from_pretrained("music-ai/music-informer-base")
```

### 2. 커스텀 데이터 준비

```bash
cd 02_data_collection

# 특정 작곡가 MIDI 수집
python scrape_midis.py \
    --composer "Chopin" \
    --output_dir ./chopin_midis \
    --min_duration 30 \
    --max_duration 300

# 품질 필터링
python filter_quality.py \
    --input_dir ./chopin_midis \
    --output_dir ./chopin_midis_filtered
```

### 3. LoRA 파인튜닝

```bash
cd 03_finetuning_scripts

python finetune_music_informer.py \
    --base_model music-ai/music-informer-base \
    --data_dir ../02_data_collection/chopin_midis_filtered \
    --use_lora \
    --lora_rank 16 \
    --epochs 20 \
    --output_dir ./chopin_music_informer
```

### 4. 생성 및 평가

```bash
cd 04_evaluation

# 파인튜닝 전 vs 후 비교
python evaluate.py \
    --base_model music-ai/music-informer-base \
    --finetuned_model ../03_finetuning_scripts/chopin_music_informer \
    --test_prompts "Generate a nocturne"
```

---

## 💡 LoRA (Low-Rank Adaptation)

### 왜 LoRA?

**문제**: 전체 모델 파인튜닝은 비용이 큼
- Music Informer: 200M 파라미터 전체 학습 필요
- GPU 메모리 부족
- 시간 오래 걸림

**해결**: LoRA - 작은 adapter만 학습
- 전체 파라미터의 0.1%만 학습
- GPU 메모리 70% 절약
- 성능은 거의 동일

### LoRA 작동 원리

```
원본 Weight: W (d × d)
LoRA 추가: ΔW = A @ B  (d × r) @ (r × d)
           where r << d

새로운 출력: W @ x + ΔW @ x
           = W @ x + (A @ B) @ x

학습 파라미터: A, B만 (r이 작으면 매우 적음)
```

### 사용 예시

```python
from peft import LoraConfig, get_peft_model

# LoRA 설정
lora_config = LoraConfig(
    r=16,  # Rank (낮을수록 파라미터 적음)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # 어디에 적용할지
    lora_dropout=0.1,
    bias="none"
)

# 모델에 LoRA 적용
model = get_peft_model(base_model, lora_config)

# 학습 가능한 파라미터만 출력
model.print_trainable_parameters()
# trainable params: 589,824 || all params: 199,148,288 || trainable%: 0.3%
```

---

## 🎵 파인튜닝 시나리오

### 1. 특정 작곡가 스타일

**목표**: 쇼팽 스타일 피아노곡 생성

**데이터**:
- 쇼팽 MIDI 100-200개
- 야마하 Disklavier 고품질 녹음

**설정**:
```yaml
task: composer_style
base_model: music_informer
data:
  composer: Chopin
  num_files: 150
lora:
  rank: 16
  alpha: 32
training:
  epochs: 30
  batch_size: 8
  lr: 1e-4
```

**결과**:
- 쇼팽 특유의 rubato, 아르페지오
- 낭만주의 화성 진행

### 2. 특정 악기 조합

**목표**: 피아노 트리오 (피아노 + 바이올린 + 첼로)

**데이터**:
- 베토벤, 브람스, 슈베르트 피아노 트리오
- 50-100 MIDI 파일

**설정**:
```yaml
task: instrumentation
base_model: music_informer
data:
  instruments: [piano, violin, cello]
  num_files: 80
lora:
  rank: 24
training:
  epochs: 40
```

### 3. 재즈 서브장르

**목표**: Bebop 스타일 즉흥연주

**데이터**:
- Charlie Parker, Dizzy Gillespie 트랜스크립션
- 30-50 솔로

**설정**:
```yaml
task: jazz_subgenre
base_model: improvnet
data:
  style: bebop
  num_files: 40
lora:
  rank: 12
training:
  epochs: 50
  use_augmentation: true
```

### 4. 게임 음악

**목표**: 레트로 8-bit 게임 음악

**데이터**:
- 닌텐도, 세가 게임 음악 MIDI
- 100-200 파일

**설정**:
```yaml
task: game_music
base_model: music_informer
data:
  genre: chiptune
  num_files: 150
constraints:
  max_polyphony: 4  # 8-bit 제약
  pitch_range: [48, 84]  # 제한된 음역
```

---

## 📊 Few-Shot Learning

### 극소량 데이터로 학습

**문제**: 데이터가 5-10개밖에 없을 때

**해결**:
1. **Data Augmentation 극대화**
2. **Prompt Tuning** 대신 LoRA
3. **Regularization 강화**

### 예시: 5개 MIDI로 파인튜닝

```python
# 데이터 증강 파이프라인
augmentation_pipeline = [
    Transpose(semitones=range(-6, 7)),     # 12개 버전
    TimeStretch(factors=[0.9, 1.0, 1.1]),  # 3개 버전
    VelocityJitter(std=5),                 # 다양한 velocity
]

# 5 × 12 × 3 = 180개로 증강

# 강한 regularization
finetune_config = {
    'dropout': 0.3,  # 높은 dropout
    'weight_decay': 0.1,
    'early_stopping_patience': 5,
    'lora_rank': 8,  # 작은 rank
}
```

---

## 🔬 고급 기법

### 1. QLoRA (Quantized LoRA)

메모리를 더욱 절약

```python
from transformers import BitsAndBytesConfig

# 4-bit 양자화
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 모델 로드 (1/4 메모리)
model = AutoModelForCausalLM.from_pretrained(
    "music-ai/music-informer-base",
    quantization_config=bnb_config
)
```

### 2. Multi-task Finetuning

여러 작업 동시 학습

```python
# 작곡가 + 장르 동시 제어
tasks = {
    'composer': ['Chopin', 'Mozart', 'Bach'],
    'genre': ['nocturne', 'sonata', 'fugue']
}

# Multi-task head
class MultiTaskHead(nn.Module):
    def __init__(self):
        self.composer_head = nn.Linear(512, 3)
        self.genre_head = nn.Linear(512, 3)
```

### 3. Continual Learning

파인튜닝 후에도 기존 능력 유지

```python
# Elastic Weight Consolidation (EWC)
def ewc_loss(model, fisher_matrix, old_params, lambda_ewc=1000):
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher_matrix:
            loss += (fisher_matrix[name] * (param - old_params[name])**2).sum()
    return lambda_ewc * loss
```

---

## 📈 평가 메트릭

### 1. 스타일 유사도

```python
# 원본 작곡가 데이터와 생성 결과 비교
def style_similarity(generated_midis, reference_midis):
    # Pitch class histogram 비교
    gen_pch = compute_pitch_class_histogram(generated_midis)
    ref_pch = compute_pitch_class_histogram(reference_midis)

    similarity = cosine_similarity(gen_pch, ref_pch)
    return similarity
```

### 2. 음악 이론 준수

```python
# 화성, 대위법 규칙 체크
def theory_compliance(midi):
    violations = 0

    # 평행 5도, 8도 체크
    violations += check_parallel_fifths(midi)

    # 음역 체크
    violations += check_range(midi)

    return 1.0 - (violations / total_checks)
```

### 3. 사람 평가

```python
# A/B 테스트
human_evaluation = {
    'musicality': 8.5,  # 1-10
    'style_match': 9.0,
    'creativity': 7.5,
    'technical_quality': 8.8
}
```

---

## 💾 체크포인트 관리

### Best Practices

```python
# 여러 버전 저장
checkpoints = {
    'base': './checkpoints/base.pt',
    'epoch_10': './checkpoints/chopin_ep10.pt',
    'epoch_20': './checkpoints/chopin_ep20.pt',
    'best_val_loss': './checkpoints/chopin_best.pt',
}

# LoRA weights만 저장 (작은 용량)
torch.save(lora_weights, 'chopin_lora.pt')  # ~10MB
# vs 전체 모델: ~800MB
```

---

## 🎓 학습 로드맵

### Week 1-2: LoRA 이해
- LoRA 논문 읽기
- 간단한 LoRA 구현

### Week 3-4: 데이터 수집
- 목표 스타일 데이터 100개 수집
- 품질 필터링, 전처리

### Week 5-6: 파인튜닝
- Music Informer + LoRA
- 하이퍼파라미터 튜닝

### Week 7-8: 평가 및 개선
- 생성 품질 평가
- 반복 개선

---

## 📖 참고 자료

### 논문
- [LoRA (2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA (2023)](https://arxiv.org/abs/2305.14314)
- [Prefix Tuning (2021)](https://arxiv.org/abs/2101.00190)

### 라이브러리
- [PEFT (HuggingFace)](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

## ✅ 완료 체크리스트

- [ ] LoRA 개념 이해
- [ ] 커스텀 데이터 100개 이상 수집
- [ ] Music Informer 파인튜닝 성공
- [ ] 스타일 유사도 > 0.8
- [ ] 고품질 음악 생성
- [ ] 파인튜닝 모델 배포

**축하합니다! 나만의 음악 AI 모델을 만들었습니다!** 🎉
