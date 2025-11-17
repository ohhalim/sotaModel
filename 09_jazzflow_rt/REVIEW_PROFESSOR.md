# 🎓 JazzFlow-RT 논문 심사 보고서

**심사위원**: Prof. [Music Information Retrieval & AI]
**심사 날짜**: 2025-11-17
**판정**: **Major Revision Required** ⚠️

---

## 📊 종합 평가

| 항목 | 점수 | 비고 |
|------|------|------|
| **기술적 타당성** | 3/10 | 심각한 구현 오류 다수 |
| **실험 검증** | 0/10 | **실험 결과 전무** |
| **재현 가능성** | 4/10 | 코드는 있으나 작동 불확실 |
| **혁신성** | 5/10 | 아이디어는 좋으나 검증 없음 |
| **논문 품질** | 2/10 | 주장만 있고 증거 없음 |

**종합 점수**: **2.8/10** (Reject → Major Revision)

---

## 🔴 Critical Issues (필수 수정)

### 1. **실험 결과가 전혀 없음** ❌

**문제점:**
```markdown
README.md:
## 🔬 실험 결과 (예상)  ← 이게 문제!

| 모델 | 지연시간 | RTF |
|------|----------|-----|
| **JazzFlow-RT (ours)** | **45ms** | **2.2x** |
```

**심사평:**
- 모든 성능 수치가 "예상(expected)"
- 실제 실험을 단 한 번도 수행하지 않음
- 이것은 논문이 아니라 "제안서"
- **ICML, NeurIPS 등 어떤 학회도 accept 불가능**

**필수 수정:**
1. MAESTRO 데이터셋으로 실제 학습
2. 최소 3개 baseline과 비교 (Music Informer, ImprovNet, MusicGen)
3. 실제 측정된 지연시간, 메모리, 품질 점수
4. 통계적 유의성 검증 (t-test, p<0.05)

---

### 2. **Chord Extraction이 더미 구현** ❌

**코드:**
```python
# data_processing/dataset.py:143
def _extract_chords(self, midi_path: Path):
    """코드 진행 추출 (간단한 버전)"""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        # 간단한 코드 감지: 동시발음 노트들로부터 코드 유추
        # TODO: 더 정교한 코드 감지 알고리즘
        chord_ids = [0] * (len(self.tokenizer.vocab))  # Dummy ← 문제!

        return chord_ids[:self.seq_len + 1]
```

**심사평:**
- 핵심 기능이 "TODO"로 남아있음
- 코드 진행은 재즈 생성의 **필수 요소**인데 더미 구현
- 이것으로는 "Chord-Conditioned Generation" 주장 불가능

**필수 수정:**
1. `music21` 사용한 실제 코드 감지 구현
2. 또는 PiJAMA 데이터셋의 코드 annotation 활용
3. Chord recognition accuracy 측정 (최소 80%+)

---

### 3. **QLoRA 구현이 작동하지 않음** ❌

**코드:**
```python
# training/finetune_qlora.py:89-102
def setup_qlora_model(base_model, ...):
    # 4-bit 양자화 (QLoRA)
    if quantization and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(...)
        base_model = prepare_model_for_kbit_training(base_model)

    # LoRA 설정
    lora_config = LoraConfig(
        target_modules=[
            "q_proj", "v_proj", "k_proj",  # ← 문제!
            "out_proj",
            "lm_head"
        ],
        ...
    )
```

**심사평:**
- `JazzFlowRT` 모델의 실제 모듈 이름과 불일치
- 실제 모듈 이름: `self.attn.q_proj` → `blocks.0.attn.q_proj`
- 이 코드로는 LoRA가 적용되지 않음
- **0.3% 파라미터만 학습** 주장은 검증되지 않음

**필수 수정:**
```python
# 올바른 target_modules
target_modules = [
    r"blocks\.\d+\.attn\.q_proj",
    r"blocks\.\d+\.attn\.k_proj",
    r"blocks\.\d+\.attn\.v_proj",
    r"blocks\.\d+\.attn\.out_proj",
    "lm_head"
]
```

---

### 4. **Vocabulary Size 불일치** ❌

**문제:**
```python
# midi_tokenizer.py에서 vocab_size 계산
vocab = []
vocab.append("<PAD>")      # 1
vocab.append("<BOS>")      # 2
vocab.append("<EOS>")      # 3
vocab.append("<UNK>")      # 4
vocab.append("Bar")        # 5
# Position: 16개
# Note_On: 88개 (21-108)
# Note_Off: 88개
# Velocity: 32개
# Time_Shift: 100개
# Tempo: 60개
# Chord: 96개 (12 roots × 8 types)

# 총 = 4 + 1 + 16 + 88 + 88 + 32 + 100 + 60 + 96 = 485

# 그런데 모델에서는:
model = JazzFlowRT(
    midi_vocab_size=512,  # ← 485가 아니라 512?
    ...
)
```

**심사평:**
- Vocabulary size가 일관되지 않음
- Tokenizer가 만드는 vocab과 모델 입력이 불일치
- IndexError 발생 가능성 높음

**필수 수정:**
1. Tokenizer.vocab_size를 자동으로 모델에 전달
2. 또는 명확하게 문서화

---

### 5. **실시간성 검증 없음** ❌

**주장:**
> "50ms 이하 지연시간 (기존 대비 10배 빠름)"

**코드 확인:**
```python
# inference/live_jam.py:208
generation_time = (time.time() - bar_start_time) * 1000  # ms
real_time_factor = self.bar_duration / (generation_time / 1000.0)
```

**심사평:**
- 벤치마크 코드는 있으나 **실제 측정 결과 없음**
- KV-cache 사용하지만 효과 검증 안 됨
- "50ms" 주장에 대한 증거 전무
- GPU vs CPU 비교 없음
- Batch size 영향 분석 없음

**필수 수정:**
1. 10회 이상 반복 측정 (평균 ± 표준편차)
2. 다양한 하드웨어에서 벤치마크 (CPU, GPU, different models)
3. Ablation: KV-cache 유무 비교
4. 다른 모델들과 동일 조건에서 비교

---

## 🟡 Major Issues (강력 권장)

### 6. **Jazz Metrics가 너무 단순** 🟡

**코드:**
```python
# evaluation/metrics.py:88-91
pitch_class = (note['pitch'] - root) % 12
if pitch_class in scale:
    correct_duration += note_duration
```

**심사평:**
- Harmonic consistency 계산이 과도하게 단순화
- 재즈의 텐션 노트 (9th, 11th, 13th) 고려 안 함
- 코드 외음(outside notes)도 재즈에서는 정상인데 패널티
- Swing ratio 계산이 부정확 (IOI ratio만으로는 부족)

**권장 수정:**
1. 텐션 노트에 가중치 부여
2. 재즈 이론에 기반한 "acceptable outside notes" 정의
3. Groove detection 추가 (not just swing ratio)
4. 전문가 평가 (human evaluation) 추가

---

### 7. **Dataset Augmentation 미흡** 🟡

**코드:**
```python
# data_processing/dataset.py:174
if self.augment:
    transpose = np.random.randint(-6, 7)
    tokens = self._transpose_tokens(tokens, transpose)
```

**심사평:**
- Transpose만 있고 다른 augmentation 없음
- Time stretch, velocity scaling 누락
- 재즈 특화 augmentation 없음 (swing variation 등)

**권장 수정:**
1. Time stretch (0.9x ~ 1.1x)
2. Velocity scaling
3. Swing ratio variation
4. Chord substitution (재즈 리하모니제이션)

---

### 8. **Pre-training 설정 미검증** 🟡

**코드:**
```python
# training/pretrain.py:383-388
def lr_lambda(step):
    if step < args.warmup_steps:
        return step / args.warmup_steps
    else:
        progress = (step - args.warmup_steps) / (args.epochs * len(train_loader) - args.warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
```

**심사평:**
- Warmup steps = 4000 (default)인데, 이게 적절한지 검증 안 됨
- Cosine annealing 주기가 전체 학습에 걸쳐있음 → 너무 길 수 있음
- Learning rate 범위 (1e-4) 검증 안 됨
- Batch size 영향 분석 없음

**권장 수정:**
1. LR range test (1e-6 ~ 1e-3)
2. Warmup steps 최적화 (grid search)
3. Different schedules 비교 (cosine vs linear vs exponential)

---

## 🟢 Minor Issues (개선 권장)

### 9. **코드 주석이 한글** 🟢

**문제:**
```python
# 재즈 코드 인코더 (ImprovNet 기반)
# 기능 화성 (T, SD, D) 인식
```

**심사평:**
- 국제 학회 제출 시 영어 주석 필요
- GitHub 공개 시 접근성 저하

---

### 10. **테스트 코드 없음** 🟢

**문제:**
```
└── tests/          # TODO
    ├── test_model.py
    ├── test_tokenizer.py
    └── test_dataset.py
```

**심사평:**
- Unit test 전무
- CI/CD 없음
- 재현성 보장 어려움

---

## 📝 논문 Draft 평가

### paper/draft.md 문제점:

1. **Abstract에 실험 결과 없음**
   ```markdown
   We achieve 87% style accuracy... ← 어디서 나온 숫자?
   ```

2. **Method 섹션이 너무 간략**
   - 수식이 하나도 없음
   - Architecture diagram 없음
   - Ablation study 계획 없음

3. **Related Work 누락**
   - MuseGAN, MusicVAE 등 중요 선행 연구 미언급
   - JazzGAN, BebopNet 비교 필요

4. **Experiments 섹션이 "예정"**
   ```markdown
   ### 4. Experiments (TODO)
   ```

---

## ✅ 수정 계획

### Phase 1: 필수 수정 (4주)
- [ ] MAESTRO 데이터셋 실제 학습 실행
- [ ] Baseline 3개 구현 및 비교
- [ ] Chord extraction 실제 구현
- [ ] QLoRA target_modules 수정
- [ ] Vocabulary size 통일
- [ ] 실시간 성능 벤치마크 (10회 측정)

### Phase 2: 강력 권장 (2주)
- [ ] Jazz metrics 개선 (텐션 노트 고려)
- [ ] Dataset augmentation 추가
- [ ] Learning rate 최적화
- [ ] Human evaluation (최소 10명)

### Phase 3: 개선 사항 (1주)
- [ ] 영어 주석으로 전환
- [ ] Unit tests 작성
- [ ] Documentation 개선
- [ ] CI/CD 설정

---

## 🎯 수정 후 재심사 기준

**Accept 조건:**
1. ✅ 실제 실험 결과 (최소 3개 baseline 비교)
2. ✅ Chord extraction 작동
3. ✅ QLoRA 실제 작동 검증
4. ✅ 50ms 지연시간 실제 측정
5. ✅ Human evaluation (p<0.05)
6. ✅ Ablation study (각 컴포넌트의 기여도)
7. ✅ 재현 가능 (README 따라 했을 때 결과 재현)

**현재 상태로는: REJECT**
**수정 후: Major Revision → Accept 가능**

---

## 💡 긍정적인 면

1. ✅ 코드 구조는 잘 짜여있음
2. ✅ 아이디어 자체는 참신함 (3개 모델 결합)
3. ✅ Documentation이 상세함
4. ✅ 재현 가능성을 고려한 설계
5. ✅ 실용적인 응용 (live jam session)

**결론: 아이디어는 좋으나, 실험 검증이 필수**

---

**Recommendation: MAJOR REVISION REQUIRED**

**심사위원 서명**: Prof. [MIR & AI]
**날짜**: 2025-11-17
