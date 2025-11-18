# ImprovNet 논문 상세 분석

**논문**: ImprovNet - Generating Controllable Musical Improvisations with Iterative Corruption Refinement
**저자**: Keshav Bhandari et al. (Queen Mary University of London, SUTD Singapore)
**출판**: arXiv:2502.04522v4 (2025년 5월 16일)
**코드**: https://github.com/keshavbhandari/improvnet

---

## 📋 Executive Summary

ImprovNet은 **Corruption-Refinement 전략**을 사용하여 표현력 있고 제어 가능한 음악 즉흥연주를 생성하는 Transformer 기반 모델입니다. 단일 모델로 5가지 작업(Cross-genre/Intra-genre improvisation, Harmonization, Continuation, Infilling)을 통합한 최초의 시스템입니다.

### 핵심 성과
- **79%** 참가자가 재즈 스타일 식별 성공 (p=0.0037)
- **56%** 참가자가 Short Continuation에서 AMT보다 ImprovNet 선호
- **76%** 참가자가 재즈 harmonization 정확히 식별 (p=0.0133)

---

## 🎯 연구 목적 및 문제 정의

### 해결하려는 문제
1. **제한된 재즈 데이터셋**: 특히 expressive polyphonic solo piano jazz 부족
2. **단편화된 모델**: Style transfer, harmonization, infilling 등이 별도 모델로 존재
3. **제어 가능성 부족**: 사용자가 스타일 변환 강도를 조절할 수 없음
4. **표현력 부족**: 기존 모델들은 highly quantized encoding 사용

### 연구 질문
> "단일 모델로 표현력 있는 성능 수준의 음악 즉흥연주를 생성하면서, 사용자가 스타일 변환 강도와 구조적 유사성을 제어할 수 있는가?"

---

## 🏗️ 방법론

### 1. 핵심 아이디어: Corruption-Refinement

```
원본 악보 → 의도적으로 망가뜨림 (Corruption) → 다시 복원 (Refinement)
```

**Self-supervised Learning**:
- Ground truth는 원본 음악
- Corruption function으로 다양한 방식으로 음악을 변형
- 모델은 변형된 음악을 원본으로 복원하는 것을 학습

### 2. 9가지 Corruption Functions

| # | 함수 이름 | 설명 | 목적 |
|---|----------|------|------|
| 1 | **Pitch Velocity Mask** | 음높이/세기 마스킹 | 멜로디/다이나믹스 재생성 학습 |
| 2 | **Onset Duration Mask** | 시작시간/길이 마스킹 | 싱코페이션 학습 (재즈) |
| 3 | **Whole Mask** | 전체 세그먼트 마스킹 | Continuation/Infilling |
| 4 | **Permute Pitch** | 음높이만 섞기 | 음높이 관계 학습 |
| 5 | **Permute Pitch Velocity** | 음높이+세기 섞기 | 전체 재구성 |
| 6 | **Fragmentation** | 20-50%만 유지 | 변주 생성 |
| 7 | **Incorrect Transposition** | ±5 semitones 틀린 이조 | 재즈 하모니/크로매틱 스케일 |
| 8 | **Note Modification** | 10-40% 음표 추가/제거 | 변주 및 밀도 조절 |
| 9 | **Skyline** | 멜로디만 추출 | Harmonization |

**Example - Incorrect Transposition**:
```
Original:  C  E  G  (C major chord)
Corrupt:   C# F  G# (Random ±5 semitone shift)
Refine:    C  E  G  (Or jazz version: C E G B♭)
```

### 3. Tokenization: Aria

```
┌─────────────────────────────────────┐
│ Chunked Absolute Onset Encoding     │
├─────────────────────────────────────┤
│ • 5초 세그먼트로 분할               │
│ • Onset: 10ms 단위 (절대 시간)     │
│ • Duration: 10ms 단위               │
│ • Velocity: 15 MIDI units 단위     │
│ • Pitch+Velocity: 단일 토큰 병합   │
│ • Sustain pedal 정보 포함          │
│ • Minimal quantization              │
└─────────────────────────────────────┘
```

**장점**:
- Human-like performance 표현 가능
- Timing, dynamics, articulation 세밀 제어

**단점**:
- Vocabulary size가 곡 길이에 비례
- 해결책: `<T>` 토큰으로 5초마다 onset reset

### 4. 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Encoder (12L, 8H, 512D)              │
│  Input: Left Context + Corrupted Segment + Right Context│
│         + Genre Token + Corruption Type Token           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    Decoder (12L, 8H, 512D)              │
│  Output: Refined Segment (Original)                     │
└─────────────────────────────────────────────────────────┘
```

**Context Window**:
- Training: L, R ∈ [1, 5] (randomly chosen)
- Generation: 항상 양쪽 컨텍스트 사용
- Total: 최대 11 segments (55초)

### 5. Iterative Generation Framework

**수식**:
```
For q = 1, ..., Q (passes):
    For i = 1, ..., N (segments):
        if (Pq < α) and Wi = 1:
            si^(q) = rθ(si^(q-1))  # Refine
        else:
            si^(q) = si^(q-1)      # Keep
```

**User Control Parameters**:
- `Q`: Number of passes (1-10)
- `α`: Corruption rate (0.0-1.0)
- `fj`: Corruption function (1-9)
- `Ct`: Target genre token
- `L, R`: Context window size
- `preservation_ratio`: Novel segment 보존 비율

**Example - Classical to Jazz (3 passes)**:
```
Pass 1: α=1.0, fj=whole_mask     → 80% jazz
Pass 2: α=0.5, fj=onset_duration → 90% jazz
Pass 3: α=0.3, fj=fragmentation  → 95% jazz
```

---

## 📊 실험 설정

### 데이터셋

| 데이터셋 | 용도 | 시간 | 설명 |
|---------|------|------|------|
| **ATEPP** | Pre-training | ~1000h | 클래식 피아노 (retranscribed) |
| **Maestro** | Fine-tuning | 177h | 클래식 피아노 (expressive) |
| **PiJAMA** | Fine-tuning | 200h+ | 재즈 피아노 (retranscribed) |
| **Doug McKenzie** | Fine-tuning | 307 pieces | 재즈 MIDI (short) |
| **Wikifonia** | Evaluation | 100 pieces | Harmonization 평가 |

**Retranscription**:
- State-of-the-art transcription model 사용
- MIDI 정확도 향상 (특히 PiJAMA)

### 학습 하이퍼파라미터

**Pre-training**:
- Steps: 360K
- Batch size: 4
- Learning rate: 1×10⁻⁴
- Weight decay: 0.01
- Warm-up ratio: 0.3

**Fine-tuning**:
- Steps: 318K
- Batch size: 4
- Gradient accumulation: 3
- Learning rate: 5×10⁻⁵
- Weight decay: 0.01
- Warm-up ratio: 0.1

### Genre Classifier (보조 모델)

**목적**: Cross-genre improvisation 평가

**아키텍처**:
- Encoder와 동일 (12L, 8H, 512D)
- Max seq length: 1024 tokens
- Binary cross-entropy loss

**학습**:
- Steps: 16K
- Batch size: 64
- Dropout: 0.1
- Input: 3 consecutive segments (15초)

---

## 📈 평가 결과

### 1. Cross-Genre Improvisation (CGI)

**Objective Metrics**:

| Corruption Function | Jazz Probability (10 passes) | SSM Correlation (10 passes) |
|---------------------|------------------------------|------------------------------|
| Whole Mask | **0.85** | **0.25** |
| Onset Duration Mask | 0.80 | 0.35 |
| Fragmentation | 0.65 | 0.55 |
| Skyline | 0.60 | **0.65** |
| Note Modification | 0.58 | 0.62 |

**해석**:
- **Whole Mask**: 가장 강한 스타일 변환, 구조 변화 큼
- **Skyline**: 구조 유지하면서 스타일 변환
- **Note Modification**: 점진적 변주

**Corruption Rate 영향**:
```
α=0.25: Jazz Prob=0.35, SSM=0.75
α=0.50: Jazz Prob=0.50, SSM=0.55
α=0.75: Jazz Prob=0.65, SSM=0.40
α=1.00: Jazz Prob=0.80, SSM=0.30
```

**Subjective Results**:

| Model | Interest | Human-like | Overall | Structural Sim. | Genre ID |
|-------|----------|-----------|---------|-----------------|----------|
| **ImprovNet CGI** | 3.36/5 | 2.71/5 | 3.11/5 | 3.21/5 | **79%** |
| **ImprovNet IGI** | 3.39/5 | **3.25/5** | 3.21/5 | **4.07/5** | - |
| **Original** | 3.43/5 | 5.00/5 | 3.64/5 | - | - |

**통계적 유의성**:
- Genre identification: p=0.0037 (α=0.05)
- One-sample binomial test

### 2. Short Prompt Continuation

**Setup**:
- 20초 프롬프트 → 10초 생성
- 비교 대상: AMT (Anticipatory Music Transformer)

| Model | Avg. IOI | Note Density | Unique Pitches | PCTM Cosine Sim↑ | Pitch Class KL↓ |
|-------|----------|--------------|----------------|-------------------|------------------|
| **ImprovNet** | 0.1244 | 37.49 | **28.53** | **0.3470** | **1.2500** |
| AMT | 0.1386 | 67.66 | 25.15 | 0.3074 | 1.6084 |
| Original | **0.1405** | **30.85** | 27.07 | - | - |

**해석**:
- **ImprovNet이 원본에 더 가까움**
- AMT는 Note Density가 과도하게 높음 (67.66 vs 30.85)
- ImprovNet은 Pitch diversity도 원본과 유사

**Subjective**: 56% 참가자가 ImprovNet 선호 (vs 20% AMT)

### 3. Short Infilling

**Setup**:
- 0:20 + 30:50 seconds 제공
- 20:30 seconds 생성

| Model | Avg. IOI | Note Density | Unique Pitches | PCTM Cosine Sim↑ | Pitch Class KL↓ |
|-------|----------|--------------|----------------|-------------------|------------------|
| **ImprovNet** | 0.1224 | 36.30 | **28.78** | **0.4036** | **0.8907** |
| AMT | 0.1508 | 49.63 | 25.45 | 0.3624 | 1.4593 |
| Original | **0.1394** | **32.44** | 27.88 | - | - |

**해석**:
- **ImprovNet이 더 우수** (모든 metric에서 원본에 가까움)
- Corruption functions의 data augmentation 효과

### 4. Harmonization

**Setup**:
- Wikifonia 데이터셋 (100 pieces)
- Monophonic melody → Polyphonic harmonization
- Logit constraints 사용

| Model | Poly. Rate | Note Density | Tonal Tension | Chord Diversity | Pitch in Scale |
|-------|-----------|--------------|---------------|-----------------|----------------|
| **ImprovNet w/ constraints** | **0.91** | 35.95 | 0.62 | **18.30** | 80.17 |
| ImprovNet w/o constraints | 0.25 | 11.06 | 0.75 | 13.62 | 80.26 |
| Random Chords | - | - | **0.91** | 37.38 | 63.64 |
| Monophonic | 0.00 | 8.71 | - | - | - |
| Original | **0.96** | **19.20** | **0.46** | 13.29 | **79.40** |

**해석**:
- **Constraints 필수**: w/o constraints는 harmonization 실패 (0.25)
- Chord Diversity 높음: 재즈 harmonization의 특징
- Tonal Tension 약간 높음: 재즈의 dissonance 반영

**Subjective Results**:

| Model | Interest | Match | Overall | Genre ID |
|-------|----------|-------|---------|----------|
| **ImprovNet CGH** (cross-genre) | **3.54/5** | 2.65/5 | 2.92/5 | **76%** |
| **ImprovNet IGH** (intra-genre) | 2.58/5 | **3.27/5** | 2.54/5 | - |
| Original | 3.38/5 | **3.89/5** | **3.31/5** | - |

**해석**:
- Cross-genre가 **더 흥미로움** (3.54 vs 2.58)
- Intra-genre가 melody와 **더 잘 맞음** (3.27 vs 2.65)
- Genre identification: p=0.0133 (통계적 유의)

---

## 🎛️ 제어 메커니즘

### 1. Genre Conditioning

```python
# 학습 시
genre_token = original_genre  # <CLASSICAL> or <JAZZ>

# 생성 시
target_genre = user_selected_genre  # <JAZZ>
```

### 2. Corruption Function Selection

**추천 조합**:

| 목표 | Corruption Functions | Passes | α |
|------|---------------------|--------|---|
| **Strong Jazz Conversion** | whole_mask, onset_duration_mask | 5 | 1.0, 0.7, 0.5, 0.3, 0.1 |
| **Subtle Variation** | note_modification, fragmentation | 3 | 0.3, 0.2, 0.1 |
| **Melodic Improvisation** | permute_pitch, incorrect_transposition | 4 | 0.5, 0.4, 0.3, 0.2 |
| **Rhythmic Change** | onset_duration_mask | 2 | 0.8, 0.5 |

### 3. Context Window

**Classical → Jazz (CGI)**:
- Smaller R (1-2): 더 빠른 변환
- Larger R (4-5): 더 부드러운 변환

**Classical → Classical (IGI)**:
- Larger L, R (3-5): 더 일관된 스타일 유지

### 4. Preservation Ratio

```python
preservation_ratio = 0.05  # 5% of novel segments preserved

# Self-similarity & Novelty algorithm
# → 새로운 섹션 시작점, 주요 모티프 유지
```

### 5. Logit Constraints (Harmonization)

```python
# First 3 onset tokens in each segment
for note in first_3_notes:
    if abs(onset - first_note_onset) > 50ms:
        logits[onset_token] = -∞  # 불가능하게 만듦
```

**효과**:
- 첫 화음 강제 생성
- 이후 음표들이 자연스럽게 harmonization 따름

---

## 💡 핵심 통찰

### 1. Corruption-Refinement의 힘

**왜 효과적인가?**:
1. **Data Augmentation**: 9가지 corruption = 9배 데이터
2. **Robust Representation**: 다양한 변형에 대응 학습
3. **Self-supervised**: 레이블 필요 없음
4. **User Control**: Corruption type, rate로 강도 조절

**비유**:
```
Corruption-Refinement ≈ Denoising Diffusion
But with explicit, controllable "noise" types
```

### 2. Iterative Generation의 장점

**Single Pass vs Multiple Passes**:
```
Single Pass (α=1.0):
  Classical → Jazz (60% similarity)

Multiple Passes (α=1.0, 0.5, 0.3):
  Classical → Transitional → Almost Jazz → Pure Jazz
  (점진적 변환, 더 자연스러움)
```

### 3. Context Window의 중요성

**Optimal Settings (실험 결과)**:
- **CGI**: L=3-5, R=1-2 (빠른 변환)
- **IGI**: L=3-5, R=3-5 (일관성 유지)
- **Continuation**: L=3-5, R=0 (우측 컨텍스트 없음)
- **Infilling**: L=3-5, R=3-5 (양쪽 컨텍스트 필수)

### 4. Corruption Function 선택

**Genre Conversion Power**:
```
Strongest:  Whole Mask > Onset Duration > Fragmentation
Moderate:   Note Modification > Skyline
Weakest:    Permute > Incorrect Transposition
```

**Structure Preservation**:
```
Best:   Skyline > Note Modification
Good:   Fragmentation
Poor:   Onset Duration > Whole Mask
```

### 5. Harmonization의 Trick

**Logit Constraints**:
- Without: 25% polyphony (harmonization 실패)
- With: 91% polyphony (harmonization 성공)

**이유**:
- Monophonic context → 모델은 harmony 생성 압력 없음
- First chord 강제 → 나머지 자동으로 harmonization 학습 따름

---

## 🚧 한계점

### 1. Harmonization Issues

**문제**: Dense chords (35.95 note density vs 19.20 original)

**원인**:
- Wikifonia 데이터셋으로 학습 안 됨
- Jazz harmony는 복잡한 voicing 선호

**해결책** (저자 제안):
- Additional conditional tokens for chord density

### 2. Irregular Rhythms

**문제**: Onset-duration mask가 가끔 불규칙한 리듬 생성

**해결책** (저자 제안):
- Refine for consistent swing rhythm
- Tempo conditioning 추가

### 3. Phrasal Structure Distortion

**문제**: CGI에서 가끔 phrase structure 왜곡 (Human-like: 2.71/5)

**원인**:
- Strong corruption + multiple passes
- Insufficient long-term structure modeling

### 4. 10-Second Context Limit

**한계**:
- Song structure (verse, chorus) 직접 모델링 불가
- Melody, rhythm, chord progression만 가능

**해결책** (향후 연구):
- Hierarchical structure modeling
- Longer context window (Transformer-XL, Memorizing Transformer)

---

## 🔬 실험적 발견

### 1. Corruption Rate Curve

```
α=0.25: Minimal change (거의 원본 유지)
α=0.50: Moderate variation (변주 수준)
α=0.75: Strong change (스타일 변환 시작)
α=1.00: Maximum change (완전한 스타일 변환)
```

### 2. Number of Passes

```
1 Pass:  빠르지만 덜 자연스러움
3 Passes: 최적 (대부분 경우)
5 Passes: 매우 자연스러움 (시간 5배)
10 Passes: 과도 (수렴, 큰 변화 없음)
```

### 3. Corruption Function Combinations

**Best Combinations (Classical → Jazz)**:
```python
Pass 1: whole_mask, α=1.0         # 큰 변화
Pass 2: onset_duration_mask, α=0.7  # 재즈 리듬
Pass 3: incorrect_transposition, α=0.5  # 재즈 하모니
Pass 4: note_modification, α=0.3  # 세밀한 조정
```

### 4. Context Window Asymmetry

**CGI에서 발견**:
- Small R (1-2): 빠른 스타일 변환
- Large L (4-5): 부드러운 전환

**이유**:
- 우측 컨텍스트(미래)가 스타일 "고정" 역할
- 좌측 컨텍스트(과거)는 일관성 유지

---

## 📚 Related Work 비교

### vs GAN-based Models (CycleGAN, StyleGAN)

| 측면 | ImprovNet | GAN-based |
|-----|-----------|-----------|
| **Training Stability** | ✅ Stable | ❌ Unstable |
| **Mode Collapse** | ✅ No | ❌ Yes |
| **User Control** | ✅ Fine-grained | ❌ Limited |
| **Expressive Performance** | ✅ Yes (Aria tokenizer) | ❌ No (piano-roll) |

### vs VAE-based Models

| 측면 | ImprovNet | VAE-based |
|-----|-----------|-----------|
| **Disentanglement** | ✅ Explicit (corruption) | ⚠️ Implicit (latent) |
| **Control Interpretability** | ✅ High | ❌ Low |
| **Reconstruction Quality** | ✅ High | ⚠️ Medium (blurry) |

### vs AMT (Anticipatory Music Transformer)

| Task | Metric | ImprovNet | AMT |
|------|--------|-----------|-----|
| **Continuation** | PCTM Cosine Sim | **0.3470** | 0.3074 |
| | Pitch Class KL | **1.2500** | 1.6084 |
| **Infilling** | PCTM Cosine Sim | **0.4036** | 0.3624 |
| | Pitch Class KL | **0.8907** | 1.4593 |
| **User Preference** | Continuation | **56%** | 20% |

---

## 🎓 이론적 기여

### 1. Corruption-Refinement as General Framework

**Generalization**:
```
ImprovNet (Music)
  ↓
Text Editing (Corruption = typos, grammar errors)
  ↓
Image Restoration (Corruption = noise, blur)
  ↓
Video Frame Interpolation (Corruption = missing frames)
```

### 2. Link to Diffusion Models

**저자의 주장** (Section VII-A):
```
Single Pass ≈ Masked Diffusion (Stable Diffusion Inpainting)
Multiple Passes ≈ Iterative Denoising (but with controllable noise)
```

**차이점**:
- Diffusion: 랜덤 노이즈, 많은 step (50-1000)
- ImprovNet: 의도된 corruption, 적은 pass (1-10)

### 3. Self-supervised Learning for Music

**기여**:
- Weak labels (genre) 만으로 학습 가능
- Rich annotations (chord, beat, key) 불필요
- Scalable to large unlabeled datasets

---

## 💻 구현 세부사항

### Model Size

```
Encoder: 12L × 8H × 512D × 2048FFN = ~100M params
Decoder: 12L × 8H × 512D × 2048FFN = ~100M params
Total: ~200M parameters
```

### Inference Time

**Hardware**: Not specified (likely GPU)

**Estimated**:
```
1 segment (5s) generation:
  - Context encoding: ~50ms
  - Token generation: ~200ms (autoregressive)
  - Total: ~250ms

1 pass (100 segments, 500s):
  - Total: ~25s
  - RTF: 500/25 = 20× (very fast)

10 passes:
  - Total: ~250s
  - RTF: 500/250 = 2× (still real-time capable)
```

### Memory Requirements

```
Context: 11 segments × 2048 tokens = 22,528 tokens
Model: ~200M params × 4 bytes = 800MB
KV Cache: ~100MB
Total: ~1GB VRAM (reasonable)
```

---

## 🌟 실무 활용 가이드

### Use Case 1: Cross-Genre Remix

```python
# Classical piece → Jazz version
corruption_functions = [
    ("whole_mask", 1.0),
    ("onset_duration_mask", 0.7),
    ("incorrect_transposition", 0.5),
    ("note_modification", 0.3)
]

for func, alpha in corruption_functions:
    output = improvnet.refine(
        music=current_music,
        corruption_fn=func,
        corruption_rate=alpha,
        target_genre="jazz",
        context_L=4,
        context_R=2
    )
    current_music = output
```

### Use Case 2: Live Performance Variation

```python
# Real-time variation during live performance
while performing:
    # Generate variation every 5 seconds
    variation = improvnet.refine(
        music=last_5_seconds,
        corruption_fn="note_modification",
        corruption_rate=0.2,  # Subtle
        target_genre="same",  # Intra-genre
        context_L=5,
        context_R=5
    )
    play(variation)
```

### Use Case 3: Auto-Harmonization

```python
# Harmonize monophonic melody
harmonized = improvnet.harmonize(
    melody=monophonic_midi,
    target_genre="jazz",  # or "classical"
    use_logit_constraints=True,
    num_chord_notes=3  # N in paper
)
```

### Use Case 4: Continuation for Composition

```python
# Continue a musical idea
continuation = improvnet.continue(
    prompt=20_seconds_composition,
    target_length=10,  # 10 more seconds
    corruption_fn="whole_mask",
    target_genre="same"
)
```

---

## 📖 Citation

```bibtex
@article{bhandari2025improvnet,
  title={ImprovNet: Generating Controllable Musical Improvisations with Iterative Corruption Refinement},
  author={Bhandari, Keshav and Chang, Sungkyun and Lu, Tongyu and Enus, Fareza R and Bradshaw, Louis B and Herremans, Dorien and Colton, Simon},
  journal={arXiv preprint arXiv:2502.04522},
  year={2025}
}
```

---

## 🔮 미래 연구 방향 (저자 제안)

1. **Longer Context**: Transformer-XL, Memorizing Transformer로 song structure 모델링
2. **Fine-grained Tempo Control**: Tempo conditioning for consistent swing
3. **Chord Density Control**: Additional tokens for harmonization density
4. **Multi-track**: Ensemble improvisation (piano + bass + drums)
5. **Interactive Performance**: Live corruption function selection during performance

---

## 결론

ImprovNet은 **Corruption-Refinement**이라는 새로운 패러다임으로 음악 즉흥연주 생성에서 SOTA를 달성했습니다.

**핵심 강점**:
1. ✅ **통합된 프레임워크**: 5가지 작업을 단일 모델로
2. ✅ **사용자 제어**: 9가지 corruption, passes, rate로 세밀한 조절
3. ✅ **표현력**: Aria tokenizer로 human-like performance
4. ✅ **검증된 성능**: 79% genre identification, 56% user preference

**한계**:
1. ❌ Dense harmonization
2. ❌ Irregular rhythms (가끔)
3. ❌ 10-second context limit

ImprovNet은 음악 AI 연구의 새로운 방향을 제시하며, Corruption-Refinement 접근법은 다른 도메인에도 적용 가능한 일반화된 프레임워크입니다.
