# 첫 음악 생성 모델: Simple RNN

> LSTM으로 간단한 멜로디 생성하기

## 🎯 목표

이 섹션에서는 **처음으로 음악을 생성하는 AI 모델**을 만듭니다!

### 학습 내용
- LSTM 기반 시퀀스 모델
- MIDI 데이터를 학습 데이터로 변환
- 음악 생성 (다음 음 예측)
- 모델 학습 및 평가

### 예상 결과
- ✅ 간단한 멜로디 생성 성공
- ✅ 음계를 어느 정도 따르는 음악
- ✅ Loss < 2.0

---

## 📁 파일 구조

```
04_simple_rnn/
├── README.md                  # 이 파일
├── model.py                   # LSTM 모델 정의
├── dataset.py                 # MIDI 데이터셋
├── train.py                   # 학습 스크립트
├── generate.py                # 음악 생성 스크립트
├── config.py                  # 설정
└── demo.ipynb                 # 데모 노트북
```

---

## 🚀 빠른 시작

### 1. 데이터 준비

```bash
# 간단한 MIDI 파일들을 data/ 디렉토리에 넣기
# 또는 MAESTRO 데이터셋 일부 사용
mkdir -p data/midi_files
```

### 2. 학습

```bash
python train.py \
    --data_dir ./data/midi_files \
    --epochs 50 \
    --batch_size 32 \
    --hidden_dim 256
```

### 3. 생성

```bash
python generate.py \
    --checkpoint ./checkpoints/best.pt \
    --length 100 \
    --temperature 1.0 \
    --output generated.mid
```

### 4. 재생

```bash
# MIDI 플레이어로 재생
# 또는 온라인 플레이어 사용
```

---

## 🧠 모델 구조

```
입력: MIDI pitch sequence (예: [60, 64, 67, 72, ...])
         ↓
    Embedding (128 → 256)
         ↓
    LSTM (256 units, 2 layers)
         ↓
    Linear (256 → 128)
         ↓
출력: Next pitch logits
```

### 하이퍼파라미터

- **Vocabulary Size**: 128 (MIDI pitches 0-127)
- **Embedding Dim**: 256
- **Hidden Dim**: 256
- **Num Layers**: 2
- **Dropout**: 0.2
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Sequence Length**: 32

---

## 📊 학습 과정

### Epoch 1-10: 초기 학습
- Loss: ~4.0 → ~2.5
- 랜덤한 음들

### Epoch 10-30: 패턴 학습
- Loss: ~2.5 → ~1.5
- 음계를 따르기 시작

### Epoch 30-50: 정제
- Loss: ~1.5 → ~1.0
- 그럴듯한 멜로디

---

## 💡 학습 팁

1. **데이터가 충분한지 확인**
   - 최소 100개 MIDI 파일 권장
   - 비슷한 스타일의 음악이 좋음

2. **Overfitting 주의**
   - Dropout 사용
   - Early stopping

3. **Temperature 조절**
   - 낮으면 (0.5): 안전하고 반복적
   - 높으면 (1.5): 창의적이지만 불안정

---

## 🎵 다음 단계

이 RNN 모델을 완료하면:
- ✅ 기본 음악 생성 이해
- ✅ Sequence-to-sequence 학습 경험
- ✅ Transformer 준비 완료

**다음: Phase 2 - Music Transformer**
