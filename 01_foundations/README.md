# Phase 1: 기초 다지기 (Month 1-2)

> Python, PyTorch, 음악 이론 완전 정복

## 🎯 학습 목표

이 Phase에서는 음악 생성 AI를 만들기 위한 **필수 기초**를 다집니다.

### 학습 내용
1. **Python 고급 문법** - Generator, Decorator, Type Hints
2. **PyTorch 기본** - Tensor, Autograd, nn.Module
3. **음악 이론** - MIDI, 음계, 코드, 리듬
4. **첫 음악 생성 모델** - 간단한 RNN으로 멜로디 생성

### 예상 소요 시간
- **초보자**: 8주 (하루 2-3시간)
- **중급자**: 4주 (하루 1-2시간)
- **고급자**: 2주 (개념 복습)

---

## 📂 디렉토리 구조

```
01_foundations/
├── README.md                          # 이 파일
│
├── 01_python_basics/                  # Week 1-2
│   ├── README.md
│   ├── generators.py                  # Generator 패턴
│   ├── decorators.py                  # Decorator 패턴
│   ├── type_hints.py                  # Type Hints
│   ├── context_managers.py            # Context Manager
│   └── exercises.py                   # 연습 문제
│
├── 02_pytorch_basics/                 # Week 2-4
│   ├── README.md
│   ├── 01_tensors.py                  # Tensor 기초
│   ├── 02_autograd.py                 # 자동 미분
│   ├── 03_nn_module.py                # nn.Module
│   ├── 04_training_loop.py            # 학습 루프
│   ├── 05_data_loading.py             # DataLoader
│   └── pytorch_tutorial.ipynb         # 종합 튜토리얼
│
├── 03_music_theory/                   # Week 4-6
│   ├── README.md
│   ├── 01_midi_basics.py              # MIDI 기초
│   ├── 02_scales_chords.py            # 음계와 코드
│   ├── 03_rhythm.py                   # 리듬과 박자
│   ├── 04_music_analysis.py           # 음악 분석
│   └── music_theory.ipynb             # 실습 노트북
│
└── 04_simple_rnn/                     # Week 6-8
    ├── README.md
    ├── model.py                       # RNN 모델
    ├── dataset.py                     # MIDI 데이터셋
    ├── train.py                       # 학습 스크립트
    ├── generate.py                    # 음악 생성
    └── rnn_melody_generation.ipynb    # 데모 노트북
```

---

## 🚀 시작하기

### 1. Week 1-2: Python 고급 문법

```bash
cd 01_python_basics
python generators.py        # Generator 학습
python decorators.py        # Decorator 학습
python type_hints.py        # Type Hints 학습
python exercises.py         # 연습 문제
```

**학습 포인트**:
- Generator를 사용한 대용량 데이터 처리
- Decorator를 활용한 코드 재사용
- Type Hints로 코드 안정성 향상

### 2. Week 2-4: PyTorch 기본

```bash
cd 02_pytorch_basics

# 순서대로 학습
python 01_tensors.py        # Tensor 연산
python 02_autograd.py       # 자동 미분
python 03_nn_module.py      # 모델 구축
python 04_training_loop.py  # 학습 루프
python 05_data_loading.py   # 데이터 로딩

# 종합 튜토리얼
jupyter notebook pytorch_tutorial.ipynb
```

**학습 포인트**:
- Tensor 연산 마스터
- Autograd 이해 (backpropagation)
- nn.Module 상속해서 모델 만들기
- 전체 학습 파이프라인 구축

### 3. Week 4-6: 음악 이론

```bash
cd 03_music_theory

python 01_midi_basics.py     # MIDI 파일 다루기
python 02_scales_chords.py   # 음계, 코드 이론
python 03_rhythm.py          # 리듬 이론
python 04_music_analysis.py  # 음악 분석

# 실습
jupyter notebook music_theory.ipynb
```

**학습 포인트**:
- MIDI 포맷 완전 이해
- 음악 이론 (음계, 코드, 리듬)
- Python으로 음악 데이터 처리

### 4. Week 6-8: 첫 음악 생성 모델 (RNN)

```bash
cd 04_simple_rnn

# 데이터 준비 (간단한 MIDI 파일들)
# MAESTRO 일부 또는 직접 만든 MIDI 사용

# 학습
python train.py --data_dir ./data --epochs 50

# 생성
python generate.py --checkpoint ./checkpoints/best.pt --length 100

# 데모 노트북
jupyter notebook rnn_melody_generation.ipynb
```

**결과**:
✅ 간단한 멜로디 생성 RNN 완성
✅ 첫 음악 AI 모델 학습 경험
✅ 다음 Phase (Transformer) 준비 완료

---

## 📚 각 섹션별 상세 설명

### 01_python_basics

**필수 개념**:
- Generator: 대용량 MIDI 데이터셋 효율적 처리
- Decorator: 학습 시간 측정, 로깅 자동화
- Type Hints: 코드 안정성 (대규모 프로젝트 필수)

**실습 예제**:
```python
# Generator로 대용량 MIDI 파일 처리
def midi_generator(midi_dir):
    for midi_file in Path(midi_dir).glob("*.mid"):
        yield load_midi(midi_file)

# Decorator로 함수 실행 시간 측정
@timer
def train_model():
    ...
```

### 02_pytorch_basics

**필수 개념**:
- Tensor: 모든 데이터의 기본 형태
- Autograd: 자동 미분 (딥러닝의 핵심)
- nn.Module: 모델 클래스 정의
- DataLoader: 효율적 데이터 로딩

**실습 예제**:
```python
# 간단한 MLP 모델
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 학습
model = SimpleMLP(128, 256, 128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
        loss.backward()
        optimizer.step()
```

### 03_music_theory

**필수 개념**:
- MIDI 포맷: Note on/off, velocity, tempo
- 음계: Major, Minor, Pentatonic
- 코드: Triad, 7th chords
- 리듬: Time signature, quantization

**실습 예제**:
```python
import pretty_midi

# MIDI 파일 읽기
midi = pretty_midi.PrettyMIDI('song.mid')

# 노트 추출
for instrument in midi.instruments:
    for note in instrument.notes:
        print(f"Pitch: {note.pitch}, Start: {note.start}, End: {note.end}")

# C Major 스케일 생성
c_major = [60, 62, 64, 65, 67, 69, 71, 72]  # C, D, E, F, G, A, B, C
```

### 04_simple_rnn

**목표**: 간단한 멜로디를 생성하는 RNN 모델

**모델 구조**:
```
Input (pitch) → Embedding → LSTM (2 layers) → FC → Output (next pitch)
```

**학습 데이터**:
- 간단한 MIDI 멜로디 (동요, 민요 등)
- Sequence length: 32-64
- Vocabulary: 128 (MIDI pitches)

**예상 결과**:
- 10-20 에포크 후 그럴듯한 멜로디 생성
- Loss: ~2.0 이하
- 음계 규칙을 어느 정도 따름

---

## ✅ Phase 1 완료 체크리스트

완료했으면 체크하세요:

### Week 1-2: Python
- [ ] Generator 패턴 이해 및 실습
- [ ] Decorator 패턴 이해 및 실습
- [ ] Type Hints 활용
- [ ] 연습 문제 모두 해결

### Week 2-4: PyTorch
- [ ] Tensor 연산 마스터
- [ ] Autograd 완전 이해
- [ ] nn.Module로 모델 구축
- [ ] 전체 학습 루프 작성
- [ ] DataLoader로 데이터 로딩

### Week 4-6: 음악 이론
- [ ] MIDI 파일 읽기/쓰기
- [ ] 음계와 코드 이해
- [ ] 리듬과 박자 이해
- [ ] 간단한 음악 분석 코드 작성

### Week 6-8: RNN 음악 생성
- [ ] RNN 모델 구현
- [ ] MIDI 데이터셋 구축
- [ ] 모델 학습 (Loss < 2.0)
- [ ] 멜로디 생성 성공
- [ ] 생성된 MIDI 파일 재생

**모두 체크했다면 Phase 2 (Transformer)로 진행하세요!**

---

## 🔥 학습 팁

### 1. 손으로 직접 코딩하기
- 복사-붙여넣기 X
- 직접 타이핑하면서 이해

### 2. 작은 실험 반복
```python
# 예시: MIDI 노트 하나씩 변경해보기
for pitch in range(60, 73):  # C major scale
    note = pretty_midi.Note(velocity=100, pitch=pitch, start=0, end=0.5)
    # 들어보기
```

### 3. 질문하고 디버깅
- `print()` 적극 활용
- `breakpoint()` 로 중간 확인
- Stack Overflow, PyTorch Forum 활용

### 4. 매일 조금씩
- 하루 30분이라도 매일
- 한 번에 몰아서 X

---

## 📖 추가 자료

### Python
- [Real Python](https://realpython.com/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

### PyTorch
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)

### 음악 이론
- [Music Theory for Computer Musicians](https://www.amazon.com/Music-Theory-Computer-Musicians-Michael/dp/1598635034)
- [Pretty MIDI Documentation](https://craffel.github.io/pretty-midi/)

### RNN
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 🎓 다음 단계

Phase 1을 완료하면 **Phase 2 (Transformer)** 로 진행합니다:
- `02_transformer/` 디렉토리로 이동
- Music Transformer (2018) 구현
- Attention 메커니즘 마스터

**준비되었나요? 시작하세요!** 🚀

```bash
cd 01_python_basics
python generators.py
```
