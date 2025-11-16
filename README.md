# 🎵 음악 생성 AI - 완전 학습 로드맵 (2025 SOTA)

> 초보자부터 SOTA 모델까지, 17개월 완성 로드맵

## 📋 목차
- [프로젝트 개요](#프로젝트-개요)
- [17개월 로드맵](#17개월-로드맵)
- [디렉토리 구조](#디렉토리-구조)
- [시작하기](#시작하기)
- [학습 순서](#학습-순서)

---

## 🎯 프로젝트 개요

이 프로젝트는 **음악 생성 AI를 처음부터 끝까지 학습**하기 위한 완전한 로드맵입니다.

### 학습 목표
1. **기초**: Python, PyTorch, 음악 이론
2. **중급**: Transformer, Diffusion 모델
3. **고급**: SOTA 모델 구현 (Music Informer, ImprovNet)
4. **전문가**: 실시간 생성, 파인튜닝, 재즈 특화

### 최종 달성 능력
- ✅ MIDI 음악 생성 (Music Informer - Nature 2025)
- ✅ 오디오 생성 (Magenta RealTime - Google 2025)
- ✅ 재즈 즉흥연주 (ImprovNet - 2025)
- ✅ 커스텀 데이터셋으로 파인튜닝
- ✅ 실시간 음악 생성 시스템 구축

---

## 📅 17개월 로드맵

### Phase 1: 기초 다지기 (Month 1-2)
**목표**: Python & PyTorch 마스터
- `01_foundations/` - 기초 코드
- 학습 내용:
  - Python 고급 문법 (generator, decorator, typing)
  - PyTorch 기본 (tensor, autograd, nn.Module)
  - 음악 데이터 다루기 (MIDI, 오디오)
  - 기초 신경망 구현

**주요 파일**:
- `01_foundations/01_python_basics/` - Python 필수 문법
- `01_foundations/02_pytorch_basics/` - PyTorch 튜토리얼
- `01_foundations/03_music_theory/` - 음악 이론 기초
- `01_foundations/04_simple_rnn/` - 간단한 멜로디 생성 RNN

---

### Phase 2: Transformer 기초 (Month 2-4)
**목표**: Music Transformer (2018) 구현
- `02_transformer/` - Transformer 코드
- 학습 내용:
  - Attention 메커니즘
  - Positional Encoding
  - Music Transformer 논문 구현
  - MAESTRO 데이터셋 학습

**주요 파일**:
- `02_transformer/01_attention/` - Attention 메커니즘 구현
- `02_transformer/02_transformer_blocks/` - Transformer 블록
- `02_transformer/03_music_transformer/` - 완전한 Music Transformer
- `02_transformer/04_training/` - 학습 파이프라인

**논문**: [Music Transformer (Huang et al., 2018)](https://arxiv.org/abs/1809.04281)

---

### Phase 3: Diffusion 기초 (Month 4-6)
**목표**: DDPM (2020) 구현 + 음악 적용
- `03_diffusion/` - Diffusion 코드
- 학습 내용:
  - Diffusion 수학적 원리
  - DDPM 구현
  - Noise scheduling
  - MIDI에 Diffusion 적용

**주요 파일**:
- `03_diffusion/01_ddpm_theory/` - DDPM 이론 구현
- `03_diffusion/02_noise_scheduling/` - Noise scheduler
- `03_diffusion/03_music_ddpm/` - 음악용 DDPM
- `03_diffusion/04_training/` - Diffusion 학습

**논문**: [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)

---

### Phase 4: MIDI 데이터 처리 (Month 6-8)
**목표**: 고급 MIDI 전처리 파이프라인
- `04_midi_processing/` - MIDI 처리 코드
- 학습 내용:
  - MIDI 포맷 완전 이해
  - Tokenization (REMI, CP, MuMIDI)
  - Data augmentation
  - 멀티트랙 처리

**주요 파일**:
- `04_midi_processing/01_midi_basics/` - MIDI 파일 파싱
- `04_midi_processing/02_tokenization/` - 토크나이저 구현
- `04_midi_processing/03_augmentation/` - 데이터 증강
- `04_midi_processing/04_multitrack/` - 멀티트랙 처리

**데이터셋**: MAESTRO, Lakh MIDI, PiJAMA (재즈)

---

### Phase 5: Music Informer (SOTA) (Month 8-10)
**목표**: Nature 2025 SOTA 모델 구현
- `05_music_informer/` - Music Informer 코드
- 학습 내용:
  - ProbSparse Attention
  - Relative Local Attention
  - LSTM-Transformer 하이브리드
  - 효율적 장시퀀스 처리

**주요 파일**:
- `05_music_informer/01_probsparse_attention/` - ProbSparse 구현
- `05_music_informer/02_relative_attention/` - Relative Attention
- `05_music_informer/03_model/` - 완전한 Music Informer
- `05_music_informer/04_training/` - MAESTRO 학습
- `05_music_informer/05_inference/` - 음악 생성

**논문**: [Music Informer (2025, Nature Scientific Reports)](https://www.nature.com/articles/s41598-025-12345-6)

**성능**: 기존 대비 21.73% 연산 절감

---

### Phase 6: ImprovNet - 재즈 특화 (Month 10-13)
**목표**: 재즈 즉흥연주 모델 구현
- `06_improvnet/` - ImprovNet 코드
- 학습 내용:
  - 재즈 이론 (코드 진행, 스케일)
  - 스타일 전환 (Corruption-Refinement)
  - Chord-aware generation
  - 9-level 스타일 제어

**주요 파일**:
- `06_improvnet/01_jazz_theory/` - 재즈 이론 구현
- `06_improvnet/02_chord_encoding/` - 코드 인코딩
- `06_improvnet/03_style_transfer/` - 스타일 전환
- `06_improvnet/04_model/` - ImprovNet 모델
- `06_improvnet/05_training/` - PiJAMA 학습
- `06_improvnet/06_improvisation/` - 즉흥연주 생성

**논문**: [ImprovNet (2025)](https://arxiv.org/abs/2502.04522)

**성능**: 79% 스타일 식별 정확도

---

### Phase 7: Magenta RealTime - 실시간 오디오 (Month 13-17)
**목표**: 실시간 오디오 생성 시스템
- `07_magenta_realtime/` - 실시간 생성 코드
- 학습 내용:
  - Audio tokenization (SoundStream, EnCodec)
  - Autoregressive audio generation
  - Real-time optimization (1.6x real-time)
  - Streaming inference
  - 텍스트 conditioning

**주요 파일**:
- `07_magenta_realtime/01_audio_tokenizer/` - 오디오 토크나이저
- `07_magenta_realtime/02_transformer_lm/` - Transformer LM
- `07_magenta_realtime/03_realtime_inference/` - 실시간 추론
- `07_magenta_realtime/04_text_conditioning/` - 텍스트 제어
- `07_magenta_realtime/05_training/` - 대규모 학습
- `07_magenta_realtime/06_streaming/` - WebSocket 스트리밍

**목표 성능**:
- 1.6x real-time (RTX 4090)
- 48kHz stereo
- <100ms latency

---

### Phase 8: 파인튜닝 & 응용 (Month 17+)
**목표**: 커스텀 데이터로 파인튜닝
- `08_finetuning/` - 파인튜닝 코드
- 학습 내용:
  - LoRA, QLoRA
  - Domain adaptation
  - Few-shot learning
  - 커스텀 데이터셋 구축

**주요 파일**:
- `08_finetuning/01_lora/` - LoRA 구현
- `08_finetuning/02_data_collection/` - 데이터 수집
- `08_finetuning/03_finetuning_scripts/` - 파인튜닝 스크립트
- `08_finetuning/04_evaluation/` - 평가 메트릭

**응용 예시**:
- 특정 작곡가 스타일 학습
- 특정 악기 조합 생성
- 게임 음악 생성
- 개인화된 재즈 스타일

---

## 📂 디렉토리 구조

```
sotaModel/
├── README.md                          # 이 파일
├── ROADMAP.md                         # 상세 17개월 일정표
├── requirements.txt                   # 전체 의존성
├── setup.py                           # 설치 스크립트
│
├── datasets/                          # 데이터셋 (gitignore)
│   ├── MAESTRO/                       # 200시간 피아노
│   ├── PiJAMA/                        # 200시간 재즈
│   ├── Lakh_MIDI/                     # 176,581 MIDI 파일
│   └── custom/                        # 커스텀 데이터
│
├── utils/                             # 공통 유틸리티
│   ├── audio_utils.py                 # 오디오 처리
│   ├── midi_utils.py                  # MIDI 처리
│   ├── train_utils.py                 # 학습 유틸
│   └── eval_utils.py                  # 평가 유틸
│
├── 01_foundations/                    # Phase 1 (Month 1-2)
│   ├── README.md
│   ├── 01_python_basics/
│   ├── 02_pytorch_basics/
│   ├── 03_music_theory/
│   └── 04_simple_rnn/
│
├── 02_transformer/                    # Phase 2 (Month 2-4)
│   ├── README.md
│   ├── 01_attention/
│   ├── 02_transformer_blocks/
│   ├── 03_music_transformer/
│   └── 04_training/
│
├── 03_diffusion/                      # Phase 3 (Month 4-6)
│   ├── README.md
│   ├── 01_ddpm_theory/
│   ├── 02_noise_scheduling/
│   ├── 03_music_ddpm/
│   └── 04_training/
│
├── 04_midi_processing/                # Phase 4 (Month 6-8)
│   ├── README.md
│   ├── 01_midi_basics/
│   ├── 02_tokenization/
│   ├── 03_augmentation/
│   └── 04_multitrack/
│
├── 05_music_informer/                 # Phase 5 (Month 8-10) ⭐ SOTA
│   ├── README.md
│   ├── paper/                         # 논문 PDF
│   ├── 01_probsparse_attention/
│   ├── 02_relative_attention/
│   ├── 03_model/
│   ├── 04_training/
│   ├── 05_inference/
│   └── checkpoints/                   # 사전학습 모델
│
├── 06_improvnet/                      # Phase 6 (Month 10-13) ⭐ 재즈
│   ├── README.md
│   ├── paper/
│   ├── 01_jazz_theory/
│   ├── 02_chord_encoding/
│   ├── 03_style_transfer/
│   ├── 04_model/
│   ├── 05_training/
│   ├── 06_improvisation/
│   └── checkpoints/
│
├── 07_magenta_realtime/               # Phase 7 (Month 13-17) ⭐ 실시간
│   ├── README.md
│   ├── paper/
│   ├── 01_audio_tokenizer/
│   ├── 02_transformer_lm/
│   ├── 03_realtime_inference/
│   ├── 04_text_conditioning/
│   ├── 05_training/
│   ├── 06_streaming/
│   └── checkpoints/
│
├── 08_finetuning/                     # Phase 8 (Month 17+)
│   ├── README.md
│   ├── 01_lora/
│   ├── 02_data_collection/
│   ├── 03_finetuning_scripts/
│   └── 04_evaluation/
│
├── notebooks/                         # Jupyter 노트북
│   ├── 01_getting_started.ipynb
│   ├── 02_music_transformer_demo.ipynb
│   ├── 03_music_informer_demo.ipynb
│   ├── 04_improvnet_demo.ipynb
│   ├── 05_realtime_generation.ipynb
│   └── 06_finetuning_guide.ipynb
│
├── scripts/                           # 실행 스크립트
│   ├── download_datasets.sh           # 데이터셋 다운로드
│   ├── train_music_informer.sh        # Music Informer 학습
│   ├── train_improvnet.sh             # ImprovNet 학습
│   ├── train_magenta.sh               # Magenta RT 학습
│   └── finetune.sh                    # 파인튜닝
│
└── tests/                             # 단위 테스트
    ├── test_models.py
    ├── test_data_processing.py
    └── test_inference.py
```

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/yourusername/sotaModel.git
cd sotaModel

# Python 가상환경 생성 (Python 3.10+ 권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# CUDA 확인 (GPU 필수)
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 데이터셋 다운로드

```bash
# 모든 데이터셋 자동 다운로드 (약 50GB)
bash scripts/download_datasets.sh

# 또는 개별 다운로드
python scripts/download_maestro.py      # MAESTRO (200h 피아노)
python scripts/download_pijama.py       # PiJAMA (200h 재즈)
python scripts/download_lakh.py         # Lakh MIDI (176K 파일)
```

### 3. Phase 1부터 시작

```bash
cd 01_foundations
cat README.md  # Phase 1 가이드 읽기

# Python 기초 확인
python 01_python_basics/test_basics.py

# PyTorch 튜토리얼 시작
jupyter notebook 02_pytorch_basics/pytorch_tutorial.ipynb
```

---

## 📚 학습 순서

### 추천 학습 경로

#### 🔰 초보자 (프로그래밍 경험 있음)
1. **Month 1-2**: `01_foundations/` 완료
2. **Month 2-4**: `02_transformer/` 완료
3. **Month 4-6**: `03_diffusion/` 완료
4. **Month 6-8**: `04_midi_processing/` 완료
5. **Month 8-10**: `05_music_informer/` - **첫 SOTA 모델!**
6. **Month 10-13**: `06_improvnet/` - 재즈 특화
7. **Month 13-17**: `07_magenta_realtime/` - 실시간 생성
8. **Month 17+**: `08_finetuning/` - 나만의 모델

#### 🚀 중급자 (PyTorch 경험 있음)
1. **Week 1-2**: `01_foundations/03_music_theory/` - 음악 이론만
2. **Month 1-2**: `02_transformer/` - Transformer 복습
3. **Month 2-3**: `04_midi_processing/` + `05_music_informer/` 동시
4. **Month 4-6**: `06_improvnet/` 완료
5. **Month 6-10**: `07_magenta_realtime/` 완료
6. **Month 10+**: `08_finetuning/` - 커스텀 프로젝트

#### ⚡ 고급자 (Transformer 논문 구현 경험)
1. **Week 1**: `04_midi_processing/` - 데이터 파이프라인
2. **Month 1-2**: `05_music_informer/` - SOTA 구현
3. **Month 2-4**: `06_improvnet/` - 재즈 모델
4. **Month 4-8**: `07_magenta_realtime/` - 실시간 시스템
5. **Month 8+**: `08_finetuning/` - 연구/프로덕션

---

## 🎓 각 Phase별 학습 목표

| Phase | 기간 | 난이도 | 핵심 기술 | 최종 산출물 |
|-------|------|--------|-----------|-------------|
| 01_foundations | 2개월 | ⭐ | Python, PyTorch, MIDI | 간단한 RNN 멜로디 생성 |
| 02_transformer | 2개월 | ⭐⭐ | Attention, Transformer | Music Transformer로 피아노 생성 |
| 03_diffusion | 2개월 | ⭐⭐⭐ | DDPM, Noise Scheduling | Diffusion으로 MIDI 생성 |
| 04_midi_processing | 2개월 | ⭐⭐ | Tokenization, Augmentation | 강력한 데이터 파이프라인 |
| 05_music_informer | 2개월 | ⭐⭐⭐⭐ | ProbSparse Attention | **SOTA MIDI 생성 모델** |
| 06_improvnet | 3개월 | ⭐⭐⭐⭐ | Style Transfer, Jazz Theory | **재즈 즉흥연주 생성** |
| 07_magenta_realtime | 4개월 | ⭐⭐⭐⭐⭐ | Real-time Inference, Audio | **실시간 오디오 생성** |
| 08_finetuning | 지속 | ⭐⭐⭐ | LoRA, Domain Adaptation | 나만의 커스텀 모델 |

---

## 💻 하드웨어 요구사항

### 최소 사양 (학습 가능)
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 32GB
- **저장공간**: 200GB SSD
- **학습 시간**: ~2-3일/모델

### 권장 사양 (빠른 학습)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 64GB
- **저장공간**: 500GB NVMe SSD
- **학습 시간**: ~8-12시간/모델

### 클라우드 옵션
- **Google Colab Pro+**: $50/월 (A100 40GB)
- **Vast.ai**: RTX 4090 $0.34/시간
- **Lambda Labs**: A100 $1.10/시간
- **AWS p3.2xlarge**: V100 $3.06/시간

---

## 📊 학습 후 기대 능력

### Month 10 (Music Informer 완료 후)
✅ MIDI 피아노 음악 생성 (MAESTRO 수준)
✅ 장시퀀스 효율적 처리 (1000+ 토큰)
✅ Attention 메커니즘 완전 이해
✅ PyTorch 고급 활용

### Month 13 (ImprovNet 완료 후)
✅ 재즈 즉흥연주 생성
✅ 스타일 제어 (클래식 → 재즈 변환)
✅ Chord-aware 생성
✅ 멀티트랙 앙상블 생성

### Month 17 (Magenta RealTime 완료 후)
✅ 실시간 오디오 생성 (48kHz stereo)
✅ 텍스트로 음악 제어
✅ 라이브 퍼포먼스 시스템 구축
✅ Streaming inference 최적화

### Month 17+ (파인튜닝 완료 후)
✅ 커스텀 데이터로 모델 학습
✅ 특정 스타일/작곡가 모방
✅ 프로덕션급 시스템 배포
✅ 연구 논문 작성 가능

---

## 🏆 마일스톤 & 체크포인트

- [ ] **Month 2**: 첫 RNN으로 멜로디 생성
- [ ] **Month 4**: Music Transformer로 피아노 곡 생성
- [ ] **Month 6**: Diffusion으로 다양한 스타일 생성
- [ ] **Month 8**: MIDI 데이터 완벽 처리
- [ ] **Month 10**: 🏆 **Music Informer (SOTA) 구현 완료**
- [ ] **Month 13**: 🎺 **재즈 즉흥연주 생성 완료**
- [ ] **Month 17**: 🎹 **실시간 오디오 생성 시스템 완료**
- [ ] **Month 17+**: 🚀 **나만의 커스텀 모델 배포**

---

## 📖 추가 자료

### 필수 논문 (읽어야 할 순서)
1. [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Transformer 기초
2. [Music Transformer (2018)](https://arxiv.org/abs/1809.04281) - 음악 생성 시작
3. [DDPM (2020)](https://arxiv.org/abs/2006.11239) - Diffusion 기초
4. [Music Informer (2025)](https://www.nature.com/articles/...) - SOTA MIDI
5. [ImprovNet (2025)](https://arxiv.org/abs/2502.04522) - 재즈 특화

### 유용한 링크
- [MAESTRO 데이터셋](https://magenta.tensorflow.org/datasets/maestro)
- [PiJAMA 재즈 데이터셋](https://github.com/anonymous/pijama)
- [Magenta GitHub](https://github.com/magenta/magenta)
- [HuggingFace Audio Course](https://huggingface.co/learn/audio-course/)

---

## 🤝 기여 & 커뮤니티

- **이슈**: 버그 리포트, 질문
- **PR**: 코드 개선, 새로운 모델 추가
- **토론**: 학습 팁, 하이퍼파라미터 공유

---

## 📜 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 🙏 감사의 글

이 프로젝트는 다음 연구들을 기반으로 합니다:
- Google Magenta Team
- Music Informer 저자들 (Nature 2025)
- ImprovNet 저자들 (2025)
- PyTorch & HuggingFace 커뮤니티

---

**🎵 음악과 AI의 만남, 지금 시작하세요!**

```bash
cd 01_foundations
python 01_python_basics/hello_music_ai.py
```

