# JazzFlow-RT 설치 가이드

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 3.8+ 필요
python --version

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

### 2. 패키지 설치

```bash
# JazzFlow-RT 디렉토리로 이동
cd 09_jazzflow_rt

# 필수 패키지 설치
pip install -r requirements.txt

# 또는 setup.py로 설치
pip install -e .
```

### 3. 빠른 테스트

```bash
# 모든 컴포넌트가 작동하는지 확인
python quick_test.py
```

예상 출력:
```
🎺 JazzFlow-RT Quick Test

============================================================
1️⃣  MIDI Tokenizer 테스트
============================================================
✅ Tokenizer 생성 성공
   Vocabulary size: 2048
   ...

============================================================
2️⃣  JazzFlow-RT 모델 테스트
============================================================
✅ 모델 생성 성공
   파라미터 수: 5.23M
   ...

🎉 모든 테스트 통과!
```

### 4. 간단한 데모

```bash
# 재즈 MIDI 생성
python demo.py --chords "Dm7,G7,Cmaj7,Am7" --style 6 --bars 8

# 출력: demo_output.mid
```

---

## 📦 상세 설치 옵션

### 최소 설치 (코어만)

```bash
pip install torch numpy pretty-midi tqdm
```

### 전체 설치 (파인튜닝 포함)

```bash
pip install -r requirements.txt
```

### GPU 지원

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 🎯 다음 단계

### 1. 데이터셋 다운로드

**MAESTRO (사전학습)**
```bash
# https://magenta.tensorflow.org/datasets/maestro
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip -d datasets/MAESTRO
```

**PiJAMA (재즈 파인튜닝)**
```bash
# https://github.com/PRamoneda/PiJAMA
git clone https://github.com/PRamoneda/PiJAMA.git datasets/PiJAMA
```

### 2. 사전학습

```bash
python training/pretrain.py \
    --data_dir ./datasets/MAESTRO \
    --epochs 100 \
    --batch_size 16 \
    --use_amp \
    --checkpoint_dir ./checkpoints/pretrain
```

### 3. QLoRA 파인튜닝

```bash
python training/finetune_qlora.py \
    --base_model ./checkpoints/pretrain/best.pt \
    --data_dir ./datasets/PiJAMA \
    --use_qlora \
    --epochs 30 \
    --output_dir ./models/jazzflow_finetuned
```

### 4. 실시간 잼 세션

```bash
python inference/live_jam.py \
    --checkpoint ./models/jazzflow_finetuned/best.pt \
    --chords "Dm7,G7,Cmaj7,Am7" \
    --style 6 \
    --tempo 140 \
    --bars 32
```

---

## ⚙️ 시스템 요구사항

### 최소 사양
- **CPU**: 4코어 이상
- **RAM**: 8GB 이상
- **저장공간**: 10GB 이상

### 권장 사양 (학습용)
- **GPU**: NVIDIA GPU (8GB VRAM 이상)
  - RTX 3060 (12GB) 이상 권장
  - Tesla T4, V100, A100
- **RAM**: 16GB 이상
- **저장공간**: 50GB 이상 (데이터셋 포함)

### 추론/생성만
- **CPU**: 작은 모델은 CPU만으로도 가능
- **RAM**: 4GB 이상
- GPU 없이도 작동 (느림)

---

## 🐛 문제 해결

### torch 설치 오류

```bash
# PyTorch 공식 사이트에서 설치
# https://pytorch.org/get-started/locally/
```

### pretty_midi 오류

```bash
pip install --upgrade pretty-midi mido
```

### CUDA out of memory

```bash
# 배치 크기 줄이기
python training/pretrain.py --batch_size 8

# Gradient accumulation 사용
python training/pretrain.py --batch_size 4 --accumulation_steps 4
```

### Import 오류

```bash
# 현재 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH="${PYTHONPATH}:/path/to/09_jazzflow_rt"
```

---

## 📝 체크리스트

- [ ] Python 3.8+ 설치
- [ ] 가상환경 생성 및 활성화
- [ ] requirements.txt 설치
- [ ] quick_test.py 실행 성공
- [ ] demo.py로 MIDI 생성 성공
- [ ] 데이터셋 다운로드 (선택)
- [ ] 모델 학습 시작 (선택)

---

## 💡 도움말

문제가 발생하면:
1. quick_test.py 실행해서 어디서 실패하는지 확인
2. Python 버전 확인 (`python --version`)
3. PyTorch 설치 확인 (`python -c "import torch; print(torch.__version__)"`)
4. Issue 제보

**모든 준비 완료!** 🎺
