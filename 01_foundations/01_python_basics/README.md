# Week 1-2: Python 고급 문법

> 음악 AI를 위한 필수 Python 기술

## 🎯 학습 목표

1. **Generator** - 대용량 MIDI 데이터 효율적 처리
2. **Decorator** - 코드 재사용 및 로깅 자동화
3. **Type Hints** - 코드 안정성 향상
4. **Context Manager** - 리소스 관리

---

## 📚 학습 순서

```bash
# 1. Generator 패턴
python generators.py

# 2. Decorator 패턴
python decorators.py

# 3. Type Hints
python type_hints.py

# 4. Context Manager
python context_managers.py

# 5. 연습 문제
python exercises.py
```

---

## 🔑 핵심 개념

### 1. Generator
대용량 데이터를 메모리 효율적으로 처리

```python
# ❌ 나쁜 예: 모든 데이터를 메모리에 로드
def load_all_midis(directory):
    midis = []
    for file in os.listdir(directory):
        midis.append(load_midi(file))  # 메모리 부족!
    return midis

# ✅ 좋은 예: Generator 사용
def load_midis_generator(directory):
    for file in os.listdir(directory):
        yield load_midi(file)  # 한 번에 하나씩
```

### 2. Decorator
함수에 기능 추가 (로깅, 타이밍 등)

```python
@timer  # 실행 시간 자동 측정
@logger  # 자동 로깅
def train_model():
    ...
```

### 3. Type Hints
코드 안정성 향상

```python
def generate_music(
    model: nn.Module,
    length: int,
    temperature: float = 1.0
) -> List[int]:
    ...
```

---

## ✅ 완료 체크리스트

- [ ] generators.py 실행 및 이해
- [ ] decorators.py 실행 및 이해
- [ ] type_hints.py 실행 및 이해
- [ ] context_managers.py 실행 및 이해
- [ ] exercises.py 모든 문제 해결

**다음: 02_pytorch_basics**
