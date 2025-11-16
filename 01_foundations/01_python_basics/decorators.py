"""
Decorator 패턴 학습
- 함수 실행 시간 측정
- 로깅 자동화
- 캐싱
"""

import time
import functools
from typing import Callable, Any


# ===== 기본 Decorator =====

def simple_decorator(func: Callable) -> Callable:
    """가장 간단한 Decorator"""

    def wrapper(*args, **kwargs):
        print(f"함수 {func.__name__} 실행 전")
        result = func(*args, **kwargs)
        print(f"함수 {func.__name__} 실행 후")
        return result

    return wrapper


@simple_decorator
def say_hello(name: str):
    print(f"Hello, {name}!")


# ===== 실행 시간 측정 Decorator =====

def timer(func: Callable) -> Callable:
    """함수 실행 시간 측정"""

    @functools.wraps(func)  # 원본 함수 메타데이터 보존
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time

        print(f"⏱️  {func.__name__} 실행 시간: {elapsed:.4f}초")
        return result

    return wrapper


@timer
def train_model_simulation():
    """모델 학습 시뮬레이션"""
    time.sleep(0.5)  # 0.5초 대기
    return "학습 완료"


# ===== 로깅 Decorator =====

def logger(func: Callable) -> Callable:
    """함수 호출 로깅"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_str = ', '.join(repr(a) for a in args)
        kwargs_str = ', '.join(f"{k}={v!r}" for k, v in kwargs.items())
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))

        print(f"📝 호출: {func.__name__}({all_args})")

        result = func(*args, **kwargs)

        print(f"📤 결과: {result!r}")
        return result

    return wrapper


@logger
def generate_music(length: int, temperature: float = 1.0):
    """음악 생성 시뮬레이션"""
    return f"음악 생성됨: {length}개 노트, temp={temperature}"


# ===== 파라미터를 받는 Decorator =====

def repeat(times: int):
    """함수를 여러 번 실행"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for i in range(times):
                print(f"  [{i+1}/{times}] 실행")
                result = func(*args, **kwargs)
                results.append(result)
            return results

        return wrapper

    return decorator


@repeat(times=3)
def play_note():
    """음 재생"""
    return "♪"


# ===== 캐싱 Decorator =====

def memoize(func: Callable) -> Callable:
    """결과 캐싱 (피보나치 등에 유용)"""
    cache = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        else:
            print(f"  (캐시에서 가져옴: {args})")
        return cache[args]

    return wrapper


@memoize
def fibonacci(n: int) -> int:
    """피보나치 수"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# ===== 여러 Decorator 동시 사용 =====

@timer
@logger
def complex_function(x: int, y: int):
    """여러 Decorator 적용"""
    time.sleep(0.1)
    return x + y


# ===== 클래스 기반 Decorator =====

class CountCalls:
    """함수 호출 횟수 세기"""

    def __init__(self, func: Callable):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"🔢 호출 횟수: {self.count}")
        return self.func(*args, **kwargs)


@CountCalls
def generate_note():
    """노트 생성"""
    return "C4"


# ===== 실제 활용: 학습 모니터링 =====

def monitor_training(func: Callable) -> Callable:
    """학습 진행 모니터링 Decorator"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n{'='*50}")
        print(f"🚀 {func.__name__} 시작")
        print(f"{'='*50}")

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            print(f"\n{'='*50}")
            print(f"✅ {func.__name__} 완료")
            print(f"⏱️  소요 시간: {elapsed:.2f}초")
            print(f"{'='*50}\n")

            return result

        except Exception as e:
            elapsed = time.time() - start_time

            print(f"\n{'='*50}")
            print(f"❌ {func.__name__} 실패")
            print(f"⚠️  에러: {e}")
            print(f"⏱️  소요 시간: {elapsed:.2f}초")
            print(f"{'='*50}\n")

            raise

    return wrapper


@monitor_training
def train_music_model(epochs: int):
    """음악 모델 학습"""
    for epoch in range(epochs):
        time.sleep(0.1)  # 학습 시뮬레이션
        print(f"Epoch {epoch+1}/{epochs} - Loss: {1.0 / (epoch + 1):.4f}")
    return "학습 완료"


# ===== 테스트 =====

def main():
    """모든 Decorator 예제 실행"""
    print("🎨 Decorator 패턴 학습\n")

    print("=== 1. 간단한 Decorator ===")
    say_hello("Alice")

    print("\n=== 2. 실행 시간 측정 ===")
    train_model_simulation()

    print("\n=== 3. 로깅 ===")
    generate_music(100, temperature=0.8)

    print("\n=== 4. 반복 실행 ===")
    results = play_note()
    print(f"결과: {results}")

    print("\n=== 5. 캐싱 (Memoization) ===")
    print(f"fibonacci(10) = {fibonacci(10)}")
    print(f"fibonacci(10) = {fibonacci(10)}")  # 캐시에서 가져옴

    print("\n=== 6. 여러 Decorator ===")
    complex_function(10, 20)

    print("\n=== 7. 클래스 기반 Decorator ===")
    generate_note()
    generate_note()
    generate_note()

    print("\n=== 8. 학습 모니터링 ===")
    train_music_model(epochs=5)

    print("\n✅ Decorator 학습 완료!")
    print("\n💡 핵심 요약:")
    print("1. Decorator는 함수에 기능을 추가")
    print("2. @functools.wraps로 메타데이터 보존")
    print("3. 로깅, 타이밍, 캐싱 등에 활용")
    print("4. 여러 Decorator를 겹쳐 사용 가능")


if __name__ == "__main__":
    main()
