"""
Generator 패턴 학습
- 대용량 데이터 효율적 처리
- 무한 시퀀스 생성
- 음악 데이터 스트리밍
"""

from pathlib import Path
from typing import Generator, List
import random


# ===== 기본 Generator =====

def simple_generator() -> Generator[int, None, None]:
    """
    가장 간단한 Generator

    yield를 사용하면 함수가 Generator가 됩니다.
    """
    print("Start")
    yield 1
    print("Middle")
    yield 2
    print("End")
    yield 3


def test_simple_generator():
    print("=== 간단한 Generator ===")
    gen = simple_generator()

    print(f"next(gen): {next(gen)}")  # Start, 1
    print(f"next(gen): {next(gen)}")  # Middle, 2
    print(f"next(gen): {next(gen)}")  # End, 3

    # StopIteration 발생
    # print(next(gen))


# ===== 무한 Generator =====

def infinite_counter(start: int = 0) -> Generator[int, None, None]:
    """
    무한 카운터 Generator
    """
    count = start
    while True:
        yield count
        count += 1


def test_infinite_counter():
    print("\n=== 무한 Generator ===")
    counter = infinite_counter(100)

    # 처음 10개만 출력
    for i, value in enumerate(counter):
        print(value, end=" ")
        if i >= 9:
            break
    print()


# ===== MIDI 데이터 Generator (실용 예제) =====

def midi_note_generator(
    pitch_range: tuple[int, int] = (60, 72),
    num_notes: int = 100
) -> Generator[dict, None, None]:
    """
    랜덤 MIDI 노트 생성기

    Args:
        pitch_range: (min_pitch, max_pitch)
        num_notes: 생성할 노트 수

    Yields:
        {'pitch': int, 'velocity': int, 'duration': float}
    """
    current_time = 0.0

    for _ in range(num_notes):
        note = {
            'pitch': random.randint(*pitch_range),
            'velocity': random.randint(60, 100),
            'duration': random.choice([0.25, 0.5, 0.75, 1.0]),
            'start_time': current_time
        }

        current_time += note['duration']
        yield note


def test_midi_note_generator():
    print("\n=== MIDI 노트 Generator ===")
    note_gen = midi_note_generator(pitch_range=(60, 72), num_notes=5)

    for note in note_gen:
        print(f"Pitch: {note['pitch']}, Velocity: {note['velocity']}, "
              f"Duration: {note['duration']}, Time: {note['start_time']:.2f}")


# ===== 파일 스트리밍 Generator =====

def file_line_generator(file_path: str) -> Generator[str, None, None]:
    """
    파일을 한 줄씩 읽는 Generator
    (대용량 파일 처리에 유용)
    """
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()


def batch_generator(
    data: List,
    batch_size: int
) -> Generator[List, None, None]:
    """
    데이터를 배치로 나누는 Generator

    Args:
        data: 전체 데이터
        batch_size: 배치 크기

    Yields:
        배치 (리스트)
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def test_batch_generator():
    print("\n=== 배치 Generator ===")
    data = list(range(25))  # 0-24
    batch_size = 5

    for batch_idx, batch in enumerate(batch_generator(data, batch_size)):
        print(f"Batch {batch_idx}: {batch}")


# ===== Generator Expression =====

def test_generator_expression():
    """
    Generator Expression (간결한 문법)
    """
    print("\n=== Generator Expression ===")

    # List comprehension (모두 메모리에 로드)
    squares_list = [x**2 for x in range(10)]
    print(f"List: {squares_list}")

    # Generator expression (lazy evaluation)
    squares_gen = (x**2 for x in range(10))
    print(f"Generator: {squares_gen}")
    print(f"Values: {list(squares_gen)}")


# ===== 음악 시퀀스 Generator =====

def melody_sequence_generator(
    scale: List[int],
    length: int,
    start_note: int = None
) -> Generator[int, None, None]:
    """
    음계를 따라 멜로디 생성

    Args:
        scale: 음계 (예: [60, 62, 64, 65, 67, 69, 71, 72])
        length: 멜로디 길이
        start_note: 시작 음 (None이면 랜덤)

    Yields:
        MIDI 음높이 (pitch)
    """
    if start_note is None:
        current_note = random.choice(scale)
    else:
        current_note = start_note

    yield current_note

    for _ in range(length - 1):
        # 현재 음에서 가까운 음으로 이동
        current_idx = scale.index(current_note) if current_note in scale else 0

        # 위로 또는 아래로 1-2 스텝 이동
        step = random.choice([-2, -1, 1, 2])
        new_idx = (current_idx + step) % len(scale)
        current_note = scale[new_idx]

        yield current_note


def test_melody_generator():
    print("\n=== 멜로디 Generator ===")

    # C Major 스케일
    c_major = [60, 62, 64, 65, 67, 69, 71, 72]

    melody_gen = melody_sequence_generator(c_major, length=16, start_note=60)

    melody = list(melody_gen)
    print(f"생성된 멜로디 (MIDI pitches): {melody}")

    # 음이름으로 변환
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    melody_names = [note_names[pitch % 12] for pitch in melody]
    print(f"음이름: {melody_names}")


# ===== Generator 메모리 효율성 비교 =====

def test_memory_efficiency():
    """
    Generator vs List 메모리 사용량 비교
    """
    print("\n=== 메모리 효율성 비교 ===")
    import sys

    # List (모든 데이터 메모리에 저장)
    large_list = [x**2 for x in range(1000000)]
    print(f"List 크기: {sys.getsizeof(large_list) / 1024 / 1024:.2f} MB")

    # Generator (필요할 때만 생성)
    large_gen = (x**2 for x in range(1000000))
    print(f"Generator 크기: {sys.getsizeof(large_gen) / 1024:.2f} KB")

    print("👉 Generator가 훨씬 메모리 효율적!")


# ===== send()와 양방향 Generator =====

def echo_generator():
    """
    양방향 Generator (send 사용)
    """
    print("Generator 시작")
    while True:
        received = yield
        print(f"받은 값: {received}")


def test_send():
    print("\n=== send()를 사용한 양방향 Generator ===")
    gen = echo_generator()
    next(gen)  # 처음 시작

    gen.send("안녕하세요")
    gen.send(123)
    gen.send([1, 2, 3])


# ===== 실행 =====

def main():
    """모든 Generator 예제 실행"""
    print("🎵 Generator 패턴 학습\n")

    test_simple_generator()
    test_infinite_counter()
    test_midi_note_generator()
    test_batch_generator()
    test_generator_expression()
    test_melody_generator()
    test_memory_efficiency()
    # test_send()  # 주석 해제하면 무한 루프

    print("\n✅ Generator 학습 완료!")
    print("\n💡 핵심 요약:")
    print("1. Generator는 yield로 값을 하나씩 반환")
    print("2. 메모리 효율적 (대용량 데이터 처리)")
    print("3. 무한 시퀀스 생성 가능")
    print("4. 음악 데이터 스트리밍에 적합")


if __name__ == "__main__":
    main()
