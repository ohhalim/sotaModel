"""
Type Hints 학습
- 코드 안정성 향상
- IDE 자동완성
- 타입 체킹
"""

from typing import List, Dict, Tuple, Optional, Union, Callable, Any, TypeVar, Generic
import torch
import torch.nn as nn


# ===== 기본 Type Hints =====

def add_numbers(a: int, b: int) -> int:
    """정수 더하기"""
    return a + b


def greet(name: str) -> str:
    """인사"""
    return f"Hello, {name}!"


def is_even(n: int) -> bool:
    """짝수 판별"""
    return n % 2 == 0


# ===== 컬렉션 Type Hints =====

def sum_list(numbers: List[int]) -> int:
    """리스트 합계"""
    return sum(numbers)


def get_note_info(note_dict: Dict[str, Any]) -> str:
    """노트 정보 추출"""
    return f"Pitch: {note_dict['pitch']}, Velocity: {note_dict['velocity']}"


def get_min_max(numbers: List[float]) -> Tuple[float, float]:
    """최솟값, 최댓값 반환"""
    return min(numbers), max(numbers)


# ===== Optional =====

def find_note(notes: List[int], target: int) -> Optional[int]:
    """
    노트 찾기

    Returns:
        인덱스 또는 None (못 찾으면)
    """
    try:
        return notes.index(target)
    except ValueError:
        return None


def get_config_value(config: Dict[str, Any], key: str) -> Optional[str]:
    """설정값 가져오기 (없으면 None)"""
    return config.get(key)


# ===== Union =====

def process_input(data: Union[int, str, List[int]]) -> str:
    """
    여러 타입 처리

    Args:
        data: int, str, 또는 List[int]
    """
    if isinstance(data, int):
        return f"정수: {data}"
    elif isinstance(data, str):
        return f"문자열: {data}"
    elif isinstance(data, list):
        return f"리스트 (길이 {len(data)})"
    else:
        return "알 수 없는 타입"


# ===== Callable =====

def apply_function(
    data: List[int],
    func: Callable[[int], int]
) -> List[int]:
    """
    리스트의 각 요소에 함수 적용

    Args:
        data: 데이터 리스트
        func: int -> int 함수
    """
    return [func(x) for x in data]


def double(x: int) -> int:
    return x * 2


# ===== TypeVar (Generic) =====

T = TypeVar('T')


def first_element(items: List[T]) -> Optional[T]:
    """첫 번째 요소 반환 (Generic)"""
    return items[0] if items else None


def reverse_list(items: List[T]) -> List[T]:
    """리스트 뒤집기 (Generic)"""
    return items[::-1]


# ===== 클래스 Type Hints =====

class Note:
    """MIDI 노트"""

    def __init__(self, pitch: int, velocity: int, duration: float):
        self.pitch: int = pitch
        self.velocity: int = velocity
        self.duration: float = duration

    def __repr__(self) -> str:
        return f"Note(pitch={self.pitch}, velocity={self.velocity}, duration={self.duration})"


def transpose_note(note: Note, semitones: int) -> Note:
    """노트 트랜스포즈"""
    return Note(
        pitch=note.pitch + semitones,
        velocity=note.velocity,
        duration=note.duration
    )


# ===== PyTorch 모델 Type Hints =====

def create_model(
    input_dim: int,
    hidden_dim: int,
    output_dim: int
) -> nn.Module:
    """간단한 MLP 모델 생성"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
) -> float:
    """
    한 스텝 학습

    Returns:
        loss 값
    """
    optimizer.zero_grad()
    outputs = model(batch['input'])
    loss = criterion(outputs, batch['target'])
    loss.backward()
    optimizer.step()
    return loss.item()


# ===== 복잡한 타입 =====

NoteSequence = List[Note]
MIDIData = Dict[str, Union[int, float, List[int]]]
TrainingConfig = Dict[str, Union[int, float, str, bool]]


def process_sequence(seq: NoteSequence) -> MIDIData:
    """노트 시퀀스를 MIDI 데이터로 변환"""
    return {
        'num_notes': len(seq),
        'pitches': [note.pitch for note in seq],
        'duration': sum(note.duration for note in seq)
    }


def validate_config(config: TrainingConfig) -> bool:
    """설정 검증"""
    required_keys = ['batch_size', 'learning_rate', 'epochs']
    return all(key in config for key in required_keys)


# ===== Protocol (덕 타이핑) =====

from typing import Protocol


class Playable(Protocol):
    """연주 가능한 객체"""

    def play(self) -> None:
        ...


class Piano:
    def play(self) -> None:
        print("🎹 피아노 연주")


class Guitar:
    def play(self) -> None:
        print("🎸 기타 연주")


def perform(instrument: Playable) -> None:
    """악기 연주"""
    instrument.play()


# ===== Generic 클래스 =====

class Stack(Generic[T]):
    """Generic 스택"""

    def __init__(self):
        self._items: List[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> Optional[T]:
        return self._items.pop() if self._items else None

    def peek(self) -> Optional[T]:
        return self._items[-1] if self._items else None


# ===== 테스트 =====

def main():
    """모든 Type Hints 예제 실행"""
    print("📝 Type Hints 학습\n")

    print("=== 1. 기본 타입 ===")
    print(f"add_numbers(10, 20) = {add_numbers(10, 20)}")
    print(f"greet('Alice') = {greet('Alice')}")
    print(f"is_even(42) = {is_even(42)}")

    print("\n=== 2. 컬렉션 ===")
    numbers = [1, 2, 3, 4, 5]
    print(f"sum_list({numbers}) = {sum_list(numbers)}")

    min_val, max_val = get_min_max([1.5, 2.7, 0.3, 4.2])
    print(f"min_max = ({min_val}, {max_val})")

    print("\n=== 3. Optional ===")
    notes = [60, 64, 67, 72]
    print(f"find_note({notes}, 67) = {find_note(notes, 67)}")
    print(f"find_note({notes}, 99) = {find_note(notes, 99)}")

    print("\n=== 4. Union ===")
    print(process_input(42))
    print(process_input("hello"))
    print(process_input([1, 2, 3]))

    print("\n=== 5. Callable ===")
    data = [1, 2, 3, 4, 5]
    doubled = apply_function(data, double)
    print(f"apply_function({data}, double) = {doubled}")

    # Lambda도 가능
    squared = apply_function(data, lambda x: x ** 2)
    print(f"apply_function({data}, square) = {squared}")

    print("\n=== 6. Generic (TypeVar) ===")
    print(f"first_element([1, 2, 3]) = {first_element([1, 2, 3])}")
    print(f"first_element(['a', 'b', 'c']) = {first_element(['a', 'b', 'c'])}")
    print(f"reverse_list([1, 2, 3]) = {reverse_list([1, 2, 3])}")

    print("\n=== 7. 클래스 타입 ===")
    note = Note(pitch=60, velocity=100, duration=0.5)
    print(f"원본: {note}")
    transposed = transpose_note(note, 12)
    print(f"12반음 상승: {transposed}")

    print("\n=== 8. 복잡한 타입 ===")
    sequence: NoteSequence = [
        Note(60, 100, 0.5),
        Note(64, 90, 0.5),
        Note(67, 95, 1.0)
    ]
    midi_data = process_sequence(sequence)
    print(f"MIDI 데이터: {midi_data}")

    print("\n=== 9. Protocol (덕 타이핑) ===")
    perform(Piano())
    perform(Guitar())

    print("\n=== 10. Generic 클래스 ===")
    int_stack: Stack[int] = Stack()
    int_stack.push(1)
    int_stack.push(2)
    int_stack.push(3)
    print(f"pop: {int_stack.pop()}")
    print(f"peek: {int_stack.peek()}")

    str_stack: Stack[str] = Stack()
    str_stack.push("A")
    str_stack.push("B")
    print(f"pop: {str_stack.pop()}")

    print("\n✅ Type Hints 학습 완료!")
    print("\n💡 핵심 요약:")
    print("1. Type Hints는 코드 안정성 향상")
    print("2. IDE 자동완성 지원")
    print("3. mypy로 타입 체킹 가능")
    print("4. 런타임에는 영향 없음 (힌트일 뿐)")
    print("5. 대규모 프로젝트에서 필수!")


if __name__ == "__main__":
    main()
