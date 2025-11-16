"""
MIDI 처리 유틸리티
- MIDI 파일 읽기/쓰기
- 토크나이제이션
- 시각화
"""

import pretty_midi
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import mido


def load_midi(midi_path: str) -> pretty_midi.PrettyMIDI:
    """
    MIDI 파일 로드

    Args:
        midi_path: MIDI 파일 경로

    Returns:
        PrettyMIDI 객체
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        return midi
    except Exception as e:
        raise ValueError(f"MIDI 파일 로드 실패: {midi_path}\n{e}")


def save_midi(midi: pretty_midi.PrettyMIDI, output_path: str):
    """
    MIDI 파일 저장

    Args:
        midi: PrettyMIDI 객체
        output_path: 출력 경로
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    midi.write(output_path)
    print(f"MIDI 저장됨: {output_path}")


def midi_to_piano_roll(
    midi: pretty_midi.PrettyMIDI,
    fs: int = 100,
    pedal_threshold: int = 64
) -> np.ndarray:
    """
    MIDI를 피아노 롤로 변환

    Args:
        midi: PrettyMIDI 객체
        fs: 샘플링 주파수 (Hz)
        pedal_threshold: 페달 임계값

    Returns:
        Piano roll (128 x T)
    """
    piano_roll = midi.get_piano_roll(fs=fs, pedal_threshold=pedal_threshold)
    return piano_roll


def piano_roll_to_notes(
    piano_roll: np.ndarray,
    fs: int = 100,
    velocity: int = 100
) -> List[pretty_midi.Note]:
    """
    피아노 롤을 노트 리스트로 변환

    Args:
        piano_roll: Piano roll (128 x T)
        fs: 샘플링 주파수
        velocity: 기본 velocity

    Returns:
        Note 리스트
    """
    notes = []

    for pitch in range(128):
        # 해당 pitch의 onset/offset 찾기
        note_events = np.diff(np.concatenate([[0], piano_roll[pitch] > 0, [0]]))
        onsets = np.where(note_events == 1)[0]
        offsets = np.where(note_events == -1)[0]

        for onset, offset in zip(onsets, offsets):
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=onset / fs,
                end=offset / fs
            )
            notes.append(note)

    return sorted(notes, key=lambda x: x.start)


def extract_notes(midi: pretty_midi.PrettyMIDI) -> List[Dict]:
    """
    MIDI에서 노트 정보 추출

    Returns:
        [{pitch, start, end, velocity, instrument}, ...]
    """
    notes = []

    for inst_idx, instrument in enumerate(midi.instruments):
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity,
                'instrument': inst_idx,
                'duration': note.end - note.start
            })

    return sorted(notes, key=lambda x: x['start'])


def notes_to_midi(
    notes: List[Dict],
    output_path: str,
    program: int = 0,
    tempo: int = 120
):
    """
    노트 리스트를 MIDI 파일로 저장

    Args:
        notes: [{pitch, start, end, velocity}, ...]
        output_path: 출력 경로
        program: MIDI 프로그램 번호 (0=피아노)
        tempo: BPM
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program)

    for note_dict in notes:
        note = pretty_midi.Note(
            velocity=note_dict.get('velocity', 100),
            pitch=note_dict['pitch'],
            start=note_dict['start'],
            end=note_dict['end']
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    save_midi(midi, output_path)


def get_tempo_changes(midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, float]]:
    """
    템포 변화 추출

    Returns:
        [(time, tempo), ...]
    """
    return midi.get_tempo_changes()


def get_time_signature_changes(midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, int, int]]:
    """
    박자 변화 추출

    Returns:
        [(time, numerator, denominator), ...]
    """
    return midi.time_signature_changes


def transpose_midi(midi: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
    """
    MIDI 트랜스포즈 (데이터 증강용)

    Args:
        midi: 원본 MIDI
        semitones: 반음 이동 (-12 ~ 12)

    Returns:
        트랜스포즈된 MIDI
    """
    transposed = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())

    for instrument in midi.instruments:
        new_inst = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )

        for note in instrument.notes:
            new_pitch = note.pitch + semitones
            if 0 <= new_pitch <= 127:  # MIDI 범위 체크
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=new_pitch,
                    start=note.start,
                    end=note.end
                )
                new_inst.notes.append(new_note)

        transposed.instruments.append(new_inst)

    return transposed


def time_stretch_midi(midi: pretty_midi.PrettyMIDI, factor: float) -> pretty_midi.PrettyMIDI:
    """
    MIDI 시간 늘이기/줄이기

    Args:
        factor: 배율 (0.5 = 2배 빠르게, 2.0 = 2배 느리게)
    """
    stretched = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo() / factor)

    for instrument in midi.instruments:
        new_inst = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name
        )

        for note in instrument.notes:
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start * factor,
                end=note.end * factor
            )
            new_inst.notes.append(new_note)

        stretched.instruments.append(new_inst)

    return stretched


def get_midi_stats(midi: pretty_midi.PrettyMIDI) -> Dict:
    """
    MIDI 통계 정보

    Returns:
        {duration, num_notes, pitch_range, tempo, ...}
    """
    notes = extract_notes(midi)
    pitches = [n['pitch'] for n in notes]

    return {
        'duration': midi.get_end_time(),
        'num_notes': len(notes),
        'num_instruments': len([i for i in midi.instruments if not i.is_drum]),
        'pitch_min': min(pitches) if pitches else 0,
        'pitch_max': max(pitches) if pitches else 0,
        'pitch_range': max(pitches) - min(pitches) if pitches else 0,
        'tempo': midi.estimate_tempo(),
        'time_signatures': len(midi.time_signature_changes),
    }


def visualize_piano_roll(
    piano_roll: np.ndarray,
    fs: int = 100,
    save_path: Optional[str] = None
):
    """
    피아노 롤 시각화
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        piano_roll,
        aspect='auto',
        origin='lower',
        cmap='Blues',
        interpolation='nearest'
    )
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Pitch')
    ax.set_title('Piano Roll')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


# ===== 간단한 사용 예시 =====
if __name__ == "__main__":
    print("🎹 MIDI 유틸리티 테스트")

    # 간단한 MIDI 생성
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)

    # C 메이저 스케일
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        piano.notes.append(note)

    midi.instruments.append(piano)

    # 저장
    save_midi(midi, "/tmp/test_scale.mid")

    # 통계
    stats = get_midi_stats(midi)
    print(f"통계: {stats}")

    print("✅ MIDI 유틸리티 정상 작동")
