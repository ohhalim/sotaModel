"""
MIDI Tokenizer for JazzFlow-RT
Multiple tokenization schemes: REMI, Compound, MuMIDI
"""

import pretty_midi
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    """토큰 타입"""
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    TIME_SHIFT = "time_shift"
    VELOCITY = "velocity"
    TEMPO = "tempo"
    BAR = "bar"
    POSITION = "position"
    CHORD = "chord"
    PAD = "pad"
    BOS = "bos"
    EOS = "eos"


@dataclass
class MIDIToken:
    """단일 MIDI 토큰"""
    type: TokenType
    value: int
    time: float = 0.0


class REMITokenizer:
    """
    REMI Tokenizer (Revamped MIDI)

    토큰 종류:
    - Note On (pitch)
    - Note Off (pitch)
    - Velocity (quantized to 32 levels)
    - Time Shift (quantized to 100ms units)
    - Bar
    - Position (within bar, 16th notes)
    - Tempo
    - Chord (재즈용 확장)
    """

    def __init__(
        self,
        pitch_range: Tuple[int, int] = (21, 108),
        velocity_bins: int = 32,
        time_shift_bins: int = 100,  # 100ms units
        max_time_shift: float = 10.0,  # seconds
        tempo_bins: int = 60,
        tempo_range: Tuple[int, int] = (40, 220),
        beat_resolution: int = 16  # 16th notes
    ):
        self.pitch_range = pitch_range
        self.velocity_bins = velocity_bins
        self.time_shift_bins = time_shift_bins
        self.max_time_shift = max_time_shift
        self.tempo_bins = tempo_bins
        self.tempo_range = tempo_range
        self.beat_resolution = beat_resolution

        # 토큰 vocabulary 구축
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)

        # ID <-> Token 매핑
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = {i: token for token, i in self.token2id.items()}

    def _build_vocab(self) -> List[str]:
        """Vocabulary 구축"""
        vocab = []

        # Special tokens
        vocab.append("<PAD>")
        vocab.append("<BOS>")
        vocab.append("<EOS>")
        vocab.append("<UNK>")

        # Bar token
        vocab.append("Bar")

        # Position tokens (within bar)
        for pos in range(self.beat_resolution):
            vocab.append(f"Position_{pos}")

        # Note On tokens
        for pitch in range(self.pitch_range[0], self.pitch_range[1] + 1):
            vocab.append(f"Note_On_{pitch}")

        # Note Off tokens
        for pitch in range(self.pitch_range[0], self.pitch_range[1] + 1):
            vocab.append(f"Note_Off_{pitch}")

        # Velocity tokens
        for vel in range(self.velocity_bins):
            vocab.append(f"Velocity_{vel}")

        # Time Shift tokens
        for ts in range(self.time_shift_bins):
            vocab.append(f"Time_Shift_{ts}")

        # Tempo tokens
        for tempo in range(self.tempo_bins):
            vocab.append(f"Tempo_{tempo}")

        # Chord tokens (재즈 코드들)
        jazz_chords = self._get_jazz_chords()
        for chord in jazz_chords:
            vocab.append(f"Chord_{chord}")

        return vocab

    def _get_jazz_chords(self) -> List[str]:
        """재즈 코드 리스트"""
        roots = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        types = ['maj7', '7', 'm7', 'm7b5', 'dim7', 'aug', '6', 'm6']

        chords = []
        for root in roots:
            for chord_type in types:
                chords.append(f"{root}{chord_type}")

        return chords

    def encode(
        self,
        midi_path: str,
        add_bos_eos: bool = True
    ) -> List[int]:
        """
        MIDI 파일을 토큰 시퀀스로 인코딩

        Args:
            midi_path: MIDI 파일 경로
            add_bos_eos: BOS/EOS 토큰 추가 여부

        Returns:
            토큰 ID 리스트
        """
        # MIDI 로드
        midi = pretty_midi.PrettyMIDI(midi_path)

        # 이벤트 추출 및 정렬
        events = self._extract_events(midi)
        events = sorted(events, key=lambda x: (x['time'], x['type']))

        # 토큰화
        tokens = []

        if add_bos_eos:
            tokens.append(self.token2id["<BOS>"])

        current_time = 0.0
        current_bar = 0

        for event in events:
            # Time shift
            time_diff = event['time'] - current_time
            if time_diff > 0:
                time_shift_id = int(time_diff / (self.max_time_shift / self.time_shift_bins))
                time_shift_id = min(time_shift_id, self.time_shift_bins - 1)
                token = f"Time_Shift_{time_shift_id}"
                if token in self.token2id:
                    tokens.append(self.token2id[token])

            # Bar
            bar = int(event['time'] / 4.0)  # 4/4 time signature
            if bar > current_bar:
                tokens.append(self.token2id["Bar"])
                current_bar = bar

            # Position within bar
            position_in_bar = (event['time'] % 4.0) / 4.0
            position_id = int(position_in_bar * self.beat_resolution)
            position_id = min(position_id, self.beat_resolution - 1)
            token = f"Position_{position_id}"
            if token in self.token2id:
                tokens.append(self.token2id[token])

            # Event type specific
            if event['type'] == 'note_on':
                # Velocity
                vel_id = int(event['velocity'] / 127 * (self.velocity_bins - 1))
                vel_token = f"Velocity_{vel_id}"
                if vel_token in self.token2id:
                    tokens.append(self.token2id[vel_token])

                # Note On
                note_token = f"Note_On_{event['pitch']}"
                if note_token in self.token2id:
                    tokens.append(self.token2id[note_token])

            elif event['type'] == 'note_off':
                note_token = f"Note_Off_{event['pitch']}"
                if note_token in self.token2id:
                    tokens.append(self.token2id[note_token])

            current_time = event['time']

        if add_bos_eos:
            tokens.append(self.token2id["<EOS>"])

        return tokens

    def decode(
        self,
        token_ids: List[int],
        output_path: str,
        tempo: int = 120
    ):
        """
        토큰 시퀀스를 MIDI 파일로 디코딩

        Args:
            token_ids: 토큰 ID 리스트
            output_path: 출력 MIDI 경로
            tempo: BPM
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)

        current_time = 0.0
        current_velocity = 80
        active_notes = {}  # pitch -> start_time

        for token_id in token_ids:
            if token_id >= len(self.id2token):
                continue

            token = self.id2token[token_id]

            # Skip special tokens
            if token in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
                continue

            # Time shift
            if token.startswith("Time_Shift_"):
                shift_id = int(token.split("_")[-1])
                time_shift = (shift_id / self.time_shift_bins) * self.max_time_shift
                current_time += time_shift

            # Velocity
            elif token.startswith("Velocity_"):
                vel_id = int(token.split("_")[-1])
                current_velocity = int((vel_id / self.velocity_bins) * 127)

            # Note On
            elif token.startswith("Note_On_"):
                pitch = int(token.split("_")[-1])
                active_notes[pitch] = current_time

            # Note Off
            elif token.startswith("Note_Off_"):
                pitch = int(token.split("_")[-1])
                if pitch in active_notes:
                    start_time = active_notes[pitch]
                    note = pretty_midi.Note(
                        velocity=current_velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)
                    del active_notes[pitch]

        # Close any remaining notes
        for pitch, start_time in active_notes.items():
            note = pretty_midi.Note(
                velocity=current_velocity,
                pitch=pitch,
                start=start_time,
                end=current_time + 0.5
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        midi.write(output_path)

    def _extract_events(self, midi: pretty_midi.PrettyMIDI) -> List[Dict]:
        """MIDI에서 이벤트 추출"""
        events = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                # Note On
                events.append({
                    'time': note.start,
                    'type': 'note_on',
                    'pitch': note.pitch,
                    'velocity': note.velocity
                })

                # Note Off
                events.append({
                    'time': note.end,
                    'type': 'note_off',
                    'pitch': note.pitch,
                    'velocity': 0
                })

        return events


class CompoundTokenizer:
    """
    Compound Tokenizer (더 간단한 버전)
    하나의 토큰에 여러 정보 포함: (pitch, duration, velocity)
    """

    def __init__(
        self,
        pitch_range: Tuple[int, int] = (21, 108),
        duration_bins: int = 32,
        velocity_bins: int = 32
    ):
        self.pitch_range = pitch_range
        self.duration_bins = duration_bins
        self.velocity_bins = velocity_bins

        # Vocabulary size
        num_pitches = pitch_range[1] - pitch_range[0] + 1
        self.vocab_size = num_pitches * duration_bins * velocity_bins + 4  # +4 for special tokens

    def encode(self, midi_path: str) -> List[int]:
        """MIDI를 compound tokens로 인코딩"""
        midi = pretty_midi.PrettyMIDI(midi_path)

        tokens = [0]  # BOS

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                # Pitch
                pitch_id = note.pitch - self.pitch_range[0]

                # Duration (quantized)
                duration = note.end - note.start
                duration_id = int(duration * 2)  # 0.5초 단위
                duration_id = min(duration_id, self.duration_bins - 1)

                # Velocity (quantized)
                velocity_id = note.velocity // (128 // self.velocity_bins)
                velocity_id = min(velocity_id, self.velocity_bins - 1)

                # Compound token
                token_id = pitch_id * (self.duration_bins * self.velocity_bins) + \
                          duration_id * self.velocity_bins + \
                          velocity_id + 4  # offset for special tokens

                tokens.append(token_id)

        tokens.append(1)  # EOS

        return tokens


# ===== 사용 예시 =====

if __name__ == "__main__":
    print("🎹 MIDI Tokenizer 테스트\n")

    # REMI Tokenizer
    tokenizer = REMITokenizer()

    print(f"Vocabulary 크기: {tokenizer.vocab_size}")
    print(f"샘플 토큰들:")
    for i, token in enumerate(tokenizer.vocab[:20]):
        print(f"  {i}: {token}")

    # 간단한 MIDI 생성 및 테스트
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)

    # C Major scale
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    test_midi_path = "/tmp/test_tokenizer.mid"
    midi.write(test_midi_path)

    # Encode
    print(f"\n🔄 Encoding test MIDI...")
    tokens = tokenizer.encode(test_midi_path)
    print(f"토큰 수: {len(tokens)}")
    print(f"토큰 (처음 30개): {tokens[:30]}")

    # Decode
    print(f"\n🔄 Decoding tokens...")
    output_path = "/tmp/test_decoded.mid"
    tokenizer.decode(tokens, output_path)
    print(f"디코딩 완료: {output_path}")

    # Compound Tokenizer
    print(f"\n🔄 Compound Tokenizer 테스트...")
    compound_tokenizer = CompoundTokenizer()
    print(f"Compound Vocabulary 크기: {compound_tokenizer.vocab_size}")

    compound_tokens = compound_tokenizer.encode(test_midi_path)
    print(f"Compound 토큰 수: {len(compound_tokens)}")
    print(f"Compound 토큰: {compound_tokens}")

    print("\n✅ MIDI Tokenizer 테스트 완료!")
