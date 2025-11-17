"""
Chord Recognition from MIDI
실제 작동하는 코드 감지 알고리즘
"""

import pretty_midi
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


class ChordRecognizer:
    """
    MIDI에서 코드 진행 추출

    Method: Template matching + music theory
    """

    def __init__(self, time_resolution: float = 0.25):
        """
        Args:
            time_resolution: 코드 분석 시간 해상도 (초)
        """
        self.time_resolution = time_resolution

        # 코드 템플릿 정의 (pitch class sets)
        self.chord_templates = self._build_chord_templates()

        # 코드 이름 매핑
        self.chord_names = self._build_chord_names()

    def _build_chord_templates(self) -> Dict[str, List[int]]:
        """코드 타입별 pitch class set"""
        return {
            # Major chords
            'maj': [0, 4, 7],
            'maj7': [0, 4, 7, 11],
            '6': [0, 4, 7, 9],

            # Dominant chords
            '7': [0, 4, 7, 10],
            '9': [0, 4, 7, 10, 14 % 12],  # = [0, 4, 7, 10, 2]
            '13': [0, 4, 7, 10, 14 % 12, 21 % 12],  # = [0, 4, 7, 10, 2, 9]

            # Minor chords
            'm': [0, 3, 7],
            'm7': [0, 3, 7, 10],
            'm6': [0, 3, 7, 9],
            'm9': [0, 3, 7, 10, 14 % 12],

            # Diminished/Altered
            'dim': [0, 3, 6],
            'dim7': [0, 3, 6, 9],
            'm7b5': [0, 3, 6, 10],  # Half-diminished

            # Augmented
            'aug': [0, 4, 8],
            '7#5': [0, 4, 8, 10],

            # Sus chords
            'sus2': [0, 2, 7],
            'sus4': [0, 5, 7],
        }

    def _build_chord_names(self) -> Dict[int, str]:
        """Chord ID → Name 매핑"""
        names = {}
        idx = 0

        roots = ['C', 'Db', 'D', 'Eb', 'E', 'F',
                'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

        # 주요 재즈 코드만
        types = ['maj7', '7', 'm7', 'm7b5', 'dim7']

        for root_idx, root in enumerate(roots):
            for chord_type in types:
                chord_name = f"{root}{chord_type}"
                names[idx] = chord_name
                idx += 1

        return names

    def recognize_from_midi(
        self,
        midi_path: str,
        min_notes_for_chord: int = 2
    ) -> List[Dict]:
        """
        MIDI 파일에서 코드 진행 추출

        Args:
            midi_path: MIDI 파일 경로
            min_notes_for_chord: 코드로 인식할 최소 동시발음 노트 수

        Returns:
            [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'root': 2,  # D
                    'type': 'm7',
                    'name': 'Dm7',
                    'chord_id': 14
                },
                ...
            ]
        """
        # MIDI 로드
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Warning: Cannot load {midi_path}: {e}")
            return []

        # 전체 길이
        end_time = midi.get_end_time()

        # 시간 슬라이스별로 활성 노트 추출
        chords = []
        current_time = 0.0

        while current_time < end_time:
            # 현재 시간에 울리고 있는 노트들
            active_pitches = self._get_active_pitches_at_time(
                midi,
                current_time
            )

            if len(active_pitches) >= min_notes_for_chord:
                # Pitch class histogram
                pitch_classes = [p % 12 for p in active_pitches]

                # 코드 인식
                chord_info = self._recognize_chord_from_pitches(pitch_classes)

                if chord_info:
                    # 연속된 같은 코드는 병합
                    if chords and chords[-1]['name'] == chord_info['name']:
                        chords[-1]['end'] = current_time + self.time_resolution
                    else:
                        chord_info['start'] = current_time
                        chord_info['end'] = current_time + self.time_resolution
                        chords.append(chord_info)

            current_time += self.time_resolution

        return chords

    def _get_active_pitches_at_time(
        self,
        midi: pretty_midi.PrettyMIDI,
        time: float
    ) -> List[int]:
        """특정 시간에 울리고 있는 모든 pitch"""
        active = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                if note.start <= time < note.end:
                    active.append(note.pitch)

        return active

    def _recognize_chord_from_pitches(
        self,
        pitch_classes: List[int]
    ) -> Optional[Dict]:
        """
        Pitch class set에서 코드 인식

        Method: Template matching with best overlap
        """
        if not pitch_classes:
            return None

        # Pitch class histogram
        pc_set = set(pitch_classes)

        # 가장 많이 나온 음을 루트로 가정
        pc_counts = Counter(pitch_classes)
        most_common_pcs = [pc for pc, _ in pc_counts.most_common(3)]

        best_match = None
        best_score = 0

        # 각 가능한 루트에 대해 테스트
        for potential_root in most_common_pcs:
            # 루트를 0으로 정규화
            normalized_pcs = [(pc - potential_root) % 12 for pc in pc_set]
            normalized_set = set(normalized_pcs)

            # 각 코드 타입과 매칭
            for chord_type, template in self.chord_templates.items():
                template_set = set(template)

                # Jaccard similarity
                intersection = len(normalized_set & template_set)
                union = len(normalized_set | template_set)

                if union > 0:
                    score = intersection / union

                    # Prefer exact matches
                    if normalized_set == template_set:
                        score += 0.5

                    if score > best_score:
                        best_score = score
                        best_match = {
                            'root': potential_root,
                            'type': chord_type,
                            'score': score
                        }

        # 최소 스코어 이상만 인식
        if best_match and best_match['score'] >= 0.5:
            # 코드 이름
            root_names = ['C', 'Db', 'D', 'Eb', 'E', 'F',
                         'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
            chord_name = f"{root_names[best_match['root']]}{best_match['type']}"

            # Chord ID 계산 (간단한 해싱)
            chord_id = self._get_chord_id(best_match['root'], best_match['type'])

            return {
                'root': best_match['root'],
                'type': best_match['type'],
                'name': chord_name,
                'chord_id': chord_id,
                'confidence': best_match['score']
            }

        # 인식 실패 → NC (No Chord)
        return {
            'root': 0,
            'type': 'NC',
            'name': 'NC',
            'chord_id': 0,
            'confidence': 0.0
        }

    def _get_chord_id(self, root: int, chord_type: str) -> int:
        """Root + Type → Chord ID"""
        # 재즈 주요 타입만
        type_map = {
            'maj7': 0,
            '7': 1,
            'm7': 2,
            'm7b5': 3,
            'dim7': 4,
            'NC': 0
        }

        type_id = type_map.get(chord_type, 0)

        # ID = root * 5 + type
        return root * 5 + type_id + 1  # +1 to reserve 0 for PAD


# ===== 사용 예시 =====

if __name__ == "__main__":
    print("🎹 Chord Recognition 테스트\n")

    # 테스트용 MIDI 생성
    import os
    test_midi = "/tmp/test_chord_recognition.mid"

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)

    # Dm7 코드 (D, F, A, C)
    for pitch in [62, 65, 69, 72]:  # D3, F3, A3, C4
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=0.0,
            end=2.0
        )
        piano.notes.append(note)

    # G7 코드 (G, B, D, F)
    for pitch in [67, 71, 74, 77]:  # G3, B3, D4, F4
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=2.0,
            end=4.0
        )
        piano.notes.append(note)

    # Cmaj7 코드 (C, E, G, B)
    for pitch in [60, 64, 67, 71]:  # C3, E3, G3, B3
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=4.0,
            end=6.0
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(test_midi)

    # 코드 인식
    recognizer = ChordRecognizer(time_resolution=0.25)
    chords = recognizer.recognize_from_midi(test_midi)

    print(f"인식된 코드: {len(chords)}개\n")

    for chord in chords:
        print(f"  {chord['start']:.2f}s - {chord['end']:.2f}s: "
              f"{chord['name']} (confidence: {chord['confidence']:.2f}, "
              f"ID: {chord['chord_id']})")

    print("\n✅ Chord Recognition 테스트 완료!")

    # Cleanup
    os.remove(test_midi)
