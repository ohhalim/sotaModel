"""
재즈 특화 평가 메트릭
"""

import torch
import numpy as np
from typing import List, Dict
import pretty_midi
from collections import Counter


class JazzMetrics:
    """재즈 생성 평가 메트릭"""

    def __init__(self):
        # 재즈 스케일 정의
        self.jazz_scales = self._build_jazz_scales()

        # 코드-스케일 관계
        self.chord_scale_map = self._build_chord_scale_map()

    def _build_jazz_scales(self) -> Dict[str, List[int]]:
        """재즈 스케일 정의"""
        return {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'blues': [0, 3, 5, 6, 7, 10],
            'bebop_major': [0, 2, 4, 5, 7, 8, 9, 11],
            'bebop_dominant': [0, 2, 4, 5, 7, 9, 10, 11],
            'altered': [0, 1, 3, 4, 6, 8, 10],
            'wholetone': [0, 2, 4, 6, 8, 10],
        }

    def _build_chord_scale_map(self) -> Dict[str, str]:
        """코드-스케일 관계 매핑"""
        return {
            'maj7': 'major',
            '7': 'mixolydian',
            'm7': 'dorian',
            'm7b5': 'altered',
            'dim7': 'altered',
        }

    def harmonic_consistency(
        self,
        notes: List[Dict],
        chord_progression: List[Dict]
    ) -> float:
        """
        화성 일치도

        Args:
            notes: [{pitch, start, end, ...}, ...]
            chord_progression: [{chord_type, root, start, end}, ...]

        Returns:
            일치도 (0-1)
        """
        if not notes or not chord_progression:
            return 0.0

        total_duration = 0.0
        correct_duration = 0.0

        for note in notes:
            note_duration = note['end'] - note['start']
            total_duration += note_duration

            # 해당 시간의 코드 찾기
            current_chord = None
            for chord in chord_progression:
                if chord['start'] <= note['start'] < chord['end']:
                    current_chord = chord
                    break

            if current_chord is None:
                continue

            # 노트가 코드 스케일에 맞는지 확인
            chord_type = current_chord.get('type', 'maj7')
            root = current_chord.get('root', 0)

            # 스케일 가져오기
            scale_name = self.chord_scale_map.get(chord_type, 'major')
            scale = self.jazz_scales[scale_name]

            # 노트가 스케일에 속하는지
            pitch_class = (note['pitch'] - root) % 12
            if pitch_class in scale:
                correct_duration += note_duration

        consistency = correct_duration / total_duration if total_duration > 0 else 0.0

        return consistency

    def swing_ratio(self, notes: List[Dict], expected_ratio: float = 2.0) -> float:
        """
        스윙 비율 측정

        Args:
            notes: [{pitch, start, ...}, ...]
            expected_ratio: 기대 스윙 비율 (1.0=straight, 2.0=triplet swing)

        Returns:
            스윙 비율 점수 (0-1, 1=perfect)
        """
        if len(notes) < 4:
            return 0.0

        # 연속된 노트 쌍의 IOI (Inter-Onset Interval) 비율 계산
        onsets = sorted([n['start'] for n in notes])

        ratios = []
        for i in range(0, len(onsets) - 3, 2):
            ioi1 = onsets[i+1] - onsets[i]
            ioi2 = onsets[i+2] - onsets[i+1]

            if ioi2 > 0.01:  # 너무 작은 값 제외
                ratio = ioi1 / ioi2
                if 0.5 < ratio < 4.0:  # 유효 범위
                    ratios.append(ratio)

        if not ratios:
            return 0.0

        # 기대 비율과의 차이
        mean_ratio = np.mean(ratios)
        error = abs(mean_ratio - expected_ratio)

        # 점수: error가 0이면 1.0, error가 1.0 이상이면 0.0
        score = max(0.0, 1.0 - error)

        return score

    def style_diversity(self, notes: List[Dict]) -> Dict[str, float]:
        """
        스타일 다양성 측정

        Returns:
            {
                'pitch_entropy': 음높이 엔트로피,
                'rhythm_entropy': 리듬 엔트로피,
                'interval_diversity': 음정 다양성
            }
        """
        if len(notes) < 10:
            return {
                'pitch_entropy': 0.0,
                'rhythm_entropy': 0.0,
                'interval_diversity': 0.0
            }

        # Pitch entropy
        pitches = [n['pitch'] % 12 for n in notes]
        pitch_counts = Counter(pitches)
        pitch_probs = np.array(list(pitch_counts.values())) / len(pitches)
        pitch_entropy = -np.sum(pitch_probs * np.log2(pitch_probs + 1e-10))

        # Rhythm entropy (quantized IOI)
        onsets = sorted([n['start'] for n in notes])
        iois = np.diff(onsets)
        # Quantize to 100ms bins
        ioi_bins = (iois / 0.1).astype(int)
        ioi_counts = Counter(ioi_bins)
        ioi_probs = np.array(list(ioi_counts.values())) / len(ioi_bins)
        rhythm_entropy = -np.sum(ioi_probs * np.log2(ioi_probs + 1e-10))

        # Interval diversity
        sorted_notes = sorted(notes, key=lambda x: x['start'])
        intervals = []
        for i in range(len(sorted_notes) - 1):
            interval = sorted_notes[i+1]['pitch'] - sorted_notes[i]['pitch']
            intervals.append(interval)

        unique_intervals = len(set(intervals))
        interval_diversity = unique_intervals / len(intervals) if intervals else 0.0

        return {
            'pitch_entropy': pitch_entropy,
            'rhythm_entropy': rhythm_entropy,
            'interval_diversity': interval_diversity
        }

    def chord_tone_usage(
        self,
        notes: List[Dict],
        chord_progression: List[Dict]
    ) -> float:
        """
        코드 톤 사용 비율

        재즈에서 중요: 코드 톤 (root, 3rd, 5th, 7th)을 얼마나 사용하는가?

        Returns:
            코드 톤 사용 비율 (0-1)
        """
        if not notes or not chord_progression:
            return 0.0

        chord_tone_duration = 0.0
        total_duration = 0.0

        for note in notes:
            note_duration = note['end'] - note['start']
            total_duration += note_duration

            # 현재 코드 찾기
            current_chord = None
            for chord in chord_progression:
                if chord['start'] <= note['start'] < chord['end']:
                    current_chord = chord
                    break

            if current_chord is None:
                continue

            # 코드 톤 정의 (root, 3rd, 5th, 7th)
            root = current_chord.get('root', 0)
            chord_type = current_chord.get('type', 'maj7')

            # 코드 톤 계산
            if chord_type == 'maj7':
                chord_tones = [root, root + 4, root + 7, root + 11]
            elif chord_type == '7':
                chord_tones = [root, root + 4, root + 7, root + 10]
            elif chord_type == 'm7':
                chord_tones = [root, root + 3, root + 7, root + 10]
            elif chord_type == 'm7b5':
                chord_tones = [root, root + 3, root + 6, root + 10]
            else:
                chord_tones = [root]

            # 노트가 코드 톤인지 확인
            pitch_class = note['pitch'] % 12
            if pitch_class in [t % 12 for t in chord_tones]:
                chord_tone_duration += note_duration

        usage = chord_tone_duration / total_duration if total_duration > 0 else 0.0

        return usage

    def evaluate_generation(
        self,
        notes: List[Dict],
        chord_progression: List[Dict],
        expected_swing: float = 2.0
    ) -> Dict[str, float]:
        """
        종합 평가

        Returns:
            메트릭 딕셔너리
        """
        # Harmonic consistency
        harm_cons = self.harmonic_consistency(notes, chord_progression)

        # Swing ratio
        swing = self.swing_ratio(notes, expected_swing)

        # Style diversity
        diversity = self.style_diversity(notes)

        # Chord tone usage
        chord_tone_use = self.chord_tone_usage(notes, chord_progression)

        # Overall score (weighted average)
        overall = (
            harm_cons * 0.4 +
            swing * 0.2 +
            diversity['pitch_entropy'] / 4.0 * 0.2 +  # Normalize to 0-1
            chord_tone_use * 0.2
        )

        return {
            'harmonic_consistency': harm_cons,
            'swing_ratio_score': swing,
            'pitch_entropy': diversity['pitch_entropy'],
            'rhythm_entropy': diversity['rhythm_entropy'],
            'interval_diversity': diversity['interval_diversity'],
            'chord_tone_usage': chord_tone_use,
            'overall_score': overall
        }


# ===== 사용 예시 =====

if __name__ == "__main__":
    print("📊 Jazz Metrics 테스트\n")

    # 테스트 노트 (C major scale)
    notes = []
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72] * 4):
        notes.append({
            'pitch': pitch,
            'start': i * 0.5,
            'end': (i + 1) * 0.5,
            'velocity': 80
        })

    # 테스트 코드 진행 (II-V-I in C)
    chord_progression = [
        {'type': 'm7', 'root': 2, 'start': 0.0, 'end': 4.0},  # Dm7
        {'type': '7', 'root': 7, 'start': 4.0, 'end': 8.0},   # G7
        {'type': 'maj7', 'root': 0, 'start': 8.0, 'end': 16.0}, # Cmaj7
    ]

    # 평가
    metrics = JazzMetrics()

    results = metrics.evaluate_generation(
        notes=notes,
        chord_progression=chord_progression,
        expected_swing=2.0
    )

    print("평가 결과:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    print("\n✅ Jazz Metrics 테스트 완료!")
