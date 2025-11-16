"""
평가 메트릭
- 음악 품질 평가
- 음악 이론 메트릭
- Perplexity, Accuracy 등
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import pretty_midi
from scipy.stats import entropy


def calculate_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Perplexity 계산 (언어 모델 평가)

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)

    Returns:
        perplexity
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    perplexity = torch.exp(loss).item()
    return perplexity


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    """
    Top-k 정확도

    Args:
        top_k: 1 (정확도), 5 (top-5 정확도)
    """
    _, pred = logits.topk(top_k, dim=-1)
    targets = targets.unsqueeze(-1).expand_as(pred)
    correct = pred.eq(targets).any(dim=-1).float()
    accuracy = correct.mean().item()
    return accuracy


# ===== 음악 이론 메트릭 =====

def calculate_pitch_class_histogram(notes: List[Dict]) -> np.ndarray:
    """
    피치 클래스 히스토그램 (0-11)

    Args:
        notes: [{pitch, start, end, ...}, ...]

    Returns:
        histogram (12,) - C, C#, D, ..., B
    """
    histogram = np.zeros(12)

    for note in notes:
        pitch_class = note['pitch'] % 12
        duration = note.get('duration', note['end'] - note['start'])
        histogram[pitch_class] += duration

    # 정규화
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()

    return histogram


def calculate_pitch_range(notes: List[Dict]) -> Dict[str, int]:
    """
    음역 통계
    """
    if not notes:
        return {'min': 0, 'max': 0, 'range': 0}

    pitches = [n['pitch'] for n in notes]
    return {
        'min': min(pitches),
        'max': max(pitches),
        'range': max(pitches) - min(pitches),
        'mean': np.mean(pitches),
    }


def calculate_note_density(notes: List[Dict], total_duration: float) -> float:
    """
    노트 밀도 (notes per second)
    """
    if total_duration == 0:
        return 0.0
    return len(notes) / total_duration


def calculate_rhythm_complexity(notes: List[Dict]) -> Dict[str, float]:
    """
    리듬 복잡도
    """
    if len(notes) < 2:
        return {'ioi_mean': 0, 'ioi_std': 0, 'rhythm_entropy': 0}

    # Inter-Onset Interval (IOI)
    onsets = sorted([n['start'] for n in notes])
    iois = np.diff(onsets)

    # IOI 히스토그램 (양자화)
    ioi_bins = np.histogram(iois, bins=20)[0]
    ioi_probs = ioi_bins / ioi_bins.sum() if ioi_bins.sum() > 0 else ioi_bins

    return {
        'ioi_mean': np.mean(iois),
        'ioi_std': np.std(iois),
        'rhythm_entropy': entropy(ioi_probs + 1e-10),  # 리듬 엔트로피
    }


def calculate_harmonic_entropy(pitch_class_hist: np.ndarray) -> float:
    """
    화성 엔트로피 (다양성 측정)
    """
    probs = pitch_class_hist + 1e-10
    return entropy(probs)


def detect_key(pitch_class_hist: np.ndarray) -> Tuple[str, float]:
    """
    조성 감지 (Krumhansl-Schmuckler 알고리즘)

    Returns:
        (key_name, correlation)
    """
    # Krumhansl-Kessler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # 모든 조성에 대해 상관계수 계산
    best_corr = -1
    best_key = "C major"

    for i in range(12):
        # Major
        shifted_major = np.roll(major_profile, i)
        corr = np.corrcoef(pitch_class_hist, shifted_major)[0, 1]
        if corr > best_corr:
            best_corr = corr
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            best_key = f"{key_names[i]} major"

        # Minor
        shifted_minor = np.roll(minor_profile, i)
        corr = np.corrcoef(pitch_class_hist, shifted_minor)[0, 1]
        if corr > best_corr:
            best_corr = corr
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            best_key = f"{key_names[i]} minor"

    return best_key, best_corr


def calculate_polyphony(notes: List[Dict], sample_rate: int = 100) -> Dict[str, float]:
    """
    폴리포니 통계
    """
    if not notes:
        return {'mean': 0, 'max': 0}

    # 시간별 동시발음 수 계산
    max_time = max(n['end'] for n in notes)
    time_steps = int(max_time * sample_rate)
    polyphony_over_time = np.zeros(time_steps)

    for note in notes:
        start_idx = int(note['start'] * sample_rate)
        end_idx = int(note['end'] * sample_rate)
        polyphony_over_time[start_idx:end_idx] += 1

    return {
        'mean': np.mean(polyphony_over_time),
        'max': int(np.max(polyphony_over_time)),
        'std': np.std(polyphony_over_time),
    }


def calculate_interval_distribution(notes: List[Dict]) -> Dict[int, int]:
    """
    음정 분포 (melodic intervals)
    """
    if len(notes) < 2:
        return {}

    # 시간순 정렬
    sorted_notes = sorted(notes, key=lambda x: x['start'])

    # 연속된 음정 차이 계산
    intervals = {}
    for i in range(len(sorted_notes) - 1):
        interval = sorted_notes[i+1]['pitch'] - sorted_notes[i]['pitch']
        intervals[interval] = intervals.get(interval, 0) + 1

    return intervals


# ===== 생성 음악 vs 실제 음악 비교 =====

def calculate_distribution_similarity(
    generated_hist: np.ndarray,
    reference_hist: np.ndarray,
    method: str = 'kl'
) -> float:
    """
    분포 유사도

    Args:
        method: 'kl' (KL-divergence), 'js' (JS-divergence), 'cosine'
    """
    # 정규화
    p = generated_hist / (generated_hist.sum() + 1e-10)
    q = reference_hist / (reference_hist.sum() + 1e-10)

    if method == 'kl':
        # KL Divergence (낮을수록 유사)
        return entropy(p + 1e-10, q + 1e-10)

    elif method == 'js':
        # Jensen-Shannon Divergence
        m = 0.5 * (p + q)
        js = 0.5 * entropy(p + 1e-10, m + 1e-10) + 0.5 * entropy(q + 1e-10, m + 1e-10)
        return js

    elif method == 'cosine':
        # Cosine similarity (높을수록 유사)
        dot = np.dot(p, q)
        norm = np.linalg.norm(p) * np.linalg.norm(q)
        return dot / (norm + 1e-10)

    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_music_generation(
    generated_notes: List[Dict],
    reference_notes: List[Dict],
    generated_duration: float,
    reference_duration: float
) -> Dict[str, float]:
    """
    생성 음악 종합 평가

    Returns:
        메트릭 딕셔너리
    """
    # Pitch class 히스토그램
    gen_pch = calculate_pitch_class_histogram(generated_notes)
    ref_pch = calculate_pitch_class_histogram(reference_notes)

    # 조성 감지
    gen_key, gen_key_corr = detect_key(gen_pch)
    ref_key, ref_key_corr = detect_key(ref_pch)

    # 음역
    gen_range = calculate_pitch_range(generated_notes)
    ref_range = calculate_pitch_range(reference_notes)

    # 리듬
    gen_rhythm = calculate_rhythm_complexity(generated_notes)
    ref_rhythm = calculate_rhythm_complexity(reference_notes)

    # 폴리포니
    gen_poly = calculate_polyphony(generated_notes)
    ref_poly = calculate_polyphony(reference_notes)

    # 유사도
    pch_similarity = calculate_distribution_similarity(gen_pch, ref_pch, method='cosine')

    return {
        # 기본 통계
        'num_notes': len(generated_notes),
        'duration': generated_duration,
        'note_density': calculate_note_density(generated_notes, generated_duration),

        # 음역
        'pitch_range': gen_range['range'],
        'pitch_mean': gen_range['mean'],

        # 조성
        'detected_key': gen_key,
        'key_strength': gen_key_corr,

        # 리듬
        'rhythm_entropy': gen_rhythm['rhythm_entropy'],
        'ioi_mean': gen_rhythm['ioi_mean'],
        'ioi_std': gen_rhythm['ioi_std'],

        # 화성
        'harmonic_entropy': calculate_harmonic_entropy(gen_pch),

        # 폴리포니
        'polyphony_mean': gen_poly['mean'],
        'polyphony_max': gen_poly['max'],

        # 유사도 (reference와 비교)
        'pitch_class_similarity': pch_similarity,
    }


# ===== 재즈 특화 메트릭 =====

def detect_jazz_chords(notes: List[Dict], window_size: float = 0.5) -> List[str]:
    """
    재즈 코드 감지 (간단한 버전)

    Args:
        window_size: 시간 윈도우 (초)

    Returns:
        ['Cmaj7', 'Dm7', 'G7', ...]
    """
    # 이것은 매우 간단한 버전입니다.
    # 실제로는 music21이나 더 복잡한 알고리즘이 필요합니다.

    chords = []
    max_time = max(n['end'] for n in notes) if notes else 0

    for t in np.arange(0, max_time, window_size):
        # 현재 시간 윈도우의 노트들
        active_notes = [
            n for n in notes
            if n['start'] <= t < n['end']
        ]

        if len(active_notes) >= 3:
            # 피치 클래스 추출
            pitch_classes = sorted(set(n['pitch'] % 12 for n in active_notes))

            # 간단한 코드 감지 (매우 기초적)
            # 실제로는 더 복잡한 로직 필요
            chords.append(f"Chord_{len(pitch_classes)}_notes")

    return chords


def calculate_swing_ratio(notes: List[Dict]) -> float:
    """
    스윙 비율 계산 (재즈 특화)

    Returns:
        swing_ratio (1.0 = straight, 2.0 = triplet swing)
    """
    # 8분음표 쌍의 비율 계산
    # 실제 구현은 더 복잡합니다
    onsets = sorted([n['start'] for n in notes])

    if len(onsets) < 4:
        return 1.0

    # 간단한 근사
    iois = np.diff(onsets)
    pairs = []

    for i in range(0, len(iois) - 1, 2):
        if i + 1 < len(iois):
            ratio = iois[i] / (iois[i+1] + 1e-10)
            if 0.5 < ratio < 4.0:  # 유효한 범위
                pairs.append(ratio)

    if pairs:
        return np.median(pairs)
    else:
        return 1.0


# ===== 사용 예시 =====
if __name__ == "__main__":
    print("📊 평가 유틸리티 테스트")

    # 테스트 노트
    notes = [
        {'pitch': 60, 'start': 0.0, 'end': 0.5},  # C
        {'pitch': 64, 'start': 0.5, 'end': 1.0},  # E
        {'pitch': 67, 'start': 1.0, 'end': 1.5},  # G
        {'pitch': 72, 'start': 1.5, 'end': 2.0},  # C (octave)
    ]

    # Pitch class 히스토그램
    pch = calculate_pitch_class_histogram(notes)
    print(f"Pitch class histogram: {pch}")

    # 조성 감지
    key, corr = detect_key(pch)
    print(f"감지된 조성: {key} (상관계수: {corr:.3f})")

    # 음역
    pitch_range = calculate_pitch_range(notes)
    print(f"음역: {pitch_range}")

    # 리듬
    rhythm = calculate_rhythm_complexity(notes)
    print(f"리듬 복잡도: {rhythm}")

    print("✅ 평가 유틸리티 정상 작동")
