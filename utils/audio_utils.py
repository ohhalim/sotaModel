"""
오디오 처리 유틸리티
- 오디오 로드/저장
- 스펙트로그램
- 오디오 증강
"""

import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, List


def load_audio(
    audio_path: str,
    sr: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    오디오 파일 로드

    Args:
        audio_path: 오디오 파일 경로
        sr: 샘플링 레이트
        mono: 모노로 변환 여부
        duration: 로드할 길이 (초)
        offset: 시작 시간 (초)

    Returns:
        (audio, sample_rate)
    """
    audio, sample_rate = librosa.load(
        audio_path,
        sr=sr,
        mono=mono,
        duration=duration,
        offset=offset
    )
    return audio, sample_rate


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sr: int = 22050
):
    """
    오디오 저장

    Args:
        audio: 오디오 배열
        output_path: 출력 경로
        sr: 샘플링 레이트
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sr)
    print(f"오디오 저장됨: {output_path}")


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    멜 스펙트로그램 계산

    Returns:
        Mel spectrogram (n_mels, time)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    # dB 스케일로 변환
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def compute_mfcc(
    audio: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    MFCC 계산

    Returns:
        MFCC (n_mfcc, time)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc


def compute_chroma(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    """
    크로마 특징 계산 (코드 인식용)

    Returns:
        Chroma (12, time)
    """
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length
    )
    return chroma


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    n_steps: float
) -> np.ndarray:
    """
    피치 시프트 (데이터 증강)

    Args:
        n_steps: 반음 단위 (-12 ~ 12)
    """
    return librosa.effects.pitch_shift(
        y=audio,
        sr=sr,
        n_steps=n_steps
    )


def time_stretch(
    audio: np.ndarray,
    rate: float
) -> np.ndarray:
    """
    시간 늘이기/줄이기

    Args:
        rate: 배율 (0.5 = 2배 빠르게, 2.0 = 2배 느리게)
    """
    return librosa.effects.time_stretch(y=audio, rate=rate)


def add_noise(
    audio: np.ndarray,
    noise_factor: float = 0.005
) -> np.ndarray:
    """
    노이즈 추가 (데이터 증강)
    """
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0
) -> np.ndarray:
    """
    오디오 정규화
    """
    # RMS 정규화
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10 ** (target_db / 20)
    return audio * (target_rms / rms)


def split_audio(
    audio: np.ndarray,
    sr: int,
    segment_length: float = 10.0,
    overlap: float = 0.0
) -> List[np.ndarray]:
    """
    오디오를 세그먼트로 분할

    Args:
        segment_length: 세그먼트 길이 (초)
        overlap: 오버랩 (초)

    Returns:
        [segment1, segment2, ...]
    """
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)
    hop = segment_samples - overlap_samples

    segments = []
    for i in range(0, len(audio) - segment_samples + 1, hop):
        segment = audio[i:i + segment_samples]
        segments.append(segment)

    return segments


def extract_beats(
    audio: np.ndarray,
    sr: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    비트 추출

    Returns:
        (tempo, beat_frames)
    """
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    return tempo, beat_frames


def onset_detection(
    audio: np.ndarray,
    sr: int
) -> np.ndarray:
    """
    Onset 감지
    """
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


def audio_to_midi_notes(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = librosa.note_to_hz('C2'),
    fmax: float = librosa.note_to_hz('C7')
) -> List[dict]:
    """
    오디오에서 MIDI 노트 추정 (간단한 버전)

    Returns:
        [{pitch, start, end, confidence}, ...]
    """
    # F0 추정
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length
    )

    # MIDI 노트로 변환
    notes = []
    current_note = None

    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)

        if voiced and freq is not None and not np.isnan(freq):
            midi_note = librosa.hz_to_midi(freq)
            pitch = int(round(midi_note))

            if current_note is None:
                # 새 노트 시작
                current_note = {
                    'pitch': pitch,
                    'start': time,
                    'end': time,
                    'confidence': voiced_probs[i]
                }
            elif abs(pitch - current_note['pitch']) < 1:
                # 같은 노트 계속
                current_note['end'] = time
            else:
                # 다른 노트 시작
                notes.append(current_note)
                current_note = {
                    'pitch': pitch,
                    'start': time,
                    'end': time,
                    'confidence': voiced_probs[i]
                }
        else:
            # 무음
            if current_note is not None:
                notes.append(current_note)
                current_note = None

    # 마지막 노트
    if current_note is not None:
        notes.append(current_note)

    return notes


def visualize_waveform(
    audio: np.ndarray,
    sr: int,
    save_path: Optional[str] = None
):
    """
    파형 시각화
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def visualize_spectrogram(
    mel_spec: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    save_path: Optional[str] = None
):
    """
    스펙트로그램 시각화
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        mel_spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap='viridis'
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


# ===== 사용 예시 =====
if __name__ == "__main__":
    print("🎵 오디오 유틸리티 테스트")

    # 테스트용 사인파 생성
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # A4 (440Hz) 노트
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 저장
    save_audio(audio, "/tmp/test_audio.wav", sr=sr)

    # 스펙트로그램
    mel_spec = compute_mel_spectrogram(audio, sr=sr)
    print(f"멜 스펙트로그램 shape: {mel_spec.shape}")

    print("✅ 오디오 유틸리티 정상 작동")
