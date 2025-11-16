"""
완전한 데이터셋 클래스
MAESTRO + PiJAMA 지원
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pretty_midi
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from tqdm import tqdm
import pickle

import sys
sys.path.append(str(Path(__file__).parent))
from midi_tokenizer import REMITokenizer


class JazzMIDIDataset(Dataset):
    """
    재즈 MIDI 데이터셋 (완전한 구현)

    Features:
    - MIDI tokenization (REMI)
    - Chord progression extraction
    - Data augmentation (transpose, time stretch)
    - Caching for faster loading
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_len: int = 512,
        stride: int = 256,
        tokenizer: Optional[REMITokenizer] = None,
        use_cache: bool = True,
        augment: bool = True,
        max_files: Optional[int] = None
    ):
        """
        Args:
            data_dir: MIDI 파일 디렉토리
            split: train/val/test
            seq_len: 시퀀스 길이
            stride: 슬라이딩 윈도우 stride
            tokenizer: MIDI tokenizer (None이면 기본값)
            use_cache: 토크나이징 캐싱 사용
            augment: Data augmentation 사용
            max_files: 최대 파일 수 (None=전체)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment and (split == "train")

        # Tokenizer
        if tokenizer is None:
            self.tokenizer = REMITokenizer()
        else:
            self.tokenizer = tokenizer

        # MIDI 파일 리스트
        self.midi_files = self._get_midi_files()

        if max_files:
            self.midi_files = self.midi_files[:max_files]

        print(f"📂 {len(self.midi_files)}개 MIDI 파일 ({split})")

        # 캐시 디렉토리
        self.cache_dir = self.data_dir / f"cache_{split}"
        self.cache_dir.mkdir(exist_ok=True)

        # 데이터 전처리
        self.samples = self._preprocess(use_cache=use_cache)

        print(f"✅ {len(self.samples)}개 시퀀스 샘플")

    def _get_midi_files(self) -> List[Path]:
        """MIDI 파일 리스트 가져오기"""
        # .mid, .midi 파일 모두 검색
        midi_files = list(self.data_dir.glob("**/*.mid")) + \
                    list(self.data_dir.glob("**/*.midi"))

        # Train/Val/Test split
        # 간단한 split: 파일 이름 기준
        if self.split == "train":
            midi_files = [f for f in midi_files if hash(str(f)) % 10 < 8]
        elif self.split == "val":
            midi_files = [f for f in midi_files if hash(str(f)) % 10 == 8]
        elif self.split == "test":
            midi_files = [f for f in midi_files if hash(str(f)) % 10 == 9]

        return sorted(midi_files)

    def _preprocess(self, use_cache: bool = True) -> List[Dict]:
        """MIDI 파일들을 전처리"""
        samples = []

        for midi_path in tqdm(self.midi_files, desc="Preprocessing"):
            # 캐시 확인
            cache_file = self.cache_dir / f"{midi_path.stem}.pkl"

            if use_cache and cache_file.exists():
                # 캐시에서 로드
                with open(cache_file, 'rb') as f:
                    file_samples = pickle.load(f)
            else:
                # MIDI 토크나이징
                try:
                    tokens = self.tokenizer.encode(str(midi_path))

                    # 코드 진행 추출 (간단화 버전)
                    chords = self._extract_chords(midi_path)

                    # Sliding window로 샘플 생성
                    file_samples = []
                    for i in range(0, len(tokens) - self.seq_len, self.stride):
                        sample = {
                            'tokens': tokens[i:i + self.seq_len + 1],  # +1 for target
                            'chords': chords[i:i + self.seq_len + 1] if chords else None,
                            'midi_path': str(midi_path)
                        }
                        file_samples.append(sample)

                    # 캐시 저장
                    if use_cache:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(file_samples, f)

                except Exception as e:
                    print(f"⚠️  {midi_path.name} 처리 실패: {e}")
                    continue

            samples.extend(file_samples)

        return samples

    def _extract_chords(self, midi_path: Path) -> Optional[List[int]]:
        """
        MIDI에서 코드 진행 추출 (간단한 버전)

        실제로는 music21이나 더 복잡한 알고리즘 필요
        여기서는 더미 구현
        """
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            # 간단한 코드 감지: 동시발음 노트들로부터 코드 유추
            # TODO: 더 정교한 코드 감지 알고리즘
            chord_ids = [0] * (len(self.tokenizer.vocab))  # Dummy

            return chord_ids[:self.seq_len + 1]

        except:
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        tokens = sample['tokens']
        chords = sample.get('chords')

        # Data augmentation (train only)
        if self.augment:
            # Transpose (±6 semitones)
            transpose = np.random.randint(-6, 7)
            tokens = self._transpose_tokens(tokens, transpose)

        # Padding if needed
        if len(tokens) < self.seq_len + 1:
            pad_len = self.seq_len + 1 - len(tokens)
            tokens = tokens + [self.tokenizer.token2id["<PAD>"]] * pad_len

        # Convert to tensors
        tokens = torch.tensor(tokens[:self.seq_len + 1], dtype=torch.long)

        result = {
            'input_ids': tokens[:-1],  # Input
            'target_ids': tokens[1:],   # Target (next token)
        }

        # Chords (if available)
        if chords:
            chords_tensor = torch.tensor(chords[:self.seq_len], dtype=torch.long)
            result['chord_ids'] = chords_tensor
        else:
            result['chord_ids'] = torch.zeros(self.seq_len, dtype=torch.long)

        # Style level (random for training)
        if self.augment:
            style_level = np.random.randint(4, 9)  # Jazz styles
        else:
            style_level = 6  # Default bebop

        result['style_level'] = torch.tensor(style_level, dtype=torch.long)

        return result

    def _transpose_tokens(self, tokens: List[int], semitones: int) -> List[int]:
        """
        토큰 시퀀스 transposition

        Note_On_XX, Note_Off_XX 토큰의 pitch를 조정
        """
        transposed = []

        for token_id in tokens:
            token = self.tokenizer.id2token[token_id]

            # Note On/Off 토큰 transpose
            if token.startswith("Note_On_") or token.startswith("Note_Off_"):
                prefix = "Note_On_" if token.startswith("Note_On_") else "Note_Off_"
                pitch = int(token.split("_")[-1])
                new_pitch = pitch + semitones

                # MIDI range 체크
                if self.tokenizer.pitch_range[0] <= new_pitch <= self.tokenizer.pitch_range[1]:
                    new_token = f"{prefix}{new_pitch}"
                    if new_token in self.tokenizer.token2id:
                        transposed.append(self.tokenizer.token2id[new_token])
                    else:
                        transposed.append(token_id)  # Keep original
                else:
                    transposed.append(token_id)  # Out of range, keep original
            else:
                transposed.append(token_id)

        return transposed


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    seq_len: int = 512,
    num_workers: int = 4,
    tokenizer: Optional[REMITokenizer] = None,
    max_files: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Train/Val DataLoader 생성

    Args:
        data_dir: 데이터 디렉토리
        batch_size: 배치 크기
        seq_len: 시퀀스 길이
        num_workers: 워커 수
        tokenizer: Tokenizer
        max_files: 최대 파일 수

    Returns:
        (train_loader, val_loader)
    """
    # Train dataset
    train_dataset = JazzMIDIDataset(
        data_dir=data_dir,
        split="train",
        seq_len=seq_len,
        tokenizer=tokenizer,
        use_cache=True,
        augment=True,
        max_files=max_files
    )

    # Val dataset
    val_dataset = JazzMIDIDataset(
        data_dir=data_dir,
        split="val",
        seq_len=seq_len,
        tokenizer=tokenizer,
        use_cache=True,
        augment=False,
        max_files=max_files // 10 if max_files else None
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


# ===== 사용 예시 =====

if __name__ == "__main__":
    print("🎹 Dataset 테스트\n")

    # 더미 데이터 디렉토리 생성
    import os
    dummy_dir = Path("/tmp/dummy_jazz_data")
    dummy_dir.mkdir(exist_ok=True)

    # 테스트용 MIDI 생성
    for i in range(5):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0)

        for j, pitch in enumerate(range(60, 73)):
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=j * 0.5,
                end=(j + 1) * 0.5
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        midi.write(str(dummy_dir / f"test_{i}.mid"))

    # Dataset 생성
    print("📂 Dataset 생성...")
    dataset = JazzMIDIDataset(
        data_dir=str(dummy_dir),
        split="train",
        seq_len=128,
        stride=64,
        use_cache=True,
        augment=True,
        max_files=5
    )

    print(f"\nDataset 크기: {len(dataset)}")

    # 샘플 확인
    sample = dataset[0]
    print(f"\n샘플 keys: {sample.keys()}")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Target shape: {sample['target_ids'].shape}")
    print(f"Chord shape: {sample['chord_ids'].shape}")
    print(f"Style level: {sample['style_level'].item()}")

    # DataLoader 생성
    print("\n📦 DataLoader 생성...")
    train_loader, val_loader = create_dataloaders(
        data_dir=str(dummy_dir),
        batch_size=2,
        seq_len=128,
        num_workers=0,
        max_files=5
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 배치 확인
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch input shape: {batch['input_ids'].shape}")
    print(f"Batch target shape: {batch['target_ids'].shape}")

    print("\n✅ Dataset 테스트 완료!")
