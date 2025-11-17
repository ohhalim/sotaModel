"""
MIDI Data Processing - Minimal & Functional

No overcomplicated tokenization. Just what works.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pretty_midi
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class SimpleMIDITokenizer:
    """
    Simple MIDI tokenizer - no fancy REMI/CP

    Tokens:
    - Note On (0-127): pitch
    - Note Off (128-255): pitch + 128
    - Time Shift (256-355): 100 bins (0-10 sec)
    - Chord (356-415): 60 common chords
    - Special (416-419): PAD, BOS, EOS, UNK

    Total vocab: 420
    """

    def __init__(self):
        self.vocab_size = 420

        # Special tokens
        self.PAD = 416
        self.BOS = 417
        self.EOS = 418
        self.UNK = 419

        # Time shift resolution
        self.time_shift_bins = 100
        self.max_time_shift = 10.0  # seconds

    def encode(self, midi_path: str) -> List[int]:
        """MIDI file → token sequence"""
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except:
            return [self.BOS, self.EOS]

        tokens = [self.BOS]

        # Get all note events
        events = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                events.append(('note_on', note.start, note.pitch))
                events.append(('note_off', note.end, note.pitch))

        # Sort by time
        events.sort(key=lambda x: x[1])

        # Convert to tokens
        current_time = 0.0
        for event_type, time, pitch in events:
            # Time shift
            time_diff = time - current_time
            if time_diff > 0:
                time_shift_id = int((time_diff / self.max_time_shift) * self.time_shift_bins)
                time_shift_id = min(time_shift_id, self.time_shift_bins - 1)
                tokens.append(256 + time_shift_id)
                current_time = time

            # Note event
            if event_type == 'note_on':
                tokens.append(pitch)  # 0-127
            elif event_type == 'note_off':
                tokens.append(128 + pitch)  # 128-255

        tokens.append(self.EOS)

        return tokens

    def decode(self, tokens: List[int], output_path: str, tempo: int = 120):
        """Token sequence → MIDI file"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)

        current_time = 0.0
        active_notes = {}  # pitch -> start_time

        for token in tokens:
            # Skip special tokens
            if token in [self.PAD, self.BOS, self.EOS, self.UNK]:
                continue

            # Note On (0-127)
            if token < 128:
                pitch = token
                active_notes[pitch] = current_time

            # Note Off (128-255)
            elif token < 256:
                pitch = token - 128
                if pitch in active_notes:
                    start = active_notes[pitch]
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=pitch,
                        start=start,
                        end=current_time
                    )
                    piano.notes.append(note)
                    del active_notes[pitch]

            # Time Shift (256-355)
            elif token < 356:
                time_shift_id = token - 256
                time_shift = (time_shift_id / self.time_shift_bins) * self.max_time_shift
                current_time += time_shift

        # Close remaining notes
        for pitch, start in active_notes.items():
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start,
                end=current_time + 0.5
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        midi.write(output_path)


class MIDIDataset(Dataset):
    """
    Simple MIDI Dataset

    No complicated caching, just load and tokenize
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 512,
        max_files: int = None
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len

        # Find MIDI files
        self.midi_files = list(self.data_dir.glob("**/*.mid")) + \
                         list(self.data_dir.glob("**/*.midi"))

        if max_files:
            self.midi_files = self.midi_files[:max_files]

        print(f"Found {len(self.midi_files)} MIDI files")

        # Tokenizer
        self.tokenizer = SimpleMIDITokenizer()

        # Pre-tokenize all files
        self.samples = []
        print("Tokenizing MIDI files...")
        for midi_file in tqdm(self.midi_files):
            tokens = self.tokenizer.encode(str(midi_file))

            # Create sliding windows
            for i in range(0, len(tokens) - seq_len - 1, seq_len // 2):
                sample = tokens[i:i + seq_len + 1]
                if len(sample) == seq_len + 1:
                    self.samples.append(sample)

        print(f"Created {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        # Input and target
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        # Dummy chord IDs (for now)
        # TODO: Real chord recognition
        chord_ids = torch.zeros(self.seq_len, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'chord_ids': chord_ids
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 16,
    seq_len: int = 512,
    max_files: int = None,
    num_workers: int = 0
):
    """Create DataLoader"""
    dataset = MIDIDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        max_files=max_files
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


# ===== Quick Test =====

if __name__ == "__main__":
    print("MIDI Data Test\n")

    # Create test MIDI
    test_dir = Path("/tmp/test_midi")
    test_dir.mkdir(exist_ok=True)

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)

    # C major scale
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(str(test_dir / "test.mid"))

    # Test tokenizer
    tokenizer = SimpleMIDITokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    tokens = tokenizer.encode(str(test_dir / "test.mid"))
    print(f"Tokens: {len(tokens)}")
    print(f"Sample: {tokens[:20]}")

    # Decode
    tokenizer.decode(tokens, str(test_dir / "decoded.mid"))
    print("Decoded MIDI saved")

    # Test dataset
    dataset = MIDIDataset(
        data_dir=str(test_dir),
        seq_len=64
    )
    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input shape: {sample['input_ids'].shape}")
        print(f"Target shape: {sample['target_ids'].shape}")

    # Test dataloader
    dataloader = create_dataloader(
        data_dir=str(test_dir),
        batch_size=2,
        seq_len=64
    )
    print(f"Dataloader batches: {len(dataloader)}")

    print("\n✅ Data test passed!")
