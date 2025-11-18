"""
Split MAESTRO dataset into train/val/test

Following official split from maestro-v3.0.0.json
"""

import json
import shutil
from pathlib import Path
import argparse


def split_maestro(data_dir):
    """Split MAESTRO based on official metadata"""

    data_dir = Path(data_dir)

    # Read metadata
    metadata_file = data_dir / "maestro-v3.0.0.json"

    if not metadata_file.exists():
        print(f"⚠️  Metadata not found: {metadata_file}")
        print("Using filename-based split instead...")
        split_by_hash(data_dir)
        return

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Create split directories
    splits = ['train', 'validation', 'test']
    for split in splits:
        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)

    # Copy files to splits
    stats = {'train': 0, 'validation': 0, 'test': 0}

    for entry in metadata:
        midi_filename = entry['midi_filename']
        split = entry['split']

        source = data_dir / midi_filename
        target = data_dir / split / Path(midi_filename).name

        if source.exists():
            if not target.exists():
                shutil.copy(source, target)
            stats[split] += 1

    print(f"\n📊 Split Statistics:")
    print(f"  Train: {stats['train']} files")
    print(f"  Validation: {stats['validation']} files")
    print(f"  Test: {stats['test']} files")
    print(f"  Total: {sum(stats.values())} files")


def split_by_hash(data_dir):
    """Fallback: split by hash (80/10/10)"""
    data_dir = Path(data_dir)

    # Find all MIDI files
    midi_files = list(data_dir.glob("**/*.midi")) + list(data_dir.glob("**/*.mid"))

    # Create split directories
    for split in ['train', 'validation', 'test']:
        (data_dir / split).mkdir(exist_ok=True)

    # Split files
    stats = {'train': 0, 'validation': 0, 'test': 0}

    for midi_file in midi_files:
        # Hash filename to determine split
        h = hash(midi_file.name) % 10

        if h < 8:
            split = 'train'
        elif h == 8:
            split = 'validation'
        else:
            split = 'test'

        target = data_dir / split / midi_file.name

        if not target.exists():
            shutil.copy(midi_file, target)

        stats[split] += 1

    print(f"\n📊 Split Statistics (hash-based):")
    print(f"  Train: {stats['train']} files")
    print(f"  Validation: {stats['validation']} files")
    print(f"  Test: {stats['test']} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    split_maestro(args.data_dir)
    print("\n✅ Dataset split complete!")
