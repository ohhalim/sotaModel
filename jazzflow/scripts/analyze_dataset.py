"""
Analyze MAESTRO dataset

Generate statistics and visualizations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pretty_midi
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from collections import Counter


def analyze_midi_file(midi_path):
    """Extract statistics from a single MIDI file"""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        stats = {
            'duration': midi.get_end_time(),
            'num_notes': 0,
            'pitch_range': [128, 0],
            'avg_velocity': 0,
            'tempo': midi.estimate_tempo()
        }

        all_pitches = []
        all_velocities = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            for note in instrument.notes:
                stats['num_notes'] += 1
                all_pitches.append(note.pitch)
                all_velocities.append(note.velocity)

                stats['pitch_range'][0] = min(stats['pitch_range'][0], note.pitch)
                stats['pitch_range'][1] = max(stats['pitch_range'][1], note.pitch)

        if all_velocities:
            stats['avg_velocity'] = np.mean(all_velocities)
            stats['pitches'] = all_pitches

        return stats

    except Exception as e:
        return None


def analyze_dataset(data_dir, max_files=None):
    """Analyze entire dataset"""
    data_dir = Path(data_dir)

    # Find MIDI files
    midi_files = list(data_dir.glob("**/*.midi")) + list(data_dir.glob("**/*.mid"))

    if max_files:
        midi_files = midi_files[:max_files]

    print(f"📊 Analyzing {len(midi_files)} MIDI files...")
    print("")

    # Collect statistics
    all_stats = []
    total_duration = 0
    total_notes = 0
    all_pitches = []
    all_tempos = []

    for midi_file in tqdm(midi_files, desc="Processing"):
        stats = analyze_midi_file(midi_file)

        if stats:
            all_stats.append(stats)
            total_duration += stats['duration']
            total_notes += stats['num_notes']

            if 'pitches' in stats:
                all_pitches.extend(stats['pitches'])

            all_tempos.append(stats['tempo'])

    # Summary statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\n📁 Files:")
    print(f"  Total files: {len(all_stats)}")
    print(f"  Failed to parse: {len(midi_files) - len(all_stats)}")

    print(f"\n⏱️  Duration:")
    print(f"  Total: {total_duration / 3600:.1f} hours")
    print(f"  Average per file: {total_duration / len(all_stats):.1f} seconds")
    print(f"  Min: {min(s['duration'] for s in all_stats):.1f}s")
    print(f"  Max: {max(s['duration'] for s in all_stats):.1f}s")

    print(f"\n🎵 Notes:")
    print(f"  Total notes: {total_notes:,}")
    print(f"  Average per file: {total_notes / len(all_stats):.0f}")
    print(f"  Density: {total_notes / total_duration:.2f} notes/second")

    print(f"\n🎹 Pitch:")
    if all_pitches:
        pitch_counter = Counter(all_pitches)
        most_common = pitch_counter.most_common(5)

        print(f"  Range: {min(all_pitches)} - {max(all_pitches)}")
        print(f"  Unique pitches: {len(set(all_pitches))}")
        print(f"  Most common:")
        for pitch, count in most_common:
            note_name = pretty_midi.note_number_to_name(pitch)
            print(f"    {note_name} ({pitch}): {count:,} times")

    print(f"\n🎼 Tempo:")
    if all_tempos:
        print(f"  Average: {np.mean(all_tempos):.1f} BPM")
        print(f"  Min: {min(all_tempos):.1f} BPM")
        print(f"  Max: {max(all_tempos):.1f} BPM")

    # Visualizations
    print(f"\n📈 Generating visualizations...")
    plot_statistics(all_stats, all_pitches, all_tempos)

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)

    return all_stats


def plot_statistics(all_stats, all_pitches, all_tempos):
    """Generate plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Duration distribution
    durations = [s['duration'] for s in all_stats]
    axes[0, 0].hist(durations, bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Duration (seconds)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Duration Distribution')
    axes[0, 0].axvline(np.mean(durations), color='r', linestyle='--', label=f'Mean: {np.mean(durations):.1f}s')
    axes[0, 0].legend()

    # Note count distribution
    note_counts = [s['num_notes'] for s in all_stats]
    axes[0, 1].hist(note_counts, bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Number of Notes')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Note Count Distribution')
    axes[0, 1].axvline(np.mean(note_counts), color='r', linestyle='--', label=f'Mean: {np.mean(note_counts):.0f}')
    axes[0, 1].legend()

    # Pitch distribution
    if all_pitches:
        axes[1, 0].hist(all_pitches, bins=88, edgecolor='black', range=(21, 109))
        axes[1, 0].set_xlabel('MIDI Pitch')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Pitch Distribution')

    # Tempo distribution
    if all_tempos:
        axes[1, 1].hist(all_tempos, bins=50, edgecolor='black')
        axes[1, 1].set_xlabel('Tempo (BPM)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Tempo Distribution')
        axes[1, 1].axvline(np.mean(all_tempos), color='r', linestyle='--', label=f'Mean: {np.mean(all_tempos):.1f} BPM')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150)
    print("  Saved: dataset_analysis.png")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/maestro-v3.0.0/train')
    parser.add_argument('--max_files', type=int, default=None)
    args = parser.parse_args()

    analyze_dataset(args.data_dir, args.max_files)
