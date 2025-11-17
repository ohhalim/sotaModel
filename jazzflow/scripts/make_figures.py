"""
Figure Generation for Paper

Creates publication-quality figures for the paper.
All figures saved to paper/figures/
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import argparse


# Paper-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if you have LaTeX
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})


def parse_train_log(log_file):
    """Parse training log for loss curves"""
    epochs = []
    train_losses = []
    val_losses = []

    with open(log_file) as f:
        for line in f:
            # Example: "Epoch 1/20 - Train Loss: 3.456, Val Loss: 3.234"
            if 'Epoch' in line and 'Train Loss' in line:
                try:
                    parts = line.split('-')[0]
                    epoch = int(parts.split('/')[0].split()[-1])

                    train_part = line.split('Train Loss:')[1].split(',')[0]
                    train_loss = float(train_part.strip())

                    if 'Val Loss' in line:
                        val_part = line.split('Val Loss:')[1].split(',')[0]
                        val_loss = float(val_part.strip())
                    else:
                        val_loss = None

                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    if val_loss is not None:
                        val_losses.append(val_loss)
                except:
                    continue

    return epochs, train_losses, val_losses


def figure1_training_curves(exp_dir, output_dir):
    """
    Figure 1: Training curves comparing JazzFlow vs baseline

    Shows convergence speed and final performance.
    """
    exp_dir = Path(exp_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # JazzFlow
    jf_log = exp_dir / 'jazzflow' / 'train.log'
    if jf_log.exists():
        epochs, train_loss, val_loss = parse_train_log(jf_log)
        ax1.plot(epochs, train_loss, label='JazzFlow (Train)', color='#2E86AB', linestyle='-')
        if val_loss:
            ax1.plot(epochs, val_loss, label='JazzFlow (Val)', color='#2E86AB', linestyle='--')

    # Baseline
    bl_log = exp_dir / 'lstm_baseline' / 'train.log'
    if bl_log.exists():
        epochs, train_loss, val_loss = parse_train_log(bl_log)
        ax1.plot(epochs, train_loss, label='LSTM (Train)', color='#A23B72', linestyle='-')
        if val_loss:
            ax1.plot(epochs, val_loss, label='LSTM (Val)', color='#A23B72', linestyle='--')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Perplexity comparison (bar chart)
    models = ['JazzFlow', 'LSTM']
    perplexities = []

    # Load from eval logs
    for model_name, model_dir in [('JazzFlow', 'jazzflow'), ('LSTM', 'lstm_baseline')]:
        eval_log = exp_dir / model_dir / 'eval.log'
        if eval_log.exists():
            with open(eval_log) as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'Perplexity:' in line:
                        ppl = float(line.split(':')[1].strip())
                        perplexities.append(ppl)
                        break

    if len(perplexities) == 2:
        bars = ax2.bar(models, perplexities, color=['#2E86AB', '#A23B72'], alpha=0.7)
        ax2.set_ylabel('Perplexity (lower is better)')
        ax2.set_title('Test Set Perplexity')
        ax2.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, ppl in zip(bars, perplexities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ppl:.1f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / 'training_curves.pdf'
    plt.savefig(output_file)
    plt.close()

    print(f"✅ Figure 1 saved: {output_file}")


def figure2_accuracy_comparison(exp_dir, output_dir):
    """
    Figure 2: Accuracy comparison (Top-1 and Top-5)

    Grouped bar chart.
    """
    exp_dir = Path(exp_dir)
    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Load accuracies
    models = ['JazzFlow', 'LSTM']
    top1_accs = []
    top5_accs = []

    for model_name, model_dir in [('JazzFlow', 'jazzflow'), ('LSTM', 'lstm_baseline')]:
        eval_log = exp_dir / model_dir / 'eval.log'
        if eval_log.exists():
            with open(eval_log) as f:
                content = f.read()
                top1, top5 = None, None
                for line in content.split('\n'):
                    if 'Top-1 Accuracy:' in line:
                        top1 = float(line.split(':')[1].strip().rstrip('%'))
                    elif 'Top-5 Accuracy:' in line:
                        top5 = float(line.split(':')[1].strip().rstrip('%'))
                top1_accs.append(top1 if top1 else 0)
                top5_accs.append(top5 if top5 else 0)

    if len(top1_accs) == 2 and len(top5_accs) == 2:
        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, top1_accs, width, label='Top-1', color='#2E86AB', alpha=0.7)
        bars2 = ax.bar(x + width/2, top5_accs, width, label='Top-5', color='#F18F01', alpha=0.7)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        output_file = output_dir / 'accuracy_comparison.pdf'
        plt.savefig(output_file)
        plt.close()

        print(f"✅ Figure 2 saved: {output_file}")


def figure3_piano_roll_example(midi_file, output_dir, max_notes=100):
    """
    Figure 3: Piano roll visualization of generated music

    Requires pretty_midi for parsing.
    """
    try:
        import pretty_midi
    except ImportError:
        print("⚠️  pretty_midi not installed, skipping piano roll")
        return

    output_dir = Path(output_dir)

    # Load MIDI
    if not Path(midi_file).exists():
        print(f"⚠️  MIDI file not found: {midi_file}")
        return

    midi = pretty_midi.PrettyMIDI(str(midi_file))

    # Extract notes from first instrument
    if len(midi.instruments) == 0:
        print("⚠️  No instruments in MIDI")
        return

    notes = midi.instruments[0].notes[:max_notes]

    if len(notes) == 0:
        print("⚠️  No notes in MIDI")
        return

    # Create piano roll visualization
    fig, ax = plt.subplots(figsize=(10, 4))

    for note in notes:
        # Rectangle for each note
        rect = plt.Rectangle(
            (note.start, note.pitch),
            note.end - note.start,
            1,
            facecolor='#2E86AB',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.7
        )
        ax.add_patch(rect)

    # Set limits
    if notes:
        min_pitch = min(note.pitch for note in notes)
        max_pitch = max(note.pitch for note in notes)
        max_time = max(note.end for note in notes)

        ax.set_xlim(0, max_time)
        ax.set_ylim(min_pitch - 1, max_pitch + 1)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MIDI Pitch')
    ax.set_title('Generated Music (Piano Roll)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'piano_roll_example.pdf'
    plt.savefig(output_file)
    plt.close()

    print(f"✅ Figure 3 saved: {output_file}")


def figure4_dataset_statistics(data_analysis_file, output_dir):
    """
    Figure 4: Dataset statistics

    Shows distribution of MAESTRO dataset.
    """
    output_dir = Path(output_dir)

    # Load dataset statistics
    if not Path(data_analysis_file).exists():
        print(f"⚠️  Dataset analysis file not found: {data_analysis_file}")
        return

    with open(data_analysis_file) as f:
        stats = json.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))

    # Duration distribution
    if 'durations' in stats:
        ax1.hist(stats['durations'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Count')
        ax1.set_title('Recording Duration Distribution')
        ax1.grid(True, alpha=0.3)

    # Note count distribution
    if 'note_counts' in stats:
        ax2.hist(stats['note_counts'], bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Notes')
        ax2.set_ylabel('Count')
        ax2.set_title('Note Count Distribution')
        ax2.grid(True, alpha=0.3)

    # Pitch distribution
    if 'pitch_distribution' in stats:
        pitches = list(range(128))
        counts = stats['pitch_distribution']
        ax3.bar(pitches, counts, color='#A23B72', alpha=0.7)
        ax3.set_xlabel('MIDI Pitch')
        ax3.set_ylabel('Count')
        ax3.set_title('Overall Pitch Distribution')
        ax3.grid(True, axis='y', alpha=0.3)

    # Tempo distribution
    if 'tempos' in stats:
        ax4.hist(stats['tempos'], bins=30, color='#6A994E', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Tempo (BPM)')
        ax4.set_ylabel('Count')
        ax4.set_title('Tempo Distribution')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'dataset_statistics.pdf'
    plt.savefig(output_file)
    plt.close()

    print(f"✅ Figure 4 saved: {output_file}")


def make_all_figures(exp_dir):
    """Generate all figures for the paper"""
    exp_dir = Path(exp_dir)
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING PAPER FIGURES")
    print("=" * 80)
    print()

    # Figure 1: Training curves
    print("Figure 1: Training Curves")
    figure1_training_curves(exp_dir, output_dir)
    print()

    # Figure 2: Accuracy comparison
    print("Figure 2: Accuracy Comparison")
    figure2_accuracy_comparison(exp_dir, output_dir)
    print()

    # Figure 3: Piano roll (if sample exists)
    print("Figure 3: Piano Roll Example")
    sample_midi = exp_dir / 'jazzflow' / 'sample.mid'
    if sample_midi.exists():
        figure3_piano_roll_example(sample_midi, output_dir)
    else:
        print(f"⚠️  Sample MIDI not found: {sample_midi}")
    print()

    # Figure 4: Dataset statistics (if available)
    print("Figure 4: Dataset Statistics")
    data_stats = Path('data/dataset_stats.json')
    if data_stats.exists():
        figure4_dataset_statistics(data_stats, output_dir)
    else:
        print(f"⚠️  Dataset stats not found: {data_stats}")
    print()

    print("=" * 80)
    print("✅ All figures generated!")
    print(f"   Saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Experiment directory')

    args = parser.parse_args()

    make_all_figures(args.exp_dir)
