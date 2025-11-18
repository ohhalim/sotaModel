"""
Compare models and generate comparison plots/tables

Usage:
    python scripts/compare_models.py --exp_dir ./experiments/20251117_120000 --models jazzflow lstm_baseline
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re


def parse_log_file(log_path):
    """Extract metrics from training log"""
    if not Path(log_path).exists():
        return None

    with open(log_path) as f:
        lines = f.readlines()

    metrics = {
        'train_losses': [],
        'train_ppls': [],
        'epochs': []
    }

    for line in lines:
        # Parse: "Epoch 1/20 (123.4s)"
        #        "  Train Loss: 6.1234 | PPL: 456.78"

        if 'Epoch' in line and '/' in line:
            match = re.search(r'Epoch (\d+)/\d+', line)
            if match:
                epoch = int(match.group(1))

        if 'Train Loss:' in line:
            match = re.search(r'Train Loss: ([\d.]+).*PPL: ([\d.]+)', line)
            if match:
                loss = float(match.group(1))
                ppl = float(match.group(2))

                metrics['epochs'].append(epoch)
                metrics['train_losses'].append(loss)
                metrics['train_ppls'].append(ppl)

    return metrics if metrics['epochs'] else None


def parse_eval_log(log_path):
    """Extract evaluation metrics"""
    if not Path(log_path).exists():
        return None

    with open(log_path) as f:
        content = f.read()

    metrics = {}

    # Parse perplexity
    match = re.search(r'Perplexity: ([\d.]+)', content)
    if match:
        metrics['perplexity'] = float(match.group(1))

    # Parse top-1 accuracy
    match = re.search(r'Top-1 Accuracy: ([\d.]+)%', content)
    if match:
        metrics['top1_accuracy'] = float(match.group(1)) / 100

    # Parse top-5 accuracy
    match = re.search(r'Top-5 Accuracy: ([\d.]+)%', content)
    if match:
        metrics['top5_accuracy'] = float(match.group(1)) / 100

    return metrics if metrics else None


def load_model_results(exp_dir, model_name):
    """Load all results for a model"""
    model_dir = Path(exp_dir) / model_name

    results = {
        'name': model_name,
        'train': None,
        'eval': None
    }

    # Training results
    train_log = model_dir / 'train.log'
    if train_log.exists():
        results['train'] = parse_log_file(train_log)

    # Evaluation results
    eval_log = model_dir / 'eval.log'
    if eval_log.exists():
        results['eval'] = parse_eval_log(eval_log)

    return results


def plot_training_curves(all_results, output_path):
    """Plot training loss/perplexity curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for i, results in enumerate(all_results):
        if not results['train']:
            continue

        train = results['train']
        color = colors[i % len(colors)]
        label = results['name'].replace('_', ' ').title()

        # Loss curve
        ax1.plot(train['epochs'], train['train_losses'],
                label=label, color=color, linewidth=2, marker='o', markersize=4)

        # Perplexity curve
        ax2.plot(train['epochs'], train['train_ppls'],
                label=label, color=color, linewidth=2, marker='o', markersize=4)

    # Styling
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Perplexity Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training curves: {output_path}")

    plt.close()


def create_comparison_table(all_results, output_path):
    """Create comparison table"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<25} {'Perplexity':<15} {'Top-1 Acc':<15} {'Top-5 Acc':<15}")
    print("-" * 80)

    # Table data
    table_data = []

    for results in all_results:
        name = results['name'].replace('_', ' ').title()
        eval_metrics = results.get('eval', {})

        ppl = eval_metrics.get('perplexity', 'N/A')
        top1 = eval_metrics.get('top1_accuracy', 'N/A')
        top5 = eval_metrics.get('top5_accuracy', 'N/A')

        # Format
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else ppl
        top1_str = f"{top1*100:.2f}%" if isinstance(top1, float) else top1
        top5_str = f"{top5*100:.2f}%" if isinstance(top5, float) else top5

        print(f"{name:<25} {ppl_str:<15} {top1_str:<15} {top5_str:<15}")

        table_data.append({
            'model': name,
            'perplexity': ppl,
            'top1_accuracy': top1,
            'top5_accuracy': top5
        })

    print("=" * 80)

    # Save as LaTeX table
    latex_path = Path(output_path).parent / 'comparison_table.tex'
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\hline\n")
        f.write("Model & Perplexity $\\downarrow$ & Top-1 Acc. $\\uparrow$ & Top-5 Acc. $\\uparrow$ \\\\\n")
        f.write("\\hline\n")

        for row in table_data:
            name = row['model']
            ppl = f"{row['perplexity']:.2f}" if isinstance(row['perplexity'], float) else "N/A"
            top1 = f"{row['top1_accuracy']*100:.2f}\\%" if isinstance(row['top1_accuracy'], float) else "N/A"
            top5 = f"{row['top5_accuracy']*100:.2f}\\%" if isinstance(row['top5_accuracy'], float) else "N/A"

            # Bold best values
            if isinstance(row['perplexity'], float):
                all_ppls = [r['perplexity'] for r in table_data if isinstance(r['perplexity'], float)]
                if row['perplexity'] == min(all_ppls):
                    ppl = "\\textbf{" + ppl + "}"

            f.write(f"{name} & {ppl} & {top1} & {top5} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comparison of models on MAESTRO test set.}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\end{table}\n")

    print(f"\n✅ Saved LaTeX table: {latex_path}")

    # Statistical significance
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    print("\nNote: For statistical significance, run:")
    print("  python scripts/statistical_tests.py --exp_dir <exp_dir>")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--output', type=str, default='comparison.png')
    args = parser.parse_args()

    print(f"\n📊 Comparing Models")
    print(f"Experiment: {args.exp_dir}")
    print(f"Models: {', '.join(args.models)}\n")

    # Load results
    all_results = []
    for model_name in args.models:
        results = load_model_results(args.exp_dir, model_name)
        all_results.append(results)
        print(f"✅ Loaded: {model_name}")

    # Generate plots
    print(f"\n📈 Generating plots...")
    plot_training_curves(all_results, args.output)

    # Generate table
    print(f"\n📋 Generating comparison table...")
    create_comparison_table(all_results, args.output)

    print(f"\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
