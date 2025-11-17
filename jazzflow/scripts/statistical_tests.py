"""
Statistical Testing - Real Science

Bootstrap confidence intervals, significance tests.
No p-hacking, just honest statistics.
"""

import numpy as np
import json
from pathlib import Path
from scipy import stats


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """
    Bootstrap confidence interval

    Args:
        data: array of values
        n_bootstrap: number of bootstrap samples
        ci: confidence level (0.95 = 95%)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = 1 - ci
    lower = np.percentile(means, alpha/2 * 100)
    upper = np.percentile(means, (1 - alpha/2) * 100)

    return np.mean(data), lower, upper


def wilcoxon_test(data1, data2):
    """
    Wilcoxon signed-rank test (paired, non-parametric)

    Use this when comparing two models on the same test set.

    Returns:
        statistic, p_value, interpretation
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    if len(data1) != len(data2):
        raise ValueError("Data must be paired (same length)")

    # Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')

    # Interpretation
    if p_value < 0.001:
        sig = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig = "very significant (p < 0.01)"
    elif p_value < 0.05:
        sig = "significant (p < 0.05)"
    else:
        sig = "not significant (p >= 0.05)"

    return statistic, p_value, sig


def mann_whitney_test(data1, data2):
    """
    Mann-Whitney U test (unpaired, non-parametric)

    Use this when comparing two independent samples.
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    # Interpretation
    if p_value < 0.001:
        sig = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig = "very significant (p < 0.01)"
    elif p_value < 0.05:
        sig = "significant (p < 0.05)"
    else:
        sig = "not significant (p >= 0.05)"

    return statistic, p_value, sig


def compute_effect_size(data1, data2):
    """
    Cohen's d effect size

    Interpretation:
    - |d| < 0.2: small
    - |d| < 0.5: medium
    - |d| >= 0.8: large
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

    # Pooled standard deviation
    n1, n2 = len(data1), len(data2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Cohen's d
    d = (mean1 - mean2) / pooled_std

    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "small"
    elif abs_d < 0.5:
        interp = "medium"
    elif abs_d < 0.8:
        interp = "large"
    else:
        interp = "very large"

    return d, interp


def analyze_experiment_results(exp_dir):
    """
    Full statistical analysis of experiment results

    Reads eval.log files and computes:
    - Confidence intervals
    - Significance tests
    - Effect sizes
    """
    exp_dir = Path(exp_dir)

    print("=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    print()

    # Load results
    models = {}

    for model_dir in exp_dir.iterdir():
        if not model_dir.is_dir():
            continue

        eval_log = model_dir / "eval.log"
        if not eval_log.exists():
            print(f"⚠️  No eval.log for {model_dir.name}")
            continue

        # Parse eval.log
        with open(eval_log) as f:
            content = f.read()

        # Extract metrics (simple parsing)
        perplexity = None
        top1_acc = None
        top5_acc = None

        for line in content.split('\n'):
            if 'Perplexity:' in line:
                perplexity = float(line.split(':')[1].strip())
            elif 'Top-1 Accuracy:' in line:
                top1_acc = float(line.split(':')[1].strip().rstrip('%'))
            elif 'Top-5 Accuracy:' in line:
                top5_acc = float(line.split(':')[1].strip().rstrip('%'))

        if perplexity is not None:
            models[model_dir.name] = {
                'perplexity': perplexity,
                'top1_accuracy': top1_acc,
                'top5_accuracy': top5_acc
            }

    if len(models) < 2:
        print("⚠️  Need at least 2 models for comparison")
        print(f"   Found: {list(models.keys())}")
        return

    print(f"Found {len(models)} models:")
    for name, metrics in models.items():
        print(f"  - {name}")
        print(f"    Perplexity: {metrics['perplexity']:.2f}")
        print(f"    Top-1 Acc: {metrics['top1_accuracy']:.2f}%")
        print(f"    Top-5 Acc: {metrics['top5_accuracy']:.2f}%")
    print()

    # Compare JazzFlow vs baseline
    if 'jazzflow' in models and 'lstm_baseline' in models:
        print("=" * 80)
        print("JAZZFLOW vs LSTM BASELINE")
        print("=" * 80)
        print()

        jf = models['jazzflow']
        bl = models['lstm_baseline']

        # Perplexity improvement
        ppl_improvement = (bl['perplexity'] - jf['perplexity']) / bl['perplexity'] * 100
        print(f"Perplexity Improvement: {ppl_improvement:.1f}%")
        print(f"  JazzFlow: {jf['perplexity']:.2f}")
        print(f"  Baseline: {bl['perplexity']:.2f}")
        print()

        # Top-1 accuracy improvement
        acc1_improvement = jf['top1_accuracy'] - bl['top1_accuracy']
        print(f"Top-1 Accuracy Improvement: +{acc1_improvement:.2f} percentage points")
        print(f"  JazzFlow: {jf['top1_accuracy']:.2f}%")
        print(f"  Baseline: {bl['top1_accuracy']:.2f}%")
        print()

        # Top-5 accuracy improvement
        acc5_improvement = jf['top5_accuracy'] - bl['top5_accuracy']
        print(f"Top-5 Accuracy Improvement: +{acc5_improvement:.2f} percentage points")
        print(f"  JazzFlow: {jf['top5_accuracy']:.2f}%")
        print(f"  Baseline: {bl['top5_accuracy']:.2f}%")
        print()

        # Interpretation
        print("Interpretation:")
        if ppl_improvement > 10:
            print("  ✅ Strong improvement in perplexity (>10%)")
        elif ppl_improvement > 5:
            print("  ✓ Moderate improvement in perplexity (5-10%)")
        elif ppl_improvement > 0:
            print("  ~ Slight improvement in perplexity (<5%)")
        else:
            print("  ⚠️  No improvement or regression")

        if acc1_improvement > 5:
            print("  ✅ Strong improvement in top-1 accuracy (>5pp)")
        elif acc1_improvement > 2:
            print("  ✓ Moderate improvement in top-1 accuracy (2-5pp)")
        elif acc1_improvement > 0:
            print("  ~ Slight improvement in top-1 accuracy (<2pp)")
        else:
            print("  ⚠️  No improvement or regression")

        print()

    # Statistical significance (if we had per-sample data)
    print("=" * 80)
    print("NOTES ON STATISTICAL SIGNIFICANCE")
    print("=" * 80)
    print()
    print("For proper significance testing, you need:")
    print("  1. Per-sample losses (not just aggregate perplexity)")
    print("  2. Multiple runs with different seeds")
    print("  3. Wilcoxon signed-rank test or paired t-test")
    print()
    print("Current limitations:")
    print("  - Only aggregate metrics available")
    print("  - Single run per model")
    print()
    print("Recommendation:")
    print("  - Report results with error bars from multiple runs")
    print("  - Or clearly state 'single run, no significance test'")
    print()

    # Save summary
    summary = {
        'models': models,
        'comparison': {
            'perplexity_improvement_pct': ppl_improvement if 'jazzflow' in models and 'lstm_baseline' in models else None,
            'top1_improvement_pp': acc1_improvement if 'jazzflow' in models and 'lstm_baseline' in models else None,
            'top5_improvement_pp': acc5_improvement if 'jazzflow' in models and 'lstm_baseline' in models else None
        }
    }

    output_file = exp_dir / "statistical_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {output_file}")
    print()


def demo_statistical_tests():
    """Demo of statistical testing"""
    print("=" * 80)
    print("DEMO: Statistical Testing")
    print("=" * 80)
    print()

    # Simulated data: JazzFlow vs Baseline (per-sample losses)
    np.random.seed(42)

    # JazzFlow: lower loss (better)
    jazzflow_losses = np.random.gamma(shape=2, scale=1.5, size=100)

    # Baseline: higher loss (worse)
    baseline_losses = np.random.gamma(shape=2, scale=1.7, size=100)

    print("1. Bootstrap Confidence Intervals")
    print("-" * 80)

    jf_mean, jf_lower, jf_upper = bootstrap_ci(jazzflow_losses)
    print(f"JazzFlow: {jf_mean:.3f} [95% CI: {jf_lower:.3f}, {jf_upper:.3f}]")

    bl_mean, bl_lower, bl_upper = bootstrap_ci(baseline_losses)
    print(f"Baseline: {bl_mean:.3f} [95% CI: {bl_lower:.3f}, {bl_upper:.3f}]")
    print()

    print("2. Wilcoxon Signed-Rank Test (Paired)")
    print("-" * 80)

    stat, p_val, interp = wilcoxon_test(jazzflow_losses, baseline_losses)
    print(f"Statistic: {stat:.2f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Result: {interp}")
    print()

    print("3. Effect Size (Cohen's d)")
    print("-" * 80)

    d, effect = compute_effect_size(jazzflow_losses, baseline_losses)
    print(f"Cohen's d: {d:.3f}")
    print(f"Effect size: {effect}")
    print()

    print("Conclusion:")
    if p_val < 0.05 and abs(d) > 0.5:
        print("  ✅ JazzFlow is statistically significantly better")
        print("  ✅ Effect size is meaningful")
    elif p_val < 0.05:
        print("  ✓ Statistically significant but small effect")
    else:
        print("  ⚠️  Not statistically significant")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, help='Experiment directory')
    parser.add_argument('--demo', action='store_true', help='Run demo')

    args = parser.parse_args()

    if args.demo:
        demo_statistical_tests()
    elif args.exp_dir:
        analyze_experiment_results(args.exp_dir)
    else:
        print("Usage:")
        print("  python statistical_tests.py --exp_dir ./experiments/20250101_120000")
        print("  python statistical_tests.py --demo")
