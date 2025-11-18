# JazzFlow Experimental Protocol

**Paper Title**: "JazzFlow: Chord-Conditioned Jazz Piano Generation with Hybrid Transformer-LSTM Architecture"

**Target Venue**: ICML 2025 Workshop on AI and Music

---

## 1. Research Questions

**RQ1**: Does chord conditioning improve jazz generation quality?
- Hypothesis: JazzFlow (with chords) > Baseline (without chords)
- Metric: Harmonic consistency, perplexity

**RQ2**: Is the hybrid architecture effective?
- Hypothesis: Transformer+LSTM > Pure Transformer
- Metric: Generation quality, training efficiency

**RQ3**: Can we generate coherent jazz improvisations?
- Hypothesis: Generated music is indistinguishable from human
- Metric: Human evaluation (blind test)

---

## 2. Datasets

### Training Data

**MAESTRO v3.0.0**
- Size: 1,282 performances (~200 hours)
- Genre: Classical piano (for pre-training)
- Split: 80% train, 10% val, 10% test
- Download: https://magenta.tensorflow.org/datasets/maestro

**Why MAESTRO?**
- High quality recordings
- Aligned MIDI + audio
- Standard benchmark
- Publicly available

### Test Data (Optional)

**JazzMIDI** (if available)
- Real jazz performances
- For domain-specific evaluation
- If not available: use MAESTRO test split

---

## 3. Models

### Model 1: JazzFlow (Ours)
```python
JazzFlowModel(
    vocab_size=420,
    num_chords=60,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    lstm_hidden=512,
    lstm_layers=2
)
```
Parameters: ~25M

### Model 2: LSTM Baseline
```python
LSTMBaseline(
    vocab_size=420,
    embed_dim=256,
    hidden_dim=512,
    num_layers=2
)
```
Parameters: ~18M

### Model 3: Pure Transformer (Ablation)
```python
TransformerBaseline(
    vocab_size=420,
    embed_dim=256,
    num_heads=8,
    num_layers=4
)
```
Parameters: ~20M

---

## 4. Training Configuration

### Hyperparameters (Fixed)
```yaml
# Data
seq_len: 512
batch_size: 16
max_files: null  # Use all data

# Optimization
optimizer: AdamW
learning_rate: 1e-4
betas: [0.9, 0.98]
weight_decay: 0.01
grad_clip: 1.0

# Training
epochs: 20
warmup_steps: 4000
lr_schedule: cosine

# Regularization
dropout: 0.1
label_smoothing: 0.0

# System
mixed_precision: true
device: cuda
seed: 42
```

### Early Stopping
- Monitor: validation perplexity
- Patience: 3 epochs
- Save: best checkpoint

---

## 5. Evaluation Metrics

### Automatic Metrics

**1. Perplexity** (Primary)
- Lower is better
- Measures token prediction quality
- Standard metric for language models

**2. Top-K Accuracy**
- Top-1, Top-5, Top-10
- Measures prediction accuracy

**3. Note Density**
```python
density = num_notes / duration_seconds
target_range: 2-6 notes/sec (jazz typical)
```

**4. Pitch Range Coverage**
```python
coverage = len(unique_pitches) / 88
target: > 0.3 (good variation)
```

**5. Harmonic Consistency** (Jazz-specific)
```python
# For each note:
#   - Is it in the current chord scale?
#
consistency = notes_in_scale / total_notes
target: > 0.7
```

### Human Evaluation (if time/budget permits)

**Setup:**
- N = 10 participants (musicians)
- Each listens to 8 samples per model (blind)
- 5-point Likert scale

**Questions:**
1. Musicality (1-5)
2. Jazz authenticity (1-5)
3. Coherence (1-5)
4. Overall quality (1-5)

**Analysis:**
- Wilcoxon signed-rank test (non-parametric)
- Effect size (Cohen's d)
- Inter-rater reliability (Krippendorff's alpha)

---

## 6. Experimental Setup

### Experiment 1: Main Comparison
```bash
# Train all models with same config
python train.py --model jazzflow --data MAESTRO --epochs 20
python train.py --model lstm_baseline --data MAESTRO --epochs 20
python train.py --model transformer_baseline --data MAESTRO --epochs 20

# Evaluate
python evaluate.py --model jazzflow --data MAESTRO/test
python evaluate.py --model lstm_baseline --data MAESTRO/test
python evaluate.py --model transformer_baseline --data MAESTRO/test

# Compare
python compare_models.py --models jazzflow lstm_baseline transformer_baseline
```

### Experiment 2: Ablation Study
```bash
# Remove components one by one
python train.py --model jazzflow_no_chord --epochs 20
python train.py --model jazzflow_no_lstm --epochs 20
python train.py --model jazzflow_no_transformer --epochs 20

# Measure impact
python ablation_analysis.py
```

### Experiment 3: Hyperparameter Sensitivity (if time)
```bash
# Vary key hyperparameters
for lr in 5e-5 1e-4 2e-4; do
    python train.py --model jazzflow --lr $lr
done

for layers in 2 4 6; do
    python train.py --model jazzflow --num_layers $layers
done
```

---

## 7. Statistical Analysis

### Significance Testing

**For automatic metrics:**
- Bootstrap confidence intervals (95%)
- Paired t-test (if normally distributed)
- Wilcoxon signed-rank (if not normal)
- Bonferroni correction (multiple comparisons)

**For human evaluation:**
- Wilcoxon signed-rank test
- Effect size (Cohen's d or r)
- p < 0.05 for significance

### Reporting
```python
# Example
results = {
    'jazzflow': {
        'perplexity': 12.34 ± 0.56,
        'top1_acc': 0.45 ± 0.02,
        'p_value': 0.001  # vs baseline
    },
    'baseline': {
        'perplexity': 15.67 ± 0.78,
        'top1_acc': 0.38 ± 0.03
    }
}
```

---

## 8. Reproducibility Checklist

- [ ] Code published on GitHub
- [ ] Requirements.txt with exact versions
- [ ] Random seeds fixed (42)
- [ ] Dataset version specified (MAESTRO v3.0.0)
- [ ] Hyperparameters documented
- [ ] Checkpoints available (HuggingFace)
- [ ] Generated samples published
- [ ] Evaluation scripts included
- [ ] README with instructions

---

## 9. Timeline

**Week 1: Setup**
- Day 1-2: Download MAESTRO
- Day 3-4: Implement missing components
- Day 5-7: Test pipeline end-to-end

**Week 2-3: Training**
- Day 8-14: Train JazzFlow
- Day 15-21: Train baselines

**Week 4: Evaluation**
- Day 22-25: Automatic evaluation
- Day 26-28: Human evaluation (if applicable)

**Week 5-6: Writing**
- Day 29-35: Draft paper
- Day 36-42: Revisions, figures, submission

**Total: 6 weeks**

---

## 10. Expected Results

### Conservative Estimates

| Model | Perplexity | Top-1 Acc | Training Time |
|-------|------------|-----------|---------------|
| LSTM Baseline | 18-22 | 35-40% | 12h |
| Transformer | 15-18 | 40-45% | 16h |
| **JazzFlow** | **12-15** | **45-50%** | **20h** |

**Key Claims:**
1. JazzFlow achieves 15-20% lower perplexity than LSTM baseline
2. Chord conditioning improves harmonic consistency by 20%+
3. Hybrid architecture trains efficiently (<24h on 1 GPU)

**What we DON'T claim:**
- ❌ SOTA performance (we're not comparing to Music Informer, etc.)
- ❌ Human-level quality (need more extensive eval)
- ❌ Real-time performance (not the focus)

---

## 11. Failure Modes & Mitigation

### Potential Issues

**1. Training doesn't converge**
- Mitigation: Lower LR, increase warmup, check data

**2. Baseline performs equally well**
- Mitigation: This is fine! Honest reporting
- Pivot: Focus on efficiency (fewer params)

**3. No MAESTRO access**
- Mitigation: Use smaller dataset (1-2h MIDI)
- Adjust claims accordingly

**4. No GPU**
- Mitigation: Use smaller model (embed_dim=128)
- Train for fewer epochs

---

## 12. Paper Outline

```latex
Title: JazzFlow: Chord-Conditioned Jazz Piano Generation

Abstract: (250 words)
- Problem: Jazz generation lacks harmonic awareness
- Solution: Chord-conditioned hybrid architecture
- Results: 15% perplexity improvement, 72% harmonic consistency
- Impact: Efficient, reproducible baseline

1. Introduction
   - Music generation challenges
   - Jazz-specific requirements (harmony, rhythm)
   - Our contribution: simple but effective

2. Related Work
   - Music generation (Music Transformer, MuseGAN)
   - Chord-based generation (ChordGAN, etc.)
   - Positioning: engineering contribution, not SOTA

3. Method
   - Architecture (Transformer + LSTM + Chord embedding)
   - Training procedure
   - Chord tokenization

4. Experiments
   - Dataset: MAESTRO v3.0.0
   - Baselines: LSTM, Transformer
   - Metrics: Perplexity, accuracy, harmonic consistency
   - Results: Tables + figures

5. Analysis
   - Ablation study
   - Generated samples analysis
   - Failure cases (be honest!)

6. Discussion
   - What works: chord conditioning
   - What doesn't: still far from human
   - Limitations: classical data, no audio

7. Conclusion
   - Contribution: reproducible baseline
   - Future work: larger models, jazz data, audio

References (20-30 papers)
Appendix: Hyperparameters, more results
```

---

## 13. Success Criteria

**Minimum viable:**
- [x] Code runs without errors
- [ ] Trains to completion (20 epochs)
- [ ] Beats random baseline (perplexity < 50)
- [ ] Beats LSTM baseline (perplexity 10% lower)
- [ ] Paper drafted (8 pages)

**Stretch goals:**
- [ ] Human evaluation (N=10)
- [ ] Ablation study completed
- [ ] arXiv submission
- [ ] Workshop acceptance
- [ ] Code >100 GitHub stars

---

## 14. Budget

**Compute:**
- AWS p3.2xlarge (1x V100): $3/hour
- 20h training × 3 models = 60h = $180
- Alternative: Colab Pro ($10/month) or free tier

**Data:**
- MAESTRO: Free ✅
- Storage: ~50GB, S3 = ~$1/month

**Human Evaluation:**
- Participants: $10/person × 10 = $100
- Or: recruit volunteers (students, Reddit)

**Total: $180-290** (or $0 if using free resources)

---

**Status**: Protocol approved ✅
**Next**: Implement pipeline
**Owner**: You + AI Assistant
**Deadline**: 6 weeks from start
