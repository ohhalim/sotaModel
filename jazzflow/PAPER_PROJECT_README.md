# JazzFlow Paper Project

**From Zero to arXiv in One Command**

This directory contains a complete, reproducible experimental pipeline for the JazzFlow paper. Everything needed to train models, measure results, generate figures, and compile the paper.

## 🎯 Philosophy

- **Reproducible**: Every result is automatically measured, not "expected"
- **Honest**: We report what we actually achieve, not what we hoped for
- **Automated**: One command runs the entire pipeline
- **Complete**: Dataset download → training → evaluation → paper compilation

## 📁 Directory Structure

```
jazzflow/
├── Makefile                    # One-command pipeline orchestration
├── EXPERIMENTAL_PROTOCOL.md    # Complete research protocol
├── PAPER_PROJECT_README.md     # This file
│
├── model.py                    # JazzFlow architecture (200 lines)
├── baseline.py                 # LSTM baseline (100 lines)
├── data.py                     # Dataset and tokenizer (150 lines)
├── train.py                    # Training script (100 lines)
├── evaluate.py                 # Evaluation script (120 lines)
├── generate.py                 # MIDI generation (80 lines)
├── test_end_to_end.py         # End-to-end verification
│
├── scripts/
│   ├── download_maestro.sh    # Download MAESTRO v3.0.0
│   ├── split_maestro.py       # Official train/val/test split
│   ├── analyze_dataset.py     # Dataset statistics
│   ├── run_experiments.sh     # Manual experiment runner
│   ├── compare_models.py      # Model comparison + LaTeX tables
│   ├── statistical_tests.py   # Statistical analysis
│   └── make_figures.py        # Paper figure generation
│
└── paper/
    ├── paper.tex              # Full ICML 2025 format paper
    ├── references.bib         # Bibliography
    └── figures/               # Generated figures (auto-created)
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python dependencies
pip install torch numpy pretty_midi tqdm scipy matplotlib

# LaTeX (for paper compilation)
# Ubuntu/Debian: sudo apt-get install texlive-full
# macOS: brew install --cask mactex
```

### Run Complete Pipeline

```bash
# One command to rule them all
make all
```

This will:
1. Download MAESTRO dataset (~1.5GB)
2. Split into train/validation/test
3. Analyze dataset statistics
4. Train JazzFlow (~20 hours on V100)
5. Train LSTM baseline (~18 hours on V100)
6. Evaluate both models on test set
7. Generate comparison plots and tables
8. Run statistical analysis
9. Generate paper figures
10. Compile LaTeX paper to PDF

**Total time**: ~40 hours on single NVIDIA V100 GPU

### Individual Steps

```bash
# Just download data
make data

# Just train models
make train

# Just evaluate
make eval

# Just generate figures
make figures

# Just compile paper
make paper

# Check dependencies
make check

# Run tests
make test

# Clean up everything
make clean
```

## 📊 What Gets Measured

### Automatic Metrics

1. **Perplexity** (primary metric)
   - Lower is better
   - Measured on test set
   - Expected: JazzFlow < LSTM by 10-15%

2. **Top-1 Accuracy**
   - Percentage of correctly predicted next tokens
   - Expected: 40-50%

3. **Top-5 Accuracy**
   - Percentage where correct token is in top-5
   - Expected: 65-75%

4. **Harmonic Consistency** (optional)
   - Percentage of notes in correct scale
   - Requires manual implementation

### Training Metrics

- Loss curves (train and validation)
- Training time per epoch
- GPU memory usage
- Final model size (~25M parameters)

### Statistical Analysis

- Bootstrap confidence intervals
- Wilcoxon signed-rank test (if multiple runs)
- Effect size (Cohen's d)
- Significance levels

## 📈 Expected Timeline

Based on EXPERIMENTAL_PROTOCOL.md:

**Week 1**: Setup (you are here)
- Download MAESTRO: 30 minutes
- Test pipeline: 2 hours
- Verify everything works

**Week 2-3**: Training
- JazzFlow: ~20 hours
- LSTM baseline: ~18 hours
- Can run in parallel if 2 GPUs

**Week 4**: Evaluation
- Automatic evaluation: 2 hours
- Generate samples: 1 hour
- Human evaluation: optional

**Week 5-6**: Paper Writing
- Fill in actual results: 2 hours
- Generate figures: 1 hour
- Revisions: 1 week
- Submission ready!

## 🎯 Success Criteria

From EXPERIMENTAL_PROTOCOL.md Section 11:

### Minimum Viable

✅ JazzFlow trains without errors
✅ Baseline trains without errors
✅ Models achieve <100 perplexity (not random)
✅ Paper compiles to PDF

### Target

✅ JazzFlow perplexity < LSTM by 10%+
✅ Harmonic consistency > 60%
✅ Generated samples are musically coherent
✅ All figures render correctly

### Stretch

✅ Statistical significance (p < 0.05)
✅ Human evaluation shows preference
✅ Submission to workshop/arXiv
✅ GitHub stars > 50

## 📝 Paper Writing Workflow

1. **Run experiments**
   ```bash
   make train eval
   ```

2. **Generate figures and tables**
   ```bash
   make figures
   ```

3. **Fill in actual results**
   Edit `paper/paper.tex`:
   - Replace placeholder numbers in Abstract
   - Update Table 1 (main results)
   - Update Table 2 (ablation study)
   - Add any qualitative observations

4. **Compile paper**
   ```bash
   make paper
   ```

5. **Review and iterate**
   - Check paper/paper.pdf
   - Revise text as needed
   - Re-run `make paper`

## 🔬 Experiment Configuration

Defined in Makefile (can override):

```makefile
DATA_DIR = ./data/maestro-v3.0.0
EXP_DIR = ./experiments/final
EPOCHS = 20
BATCH_SIZE = 16
```

Override example:
```bash
make train EPOCHS=50 BATCH_SIZE=32
```

## 📊 Understanding Results

### Training Logs

```bash
# View JazzFlow training
tail -f experiments/final/jazzflow/train.log

# View baseline training
tail -f experiments/final/lstm_baseline/train.log
```

### Evaluation Results

```bash
# JazzFlow test set performance
cat experiments/final/jazzflow/eval.log

# Baseline test set performance
cat experiments/final/lstm_baseline/eval.log
```

### Model Comparison

```bash
# Statistical summary
cat experiments/final/statistical_summary.json

# Comparison plot
open experiments/final/comparison.png
```

### Generated Samples

```bash
# Listen to JazzFlow output
open experiments/final/jazzflow/sample.mid

# Compare with baseline
open experiments/final/lstm_baseline/sample.mid
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
make train BATCH_SIZE=8
```

### Dataset Download Fails

```bash
# Manual download
cd data
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip
```

### LaTeX Compilation Fails

```bash
# Compile manually
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### PyTorch Not Found

```bash
# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU only
pip install torch torchvision torchaudio
```

## 📚 File Descriptions

### Core Implementation

- **model.py**: JazzFlow hybrid Transformer-LSTM architecture
  - ~200 lines, well-commented
  - 25M parameters
  - Chord conditioning via embeddings

- **baseline.py**: LSTM-only baseline
  - ~100 lines
  - No chord conditioning
  - Fair comparison (similar capacity)

- **data.py**: Dataset and tokenizer
  - SimpleMIDITokenizer (420 tokens)
  - JazzMIDIDataset with caching
  - Chord pseudo-labels

### Scripts

- **train.py**: Training loop
  - Mixed precision (AMP)
  - Gradient clipping
  - Cosine annealing LR
  - Validation every epoch

- **evaluate.py**: Test set evaluation
  - Perplexity computation
  - Top-k accuracy
  - Batch processing

- **generate.py**: Autoregressive generation
  - Temperature sampling
  - Top-k filtering
  - MIDI export

### Automation

- **Makefile**: Pipeline orchestration
  - Handles dependencies
  - Parallel execution where possible
  - Clean targets

- **scripts/compare_models.py**: Comparison
  - Parses training logs
  - Generates plots
  - Creates LaTeX tables

- **scripts/statistical_tests.py**: Statistics
  - Bootstrap CI
  - Wilcoxon test
  - Effect sizes

- **scripts/make_figures.py**: Paper figures
  - Training curves
  - Accuracy comparison
  - Piano rolls
  - Dataset statistics

## 🎓 Citation

If you use this code or build upon it:

```bibtex
@article{jazzflow2025,
  title={JazzFlow: Chord-Conditioned Jazz Piano Generation with Hybrid Transformer-LSTM Architecture},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- MAESTRO dataset: Hawthorne et al. (2019)
- Music Transformer: Huang et al. (2018)
- PyTorch team
- ICML 2025 organizers

## 📧 Contact

For questions or issues:
- Open GitHub issue
- Email: your.email@domain.com

---

**Remember**: The goal is not to achieve SOTA. The goal is to do honest, reproducible science.

Good luck! 🎵
