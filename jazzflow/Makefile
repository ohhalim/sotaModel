# JazzFlow - Complete Experimental Pipeline
#
# Usage:
#   make all          # Run complete pipeline (download → train → evaluate → paper)
#   make data         # Download and prepare dataset
#   make train        # Train all models
#   make eval         # Evaluate all models
#   make figures      # Generate paper figures
#   make paper        # Compile LaTeX paper
#   make clean        # Clean up
#

.PHONY: all data train eval figures paper clean help

# Configuration
DATA_DIR = ./data/maestro-v3.0.0
EXP_DIR = ./experiments/final
EPOCHS = 20
BATCH_SIZE = 16

# Python interpreter
PYTHON = python3

# ============================================================================
# Main Targets
# ============================================================================

all: data train eval figures paper
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Complete pipeline finished!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Results:"
	@echo "  📁 Experiments: $(EXP_DIR)"
	@echo "  📊 Figures: paper/figures/"
	@echo "  📄 Paper: paper/paper.pdf"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Review results in $(EXP_DIR)"
	@echo "  2. Check figures in paper/figures/"
	@echo "  3. Read paper: open paper/paper.pdf"
	@echo "  4. Submit to arXiv or workshop"
	@echo ""

help:
	@echo "JazzFlow Experimental Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  all          Run complete pipeline"
	@echo "  data         Download and prepare MAESTRO dataset"
	@echo "  train        Train JazzFlow and baseline models"
	@echo "  eval         Evaluate models on test set"
	@echo "  figures      Generate paper figures"
	@echo "  paper        Compile LaTeX paper to PDF"
	@echo "  clean        Remove generated files"
	@echo "  help         Show this message"
	@echo ""
	@echo "Configuration:"
	@echo "  DATA_DIR     = $(DATA_DIR)"
	@echo "  EXP_DIR      = $(EXP_DIR)"
	@echo "  EPOCHS       = $(EPOCHS)"
	@echo "  BATCH_SIZE   = $(BATCH_SIZE)"
	@echo ""

# ============================================================================
# Data Preparation
# ============================================================================

data: $(DATA_DIR)/train $(DATA_DIR)/validation $(DATA_DIR)/test
	@echo ""
	@echo "✅ Dataset ready!"
	@echo ""

$(DATA_DIR)/train $(DATA_DIR)/validation $(DATA_DIR)/test:
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📥 Downloading MAESTRO v3.0.0 dataset"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	bash scripts/download_maestro.sh
	@echo ""
	@echo "📊 Analyzing dataset"
	$(PYTHON) scripts/analyze_dataset.py --data_dir $(DATA_DIR) --output data/dataset_stats.json

# ============================================================================
# Training
# ============================================================================

train: $(EXP_DIR)/jazzflow/best.pt $(EXP_DIR)/lstm_baseline/best.pt
	@echo ""
	@echo "✅ Training complete!"
	@echo ""

$(EXP_DIR)/jazzflow/best.pt: $(DATA_DIR)/train
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🎵 Training JazzFlow"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	mkdir -p $(EXP_DIR)/jazzflow
	$(PYTHON) train.py \
		--data $(DATA_DIR)/train \
		--val_data $(DATA_DIR)/validation \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--output $(EXP_DIR)/jazzflow \
		2>&1 | tee $(EXP_DIR)/jazzflow/train.log

$(EXP_DIR)/lstm_baseline/best.pt: $(DATA_DIR)/train
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🎵 Training LSTM Baseline"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	mkdir -p $(EXP_DIR)/lstm_baseline
	$(PYTHON) train.py \
		--data $(DATA_DIR)/train \
		--val_data $(DATA_DIR)/validation \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--model baseline \
		--output $(EXP_DIR)/lstm_baseline \
		2>&1 | tee $(EXP_DIR)/lstm_baseline/train.log

# ============================================================================
# Evaluation
# ============================================================================

eval: $(EXP_DIR)/jazzflow/eval.log $(EXP_DIR)/lstm_baseline/eval.log
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📊 Comparing Models"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	$(PYTHON) scripts/compare_models.py \
		--exp_dir $(EXP_DIR) \
		--models jazzflow lstm_baseline \
		--output $(EXP_DIR)/comparison.png
	@echo ""
	$(PYTHON) scripts/statistical_tests.py --exp_dir $(EXP_DIR)
	@echo ""
	@echo "✅ Evaluation complete!"
	@echo ""

$(EXP_DIR)/jazzflow/eval.log: $(EXP_DIR)/jazzflow/best.pt $(DATA_DIR)/test
	@echo ""
	@echo "📊 Evaluating JazzFlow"
	$(PYTHON) evaluate.py \
		--checkpoint $(EXP_DIR)/jazzflow/best.pt \
		--data $(DATA_DIR)/test \
		--batch_size $(BATCH_SIZE) \
		2>&1 | tee $(EXP_DIR)/jazzflow/eval.log
	@echo ""
	@echo "🎵 Generating sample"
	$(PYTHON) generate.py \
		--checkpoint $(EXP_DIR)/jazzflow/best.pt \
		--output $(EXP_DIR)/jazzflow/sample.mid \
		--length 256

$(EXP_DIR)/lstm_baseline/eval.log: $(EXP_DIR)/lstm_baseline/best.pt $(DATA_DIR)/test
	@echo ""
	@echo "📊 Evaluating LSTM Baseline"
	$(PYTHON) evaluate.py \
		--checkpoint $(EXP_DIR)/lstm_baseline/best.pt \
		--data $(DATA_DIR)/test \
		--batch_size $(BATCH_SIZE) \
		--model baseline \
		2>&1 | tee $(EXP_DIR)/lstm_baseline/eval.log

# ============================================================================
# Figures
# ============================================================================

figures: paper/figures/training_curves.pdf
	@echo ""
	@echo "✅ Figures generated!"
	@echo ""

paper/figures/training_curves.pdf: $(EXP_DIR)/jazzflow/eval.log $(EXP_DIR)/lstm_baseline/eval.log
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🎨 Generating Paper Figures"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	$(PYTHON) scripts/make_figures.py --exp_dir $(EXP_DIR)

# ============================================================================
# Paper Compilation
# ============================================================================

paper: paper/paper.pdf
	@echo ""
	@echo "✅ Paper compiled!"
	@echo ""

paper/paper.pdf: paper/paper.tex paper/figures/training_curves.pdf
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📄 Compiling LaTeX Paper"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	cd paper && pdflatex paper.tex
	cd paper && bibtex paper || true
	cd paper && pdflatex paper.tex
	cd paper && pdflatex paper.tex
	@echo ""
	@echo "✅ Paper compiled: paper/paper.pdf"

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "🧹 Cleaning up..."
	rm -rf experiments/
	rm -rf paper/figures/
	rm -rf paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
	rm -rf data/dataset_stats.json
	rm -rf __pycache__ .pytest_cache
	@echo "✅ Clean!"

clean-data:
	@echo "🧹 Removing dataset (WARNING: Large download)..."
	rm -rf $(DATA_DIR)
	@echo "✅ Dataset removed"

# ============================================================================
# Quick Checks
# ============================================================================

check:
	@echo "🔍 Checking dependencies..."
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	@$(PYTHON) -c "import pretty_midi; print(f'pretty_midi: OK')" || echo "⚠️  pretty_midi not installed"
	@which pdflatex > /dev/null && echo "pdflatex: OK" || echo "⚠️  pdflatex not installed"
	@echo ""
	@echo "✅ Dependency check complete"

test:
	@echo "🧪 Running tests..."
	$(PYTHON) test_end_to_end.py
	@echo ""
	@echo "✅ Tests passed!"
