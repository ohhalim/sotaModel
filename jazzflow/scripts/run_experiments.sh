#!/bin/bash
# Run all experiments for the paper

set -e

echo "🔬 JazzFlow Experimental Pipeline"
echo "=================================="
echo ""

# Configuration
DATA_DIR="./data/maestro-v3.0.0"
EPOCHS=20
BATCH_SIZE=16
SEQ_LEN=512

# Create experiment directory
EXP_DIR="./experiments/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_DIR"

echo "📁 Experiment directory: $EXP_DIR"
echo ""

# Save configuration
cat > "$EXP_DIR/config.txt" <<EOF
Experiment Configuration
========================
Date: $(date)
Data: $DATA_DIR
Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Sequence Length: $SEQ_LEN

Models:
1. JazzFlow (ours)
2. LSTM Baseline
3. Transformer Baseline (optional)
EOF

# Function to train a model
train_model() {
    MODEL_NAME=$1
    EXTRA_ARGS=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Training: $MODEL_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    OUTPUT_DIR="$EXP_DIR/$MODEL_NAME"
    mkdir -p "$OUTPUT_DIR"

    # Train
    python train.py \
        --data "$DATA_DIR/train" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seq_len $SEQ_LEN \
        --output "$OUTPUT_DIR" \
        $EXTRA_ARGS \
        2>&1 | tee "$OUTPUT_DIR/train.log"

    echo ""
    echo "✅ $MODEL_NAME training complete"
    echo "   Checkpoints: $OUTPUT_DIR"
    echo ""
}

# Function to evaluate a model
evaluate_model() {
    MODEL_NAME=$1

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Evaluating: $MODEL_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    CHECKPOINT="$EXP_DIR/$MODEL_NAME/best.pt"
    OUTPUT_DIR="$EXP_DIR/$MODEL_NAME"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "⚠️  Checkpoint not found: $CHECKPOINT"
        echo "   Skipping evaluation"
        return
    fi

    # Evaluate on test set
    python evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --data "$DATA_DIR/test" \
        --batch_size $BATCH_SIZE \
        2>&1 | tee "$OUTPUT_DIR/eval.log"

    # Generate samples
    echo ""
    echo "🎵 Generating sample MIDI..."
    python generate.py \
        --checkpoint "$CHECKPOINT" \
        --output "$OUTPUT_DIR/sample.mid" \
        --length 256

    echo ""
    echo "✅ $MODEL_NAME evaluation complete"
    echo ""
}

# ====================================
# Experiment 1: Main Comparison
# ====================================

echo "📊 Experiment 1: Main Comparison"
echo "=================================="
echo ""

# Model 1: JazzFlow (ours)
train_model "jazzflow" ""

# Model 2: LSTM Baseline
train_model "lstm_baseline" "--model baseline"

# Optional: Transformer baseline
# Uncomment if you have time/resources
# train_model "transformer_baseline" "--model transformer"

# ====================================
# Evaluation
# ====================================

echo ""
echo "📊 Evaluation Phase"
echo "==================="
echo ""

evaluate_model "jazzflow"
evaluate_model "lstm_baseline"
# evaluate_model "transformer_baseline"

# ====================================
# Comparison
# ====================================

echo ""
echo "📊 Model Comparison"
echo "==================="
echo ""

python scripts/compare_models.py \
    --exp_dir "$EXP_DIR" \
    --models jazzflow lstm_baseline \
    --output "$EXP_DIR/comparison.png"

# ====================================
# Summary
# ====================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All Experiments Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: $EXP_DIR"
echo ""
echo "Files:"
echo "  - Training logs: */train.log"
echo "  - Evaluation logs: */eval.log"
echo "  - Checkpoints: */best.pt"
echo "  - Generated samples: */sample.mid"
echo "  - Comparison plot: comparison.png"
echo ""
echo "Next steps:"
echo "  1. Review results in $EXP_DIR"
echo "  2. Generate paper figures: python scripts/make_figures.py"
echo "  3. Write paper: cd paper && pdflatex paper.tex"
echo ""
