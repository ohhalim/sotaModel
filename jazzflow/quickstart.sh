#!/bin/bash
# JazzFlow Quickstart - Actually Works

set -e  # Exit on error

echo "🎵 JazzFlow Quickstart"
echo "====================="
echo ""

# 1. Install dependencies
echo "1️⃣  Installing dependencies..."
pip install -q torch pretty-midi tqdm numpy
echo "✅ Dependencies installed"
echo ""

# 2. Test model
echo "2️⃣  Testing model..."
python model.py
echo ""

# 3. Test data
echo "3️⃣  Testing data processing..."
python data.py
echo ""

# 4. Test baseline
echo "4️⃣  Testing baseline..."
python baseline.py
echo ""

echo "=" * 60
echo "✅ All tests passed!"
echo "=" * 60
echo ""
echo "Next steps:"
echo "  1. Get MIDI files: download MAESTRO or use your own"
echo "  2. Train: python train.py --data ./midi_files --epochs 10"
echo "  3. Generate: python generate.py --checkpoint checkpoints/best.pt"
echo "  4. Evaluate: python evaluate.py --checkpoint checkpoints/best.pt --data ./test_midi"
echo ""
echo "That's it. No TODO, no placeholders. Just works."
