#!/bin/bash
# Download and prepare MAESTRO v3.0.0 dataset

set -e

echo "🎹 Downloading MAESTRO v3.0.0 Dataset"
echo "======================================"
echo ""

# Configuration
DATA_DIR="./data"
MAESTRO_URL="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_ZIP="$DATA_DIR/maestro-v3.0.0-midi.zip"
MAESTRO_DIR="$DATA_DIR/maestro-v3.0.0"

# Create data directory
mkdir -p "$DATA_DIR"

# Download
if [ -f "$MAESTRO_ZIP" ]; then
    echo "✅ MAESTRO zip already exists: $MAESTRO_ZIP"
else
    echo "📥 Downloading MAESTRO dataset (~1.5GB)..."
    echo "This may take 5-15 minutes depending on your connection"
    echo ""

    wget --show-progress -O "$MAESTRO_ZIP" "$MAESTRO_URL"

    echo ""
    echo "✅ Download complete"
fi

# Extract
if [ -d "$MAESTRO_DIR" ]; then
    echo "✅ MAESTRO directory already exists: $MAESTRO_DIR"
else
    echo ""
    echo "📦 Extracting dataset..."

    unzip -q "$MAESTRO_ZIP" -d "$DATA_DIR"

    echo "✅ Extraction complete"
fi

# Verify
echo ""
echo "🔍 Verifying dataset..."

NUM_MIDI=$(find "$MAESTRO_DIR" -name "*.midi" -o -name "*.mid" | wc -l)
echo "  Found $NUM_MIDI MIDI files"

if [ "$NUM_MIDI" -lt 1000 ]; then
    echo "⚠️  Warning: Expected ~1,200+ files, found $NUM_MIDI"
    echo "   Dataset may be incomplete"
else
    echo "✅ Dataset verified"
fi

# Split data
echo ""
echo "📊 Creating train/val/test splits..."
python scripts/split_maestro.py --data_dir "$MAESTRO_DIR"

# Summary
echo ""
echo "======================================"
echo "✅ MAESTRO Dataset Ready!"
echo "======================================"
echo ""
echo "Location: $MAESTRO_DIR"
echo "Total files: $NUM_MIDI"
echo ""
echo "Next steps:"
echo "  1. python scripts/analyze_dataset.py"
echo "  2. python train.py --data $MAESTRO_DIR/train"
echo ""
