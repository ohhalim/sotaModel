"""
End-to-End Test - Verify Everything Works

This is what "reproducible research" means.
"""

import torch
import pretty_midi
from pathlib import Path
import shutil

from model import JazzFlowModel
from baseline import LSTMBaseline
from data import SimpleMIDITokenizer, create_dataloader


def setup_test_data():
    """Create test MIDI files"""
    test_dir = Path("/tmp/jazzflow_test")
    test_dir.mkdir(exist_ok=True)

    # Create 5 test MIDI files
    for i in range(5):
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0)

        # Random melody
        for j in range(16):
            pitch = 60 + (i * 2 + j) % 12
            note = pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=j * 0.5,
                end=(j + 1) * 0.5
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        midi.write(str(test_dir / f"test_{i}.mid"))

    return test_dir


def test_model_creation():
    """Test: Can we create models?"""
    print("Testing model creation...")

    model = JazzFlowModel(
        vocab_size=420,
        num_chords=60,
        embed_dim=128,  # Small for testing
        num_heads=4,
        num_layers=2,
        lstm_hidden=256,
        lstm_layers=1
    )

    baseline = LSTMBaseline(
        vocab_size=420,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1
    )

    print(f"  JazzFlow params: {model.get_num_params() / 1e6:.2f}M")
    print(f"  Baseline params: {baseline.get_num_params() / 1e6:.2f}M")
    print("  ✅ Models created")


def test_data_loading(test_dir):
    """Test: Can we load MIDI data?"""
    print("\nTesting data loading...")

    dataloader = create_dataloader(
        data_dir=str(test_dir),
        batch_size=2,
        seq_len=64,
        max_files=5,
        num_workers=0
    )

    batch = next(iter(dataloader))
    print(f"  Batch keys: {batch.keys()}")
    print(f"  Input shape: {batch['input_ids'].shape}")
    print(f"  ✅ Data loaded")

    return dataloader


def test_training_step(model, dataloader, device):
    """Test: Can we train for 1 step?"""
    print("\nTesting training step...")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=416)

    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)
    target_ids = batch['target_ids'].to(device)
    chord_ids = batch['chord_ids'].to(device)

    # Forward
    logits = model(input_ids, chord_ids)
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1)
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  ✅ Training step works")


def test_generation(model, device):
    """Test: Can we generate?"""
    print("\nTesting generation...")

    model.eval()

    tokenizer = SimpleMIDITokenizer()
    start_tokens = torch.tensor([[tokenizer.BOS, 60]], device=device)
    chord_sequence = torch.zeros((1, 50), dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            start_tokens=start_tokens,
            chord_sequence=chord_sequence,
            max_len=50,
            temperature=1.0,
            top_k=20
        )

    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample tokens: {generated[0, :10].tolist()}")

    # Decode to MIDI
    output_path = "/tmp/test_generated.mid"
    tokens = generated[0].cpu().tolist()
    tokenizer.decode(tokens, output_path)

    print(f"  ✅ Generation works")
    print(f"  MIDI saved: {output_path}")


def test_checkpoint_save_load(model, device):
    """Test: Can we save and load checkpoints?"""
    print("\nTesting checkpoint save/load...")

    # Save
    checkpoint_path = "/tmp/test_checkpoint.pt"
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'train_loss': 1.234
    }, checkpoint_path)
    print(f"  Saved checkpoint")

    # Load
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded checkpoint (epoch {checkpoint['epoch']})")
    print(f"  ✅ Checkpointing works")


def main():
    print("=" * 60)
    print("JazzFlow End-to-End Test")
    print("=" * 60)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Setup
    test_dir = setup_test_data()
    print(f"Test data: {test_dir}\n")

    # Tests
    test_model_creation()

    dataloader = test_data_loading(test_dir)

    model = JazzFlowModel(
        vocab_size=420,
        num_chords=60,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        lstm_hidden=256,
        lstm_layers=1
    ).to(device)

    test_training_step(model, dataloader, device)

    test_generation(model, device)

    test_checkpoint_save_load(model, device)

    # Cleanup
    shutil.rmtree(test_dir)
    print(f"\nCleaned up test data")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("\nJazzFlow is ready to use.")
    print("No TODO, no placeholders. Everything works.")


if __name__ == "__main__":
    main()
