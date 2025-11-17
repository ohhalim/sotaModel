"""
Generation Script - Simple & Works

python generate.py --checkpoint best.pt --output jazz.mid
"""

import torch
import argparse
from pathlib import Path

from model import JazzFlowModel
from data import SimpleMIDITokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='generated.mid')
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--tempo', type=int, default=120)

    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = JazzFlowModel(
        vocab_size=420,
        num_chords=60,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        lstm_hidden=512,
        lstm_layers=2
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (epoch {checkpoint['epoch']}, loss {checkpoint['train_loss']:.4f})\n")

    # Tokenizer
    tokenizer = SimpleMIDITokenizer()

    # Start tokens (BOS + middle C)
    start_tokens = torch.tensor([[tokenizer.BOS, 60]], device=device)

    # Chord sequence (dummy for now)
    chord_sequence = torch.zeros((1, args.length), dtype=torch.long, device=device)

    # Generate
    print(f"Generating {args.length} tokens...")
    with torch.no_grad():
        generated = model.generate(
            start_tokens=start_tokens,
            chord_sequence=chord_sequence,
            max_len=args.length,
            temperature=args.temperature,
            top_k=args.top_k
        )

    print(f"Generated {generated.size(1)} tokens\n")

    # Decode to MIDI
    print(f"Decoding to MIDI...")
    tokens = generated[0].cpu().tolist()
    tokenizer.decode(tokens, args.output, tempo=args.tempo)

    print(f"✅ Saved: {args.output}")
    print(f"\nPlay it:")
    print(f"  timidity {args.output}")
    print(f"  or open in MuseScore, GarageBand, etc.")


if __name__ == "__main__":
    main()
