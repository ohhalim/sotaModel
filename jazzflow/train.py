"""
Training Script - Actually Works

No TODO, no placeholders. Just run it.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from model import JazzFlowModel
from data import create_dataloader


def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Single training epoch"""
    model.train()

    total_loss = 0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(ignore_index=416)  # Ignore PAD

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        chord_ids = batch['chord_ids'].to(device)

        optimizer.zero_grad()

        # Mixed precision
        with autocast():
            logits = model(input_ids, chord_ids)

            # Reshape for loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Stats
        batch_tokens = (target_ids != 416).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        pbar.set_postfix({'loss': total_loss / total_tokens})

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


@torch.no_grad()
def validate(model, dataloader, device):
    """Validation"""
    model.eval()

    total_loss = 0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(ignore_index=416)

    for batch in tqdm(dataloader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        chord_ids = batch['chord_ids'].to(device)

        logits = model(input_ids, chord_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )

        batch_tokens = (target_ids != 416).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data', type=str, required=True,
                       help='MIDI files directory')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Limit number of files (for testing)')

    # Model
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--lstm_hidden', type=int, default=512)

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Output
    parser.add_argument('--output', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=1)

    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading data...")
    train_loader = create_dataloader(
        data_dir=args.data,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_files=args.max_files
    )
    print(f"Train batches: {len(train_loader)}\n")

    # Model
    print("Creating model...")
    model = JazzFlowModel(
        vocab_size=420,  # SimpleMIDITokenizer vocab
        num_chords=60,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=2
    ).to(device)

    print(f"Parameters: {model.get_num_params() / 1e6:.2f}M\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )

    # Mixed precision
    scaler = GradScaler()

    # Training loop
    print(f"Training for {args.epochs} epochs...\n")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_ppl = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | PPL: {train_ppl:.2f}")

        # Save
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_ppl': train_ppl
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")

        # Save best
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = output_dir / "best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_ppl': train_ppl
            }, best_path)
            print(f"  ✅ New best! Loss: {best_loss:.4f}")

        print("=" * 60)

    print("\n🎉 Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
