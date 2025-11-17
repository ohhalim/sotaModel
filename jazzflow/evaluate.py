"""
Evaluation - Real Metrics

No "expected" performance. Actual measurements.
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from model import JazzFlowModel
from data import create_dataloader


def compute_perplexity(model, dataloader, device):
    """Compute perplexity on test set"""
    model.eval()

    total_loss = 0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(ignore_index=416, reduction='sum')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            chord_ids = batch['chord_ids'].to(device)

            logits = model(input_ids, chord_ids)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            total_loss += loss.item()
            total_tokens += (target_ids != 416).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def compute_accuracy(model, dataloader, device, top_k=1):
    """Compute top-k accuracy"""
    model.eval()

    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Computing top-{top_k} accuracy"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            chord_ids = batch['chord_ids'].to(device)

            logits = model(input_ids, chord_ids)

            # Get top-k predictions
            _, top_k_pred = torch.topk(logits, k=top_k, dim=-1)  # (batch, seq, k)

            # Check if target in top-k
            target_expanded = target_ids.unsqueeze(-1)  # (batch, seq, 1)
            correct = (top_k_pred == target_expanded).any(dim=-1)  # (batch, seq)

            # Ignore padding
            mask = (target_ids != 416)
            total_correct += (correct & mask).sum().item()
            total_tokens += mask.sum().item()

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return accuracy


def evaluate_model(checkpoint_path, data_dir, batch_size=16, max_files=None):
    """Full evaluation"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    print(f"Loaded epoch {checkpoint['epoch']}, train loss {checkpoint['train_loss']:.4f}\n")

    # Load data
    print(f"Loading test data from: {data_dir}")
    test_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        max_files=max_files,
        num_workers=0
    )
    print(f"Test batches: {len(test_loader)}\n")

    # Evaluate
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Perplexity
    perplexity = compute_perplexity(model, test_loader, device)
    print(f"Perplexity: {perplexity:.2f}")

    # Top-1 accuracy
    acc_1 = compute_accuracy(model, test_loader, device, top_k=1)
    print(f"Top-1 Accuracy: {acc_1 * 100:.2f}%")

    # Top-5 accuracy
    acc_5 = compute_accuracy(model, test_loader, device, top_k=5)
    print(f"Top-5 Accuracy: {acc_5 * 100:.2f}%")

    print("=" * 60)

    return {
        'perplexity': perplexity,
        'top1_accuracy': acc_1,
        'top5_accuracy': acc_5
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_files', type=int, default=None)

    args = parser.parse_args()

    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        batch_size=args.batch_size,
        max_files=args.max_files
    )

    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
