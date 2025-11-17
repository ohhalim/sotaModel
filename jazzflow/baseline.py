"""
Simple LSTM Baseline - For Fair Comparison

This is what we beat. No chord conditioning, just pure LSTM.
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Simple LSTM baseline (no chord conditioning)

    This is our comparison point.
    """

    def __init__(
        self,
        vocab_size=420,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        """
        Forward pass

        Args:
            token_ids: (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Embedding
        x = self.embedding(token_ids)
        x = self.dropout(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Output
        logits = self.output(lstm_out)

        return logits

    @torch.no_grad()
    def generate(self, start_tokens, max_len=512, temperature=1.0, top_k=20):
        """Autoregressive generation"""
        self.eval()

        batch_size = start_tokens.size(0)
        device = start_tokens.device

        current = start_tokens

        for step in range(start_tokens.size(1), max_len):
            # Forward
            logits = self.forward(current)

            # Last token logits
            next_logits = logits[:, -1, :] / temperature

            # Top-k
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, top_k_indices, top_k_logits)

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            current = torch.cat([current, next_token], dim=1)

        return current

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# ===== Quick Test =====

if __name__ == "__main__":
    print("LSTM Baseline Test\n")

    model = LSTMBaseline(
        vocab_size=420,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2
    )

    print(f"Parameters: {model.get_num_params() / 1e6:.2f}M")

    # Test forward
    tokens = torch.randint(0, 420, (4, 128))
    logits = model(tokens)

    print(f"Input: {tokens.shape}")
    print(f"Output: {logits.shape}")

    # Test generation
    start = torch.randint(0, 420, (1, 10))
    generated = model.generate(start, max_len=100)
    print(f"Generated: {generated.shape}")

    print("\n✅ Baseline test passed!")
    print("\nThis is what JazzFlow needs to beat.")
    print("If JazzFlow < Baseline, something is wrong.")
