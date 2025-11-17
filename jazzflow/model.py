"""
JazzFlow Model - Simple & Proven Architecture

Design: Transformer + LSTM + Chord Conditioning
Total: ~200 lines, no bloat
"""

import torch
import torch.nn as nn
import math


class ChordEmbedding(nn.Module):
    """Simple chord embedding - no overcomplicated jazz theory"""

    def __init__(self, num_chords=60, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_chords, embed_dim)

    def forward(self, chord_ids):
        # chord_ids: (batch, seq_len)
        return self.embedding(chord_ids)  # (batch, seq_len, embed_dim)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class JazzFlowModel(nn.Module):
    """
    JazzFlow: Practical Jazz Generation

    Architecture:
        - Token Embedding
        - Chord Embedding
        - Transformer Encoder (proven)
        - LSTM (proven)
        - Output projection

    No exotic components. Just solid engineering.
    """

    def __init__(
        self,
        vocab_size=512,
        num_chords=60,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        lstm_hidden=512,
        lstm_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Chord embedding (smaller)
        self.chord_embed = ChordEmbedding(num_chords, embed_dim // 4)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim + embed_dim // 4)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim + embed_dim // 4,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LSTM (adds temporal modeling)
        self.lstm = nn.LSTM(
            input_size=embed_dim + embed_dim // 4,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(lstm_hidden, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights (Xavier for transformers)"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, token_ids, chord_ids, return_hidden=False):
        """
        Forward pass

        Args:
            token_ids: (batch, seq_len) - MIDI token IDs
            chord_ids: (batch, seq_len) - Chord IDs
            return_hidden: bool - return hidden states for analysis

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: (optional) hidden states
        """
        batch_size, seq_len = token_ids.shape

        # Embeddings
        token_emb = self.token_embed(token_ids)  # (batch, seq_len, embed_dim)
        chord_emb = self.chord_embed(chord_ids)  # (batch, seq_len, embed_dim//4)

        # Concatenate
        x = torch.cat([token_emb, chord_emb], dim=-1)  # (batch, seq_len, embed_dim + embed_dim//4)

        # Positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformer
        # Create causal mask (autoregressive)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(token_ids.device)
        x = self.transformer(x, mask=mask, is_causal=True)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Output projection
        logits = self.output_proj(lstm_out)  # (batch, seq_len, vocab_size)

        if return_hidden:
            return logits, lstm_out
        else:
            return logits

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        chord_sequence,
        max_len=512,
        temperature=1.0,
        top_k=20
    ):
        """
        Autoregressive generation

        Args:
            start_tokens: (batch, start_len) - starting tokens
            chord_sequence: (batch, max_len) - chord progression
            max_len: int - maximum generation length
            temperature: float - sampling temperature
            top_k: int - top-k sampling

        Returns:
            generated: (batch, max_len) - generated token sequence
        """
        self.eval()

        batch_size = start_tokens.size(0)
        device = start_tokens.device

        # Current sequence
        current = start_tokens  # (batch, current_len)

        for step in range(start_tokens.size(1), max_len):
            # Get current chord
            current_chords = chord_sequence[:, :current.size(1)]

            # Forward pass
            logits = self.forward(current, current_chords)  # (batch, current_len, vocab_size)

            # Get last token logits
            next_logits = logits[:, -1, :] / temperature  # (batch, vocab_size)

            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                # Set others to -inf
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, top_k_indices, top_k_logits)

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Append
            current = torch.cat([current, next_token], dim=1)

        return current

    def get_num_params(self):
        """Count parameters"""
        return sum(p.numel() for p in self.parameters())


# ===== Quick Test =====

if __name__ == "__main__":
    print("JazzFlow Model Test\n")

    # Create model
    model = JazzFlowModel(
        vocab_size=512,
        num_chords=60,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        lstm_hidden=512,
        lstm_layers=2
    )

    print(f"Parameters: {model.get_num_params() / 1e6:.2f}M")

    # Test forward
    batch_size = 4
    seq_len = 128

    token_ids = torch.randint(0, 512, (batch_size, seq_len))
    chord_ids = torch.randint(0, 60, (batch_size, seq_len))

    logits = model(token_ids, chord_ids)
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {logits.shape}")

    # Test generation
    start_tokens = torch.randint(0, 512, (1, 10))
    chord_sequence = torch.randint(0, 60, (1, 100))

    generated = model.generate(
        start_tokens=start_tokens,
        chord_sequence=chord_sequence,
        max_len=100,
        temperature=1.0,
        top_k=20
    )
    print(f"Generated shape: {generated.shape}")

    print("\n✅ Model test passed!")
    print("\nArchitecture:")
    print("  - Token Embedding (256-dim)")
    print("  - Chord Embedding (64-dim)")
    print("  - Transformer Encoder (4 layers, 8 heads)")
    print("  - LSTM (2 layers, 512 hidden)")
    print("  - Output Projection (512 vocab)")
    print("\nSimple. Proven. Works.")
