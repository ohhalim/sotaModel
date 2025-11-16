"""
Simple RNN 멜로디 생성 모델
LSTM 기반 시퀀스 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MelodyLSTM(nn.Module):
    """
    MIDI 멜로디 생성을 위한 LSTM 모델

    입력: MIDI pitch sequence (batch, seq_len)
    출력: Next pitch logits (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor (batch, seq_len)
            hidden: Hidden state (h, c) 또는 None

        Returns:
            (logits, hidden_state)
            - logits: (batch, seq_len, vocab_size)
            - hidden_state: (h, c)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)

        # lstm_out: (batch, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)

        # Output
        logits = self.fc_out(lstm_out)  # (batch, seq_len, vocab_size)

        return logits, hidden

    def generate(
        self,
        start_sequence: torch.Tensor,
        length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        음악 생성

        Args:
            start_sequence: 시작 시퀀스 (1, start_len)
            length: 생성할 길이
            temperature: 샘플링 온도 (낮을수록 보수적)
            top_k: Top-k 샘플링 (None이면 전체)
            device: 디바이스

        Returns:
            생성된 시퀀스 (1, start_len + length)
        """
        self.eval()

        with torch.no_grad():
            current_sequence = start_sequence.to(device)
            hidden = None

            for _ in range(length):
                # Forward
                logits, hidden = self.forward(current_sequence, hidden)

                # 마지막 타임스텝의 logits
                next_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

                # Top-k 샘플링
                if top_k is not None:
                    top_k = min(top_k, next_logits.size(-1))
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                # 샘플링
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                # 시퀀스에 추가
                current_sequence = next_token

            # 전체 생성 시퀀스 반환
            return torch.cat([start_sequence, current_sequence], dim=1)


class MelodyLSTMWithVelocity(nn.Module):
    """
    Pitch + Velocity를 함께 예측하는 모델
    """

    def __init__(
        self,
        pitch_vocab_size: int = 128,
        velocity_vocab_size: int = 128,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # Pitch embedding
        self.pitch_embedding = nn.Embedding(pitch_vocab_size, embedding_dim // 2)

        # Velocity embedding
        self.velocity_embedding = nn.Embedding(velocity_vocab_size, embedding_dim // 2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.pitch_head = nn.Linear(hidden_dim, pitch_vocab_size)
        self.velocity_head = nn.Linear(hidden_dim, velocity_vocab_size)

    def forward(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            pitch: (batch, seq_len)
            velocity: (batch, seq_len)
            hidden: Hidden state

        Returns:
            (pitch_logits, velocity_logits, hidden)
        """
        # Embedding
        pitch_emb = self.pitch_embedding(pitch)
        velocity_emb = self.velocity_embedding(velocity)

        # Concatenate
        embedded = torch.cat([pitch_emb, velocity_emb], dim=-1)
        embedded = self.dropout(embedded)

        # LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)

        lstm_out = self.dropout(lstm_out)

        # Predict pitch and velocity
        pitch_logits = self.pitch_head(lstm_out)
        velocity_logits = self.velocity_head(lstm_out)

        return pitch_logits, velocity_logits, hidden


# ===== 사용 예시 =====

if __name__ == "__main__":
    print("🎵 Melody LSTM 모델 테스트\n")

    # 하이퍼파라미터
    vocab_size = 128
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 2
    batch_size = 4
    seq_len = 32

    # 모델 생성
    model = MelodyLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 데이터
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\n입력 shape: {x.shape}")

    # Forward
    logits, hidden = model(x)
    print(f"출력 logits shape: {logits.shape}")
    print(f"Hidden state shapes: {hidden[0].shape}, {hidden[1].shape}")

    # 생성 테스트
    print("\n=== 음악 생성 테스트 ===")
    start_sequence = torch.tensor([[60, 64, 67]])  # C Major chord
    print(f"시작 시퀀스: {start_sequence.tolist()}")

    generated = model.generate(
        start_sequence=start_sequence,
        length=10,
        temperature=1.0,
        device='cpu'
    )
    print(f"생성된 시퀀스: {generated.tolist()}")

    print("\n✅ 모델 테스트 완료!")
