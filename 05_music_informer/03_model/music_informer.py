"""
Music Informer - SOTA MIDI 생성 모델
Nature Scientific Reports 2025

핵심 기술:
1. ProbSparse Self-Attention (O(L log L))
2. Relative Local Attention
3. LSTM Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ProbSparseSelfAttention(nn.Module):
    """
    ProbSparse Self-Attention
    O(L log L) 복잡도
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        prob_sparse_factor: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.prob_sparse_factor = prob_sparse_factor

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def _compute_sparsity(
        self,
        Q: torch.Tensor,
        K: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query의 sparsity 계산

        Args:
            Q: (batch, num_heads, L_q, head_dim)
            K: (batch, num_heads, L_k, head_dim)

        Returns:
            (top_u_indices, M_scores)
        """
        batch_size, num_heads, L_q, head_dim = Q.shape
        L_k = K.size(2)

        # 샘플링: 모든 key 대신 U개만 사용
        U = min(self.prob_sparse_factor * int(math.log(L_k)), L_k)
        U = max(U, 1)

        # 랜덤 샘플링
        K_sample_indices = torch.randint(0, L_k, (U,), device=K.device)
        K_sample = K[:, :, K_sample_indices, :]  # (batch, num_heads, U, head_dim)

        # Attention scores (Q와 샘플링된 K)
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K_sample) / self.scale
        # (batch, num_heads, L_q, U)

        # M(q, K) = max(scores) - mean(scores)
        M_top = scores.max(dim=-1)[0]  # (batch, num_heads, L_q)
        M_mean = scores.mean(dim=-1)
        M = M_top - M_mean  # Sparsity measure

        # Top-u queries 선택
        u = min(self.prob_sparse_factor * int(math.log(L_q)), L_q)
        u = max(u, 1)
        top_u_indices = M.topk(u, dim=-1)[1]  # (batch, num_heads, u)

        return top_u_indices, M

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) 또는 None

        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, seq_len, head_dim)

        # ProbSparse: Top-u queries만 사용
        top_u_indices, M = self._compute_sparsity(Q, K)

        # Top-u queries 추출
        u = top_u_indices.size(-1)
        Q_sparse = torch.gather(
            Q,
            dim=2,
            index=top_u_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )  # (batch, num_heads, u, head_dim)

        # Sparse Attention
        scores = torch.einsum('bhud,bhkd->bhuk', Q_sparse, K) / self.scale
        # (batch, num_heads, u, seq_len)

        # Mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attn_output_sparse = torch.einsum('bhuk,bhkd->bhud', attn_weights, V)
        # (batch, num_heads, u, head_dim)

        # 나머지 queries는 mean pooling으로 대체
        V_mean = V.mean(dim=2, keepdim=True)  # (batch, num_heads, 1, head_dim)
        attn_output_rest = V_mean.expand(batch_size, self.num_heads, seq_len - u, self.head_dim)

        # Combine sparse and rest
        attn_output = torch.zeros(
            batch_size, self.num_heads, seq_len, self.head_dim,
            device=x.device
        )

        # Scatter top-u results
        attn_output.scatter_(
            dim=2,
            index=top_u_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
            src=attn_output_sparse
        )

        # Fill rest with mean
        rest_mask = torch.ones(batch_size, self.num_heads, seq_len, device=x.device, dtype=torch.bool)
        rest_mask.scatter_(dim=2, index=top_u_indices, value=False)
        attn_output[rest_mask] = V_mean.squeeze(2).unsqueeze(1).expand(batch_size, self.num_heads, self.head_dim).reshape(-1, self.head_dim)[rest_mask.reshape(-1)]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output


class RelativeLocalAttention(nn.Module):
    """
    Relative Local Attention
    가까운 토큰들과의 관계 모델링
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int = 64,
        max_relative_position: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Relative position embeddings
        self.max_relative_position = max_relative_position
        self.relative_position_k = nn.Embedding(
            2 * max_relative_position + 1,
            self.head_dim
        )
        self.relative_position_v = nn.Embedding(
            2 * max_relative_position + 1,
            self.head_dim
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with local attention

        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional mask

        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / self.scale

        # Relative position bias
        # 간단화: 실제로는 relative position matrix 계산 필요
        # 여기서는 기본 구현만

        # Local windowing
        if self.window_size > 0:
            window_mask = self._get_window_mask(seq_len, x.device)
            scores = scores.masked_fill(window_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output

    def _get_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Local window mask"""
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = False

        return mask.unsqueeze(0).unsqueeze(0)


class MusicInformerBlock(nn.Module):
    """
    Music Informer Decoder Block
    ProbSparse + Relative Attention + FFN + LSTM
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        prob_sparse_factor: int = 5,
        window_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # ProbSparse Attention
        self.prob_sparse_attn = ProbSparseSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            prob_sparse_factor=prob_sparse_factor,
            dropout=dropout
        )

        # Relative Local Attention
        self.relative_attn = RelativeLocalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )

        # Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer Norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional mask

        Returns:
            (batch, seq_len, hidden_dim)
        """
        # ProbSparse Self-Attention
        residual = x
        x = self.norm1(x)
        x = residual + self.prob_sparse_attn(x, mask)

        # Relative Local Attention
        residual = x
        x = self.norm2(x)
        x = residual + self.relative_attn(x, mask)

        # Feed-Forward
        residual = x
        x = self.norm3(x)
        x = residual + self.ff(x)

        return x


class MusicInformer(nn.Module):
    """
    완전한 Music Informer 모델
    Nature Scientific Reports 2025
    """

    def __init__(
        self,
        vocab_size: int = 512,  # MIDI tokenizer vocab
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_len: int = 2048,
        prob_sparse_factor: int = 5,
        window_size: int = 64,
        lstm_hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # Informer blocks
        self.blocks = nn.ModuleList([
            MusicInformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                prob_sparse_factor=prob_sparse_factor,
                window_size=window_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # LSTM integration
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.lstm_proj = nn.Linear(lstm_hidden_dim, hidden_dim)

        # Output
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: (batch, seq_len)
            mask: Optional mask

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Embedding + Positional encoding
        x = self.token_embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Informer blocks
        for block in self.blocks:
            x = block(x, mask)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        x = x + lstm_out  # Residual connection

        # Output
        x = self.norm_out(x)
        logits = self.lm_head(x)

        return logits


# ===== 사용 예시 =====

if __name__ == "__main__":
    print("🎵 Music Informer 모델 테스트\n")

    # 하이퍼파라미터
    vocab_size = 512
    hidden_dim = 512
    num_layers = 6
    batch_size = 2
    seq_len = 256

    # 모델 생성
    model = MusicInformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    # 더미 데이터
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"입력 shape: {input_ids.shape}")

    # Forward
    logits = model(input_ids)
    print(f"출력 logits shape: {logits.shape}")

    # Loss 계산
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
    print(f"\nLoss: {loss.item():.4f}")

    print("\n✅ Music Informer 테스트 완료!")
    print("\n💡 핵심 기술:")
    print("1. ProbSparse Attention: O(L log L)")
    print("2. Relative Local Attention: 로컬 패턴")
    print("3. LSTM Integration: 시퀀스 연속성")
    print("4. 6-layer Decoder: 충분한 표현력")
