"""
RNN 멜로디 생성 모델 학습 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# 상위 디렉토리의 utils import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from model import MelodyLSTM
from utils.train_utils import set_seed, get_device, Trainer, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description='Train Melody RNN')
    parser.add_argument('--data_dir', type=str, default='./data/midi_files',
                       help='MIDI 파일 디렉토리')
    parser.add_argument('--epochs', type=int, default=50,
                       help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--seq_len', type=int, default=32,
                       help='시퀀스 길이')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='LSTM 레이어 수')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='학습률')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout 비율')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='체크포인트 디렉토리')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    return parser.parse_args()


def main():
    args = parse_args()

    print("🎵 Melody RNN 학습 시작\n")
    print(f"설정:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()

    # 재현성
    set_seed(args.seed)

    # 디바이스
    device = get_device()

    # 모델
    model = MelodyLSTM(
        vocab_size=128,
        embedding_dim=256,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # TODO: 실제 데이터셋 로드
    # 여기서는 더미 데이터 사용
    print("⚠️  실제 구현 시 MIDI 데이터셋을 로드해야 합니다.")
    print("   dataset.py를 구현하고 여기서 사용하세요.\n")

    # 간단한 학습 루프 (예시)
    model.to(device)
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0
        num_batches = 10  # 더미

        for batch_idx in range(num_batches):
            # 더미 데이터
            x = torch.randint(0, 128, (args.batch_size, args.seq_len)).to(device)
            y = torch.randint(0, 128, (args.batch_size, args.seq_len)).to(device)

            # Forward
            optimizer.zero_grad()
            logits, _ = model(x)

            # Loss
            loss = criterion(
                logits.reshape(-1, 128),
                y.reshape(-1)
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_dir / f'epoch_{epoch+1}.pt')

    # Final checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
    }, Path(args.checkpoint_dir) / 'final.pt')

    print(f"\n✅ 학습 완료! 체크포인트: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
