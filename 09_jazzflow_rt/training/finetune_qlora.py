"""
JazzFlow-RT QLoRA 파인튜닝
효율적인 재즈 스타일 커스터마이징
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from architecture.jazzflow_rt import JazzFlowRT
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb


class JazzMIDIDataset(Dataset):
    """재즈 MIDI 데이터셋"""

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 512,
        chord_vocab_size: int = 256
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.chord_vocab_size = chord_vocab_size

        # MIDI 파일 리스트
        self.midi_files = list(self.data_dir.glob("**/*.mid")) + \
                         list(self.data_dir.glob("**/*.midi"))

        print(f"📂 {len(self.midi_files)}개 MIDI 파일 발견")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        """
        실제 구현에서는 MIDI 파일을 로드하고 토크나이징
        여기서는 더미 데이터
        """
        # TODO: 실제 MIDI 로딩 및 토크나이징
        # from utils.midi_utils import load_midi, tokenize_midi

        # 더미 데이터
        midi_tokens = torch.randint(0, 512, (self.seq_len,))
        chord_ids = torch.randint(0, self.chord_vocab_size, (self.seq_len,))
        chord_types = torch.randint(0, 20, (self.seq_len,))

        return {
            'midi_tokens': midi_tokens[:-1],  # Input
            'targets': midi_tokens[1:],        # Target (다음 토큰)
            'chord_ids': chord_ids[:-1],
            'chord_types': chord_types[:-1],
        }


def setup_qlora_model(
    base_model: nn.Module,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    quantization: bool = True
) -> nn.Module:
    """
    QLoRA 설정

    Args:
        base_model: 기본 모델
        lora_rank: LoRA rank (낮을수록 파라미터 적음)
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout
        quantization: 4-bit 양자화 사용 여부

    Returns:
        QLoRA 적용된 모델
    """
    print("🔧 QLoRA 설정 중...")

    # 4-bit 양자화 (QLoRA)
    if quantization and torch.cuda.is_available():
        print("  • 4-bit 양자화 활성화 (메모리 75% 절감)")

        # 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # 모델 준비
        base_model = prepare_model_for_kbit_training(base_model)

    # LoRA 설정
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "v_proj", "k_proj",  # Attention
            "out_proj",                     # Output projection
            "lm_head"                       # Language model head
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # LoRA 적용
    model = get_peft_model(base_model, lora_config)

    # 학습 가능한 파라미터 출력
    model.print_trainable_parameters()

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    style_level: int = 6,
    epoch: int = 0
) -> float:
    """
    1 에포크 학습

    Args:
        model: JazzFlow-RT 모델
        train_loader: 데이터로더
        optimizer: 옵티마이저
        device: 디바이스
        style_level: 재즈 스타일 레벨 (0-8)
        epoch: 현재 에포크

    Returns:
        평균 loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # 데이터 -> 디바이스
        midi_tokens = batch['midi_tokens'].to(device)
        targets = batch['targets'].to(device)
        chord_ids = batch['chord_ids'].to(device)
        chord_types = batch['chord_types'].to(device)

        batch_size = midi_tokens.size(0)

        # Style level (배치마다 조금씩 다르게)
        style_levels = torch.randint(
            max(0, style_level - 1),
            min(9, style_level + 2),
            (batch_size,),
            device=device
        )

        # Forward
        optimizer.zero_grad()

        logits, info = model(
            midi_tokens=midi_tokens,
            chord_ids=chord_ids,
            style_level=style_levels,
            chord_types=chord_types,
            use_cache=False  # 학습 시에는 cache 사용 안 함
        )

        # Loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 통계
        total_loss += loss.item()
        num_batches += 1

        # Progress bar 업데이트
        pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    style_level: int = 6
) -> float:
    """Validation"""
    model.eval()
    total_loss = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(val_loader, desc="Validation"):
        midi_tokens = batch['midi_tokens'].to(device)
        targets = batch['targets'].to(device)
        chord_ids = batch['chord_ids'].to(device)
        chord_types = batch['chord_types'].to(device)

        batch_size = midi_tokens.size(0)
        style_levels = torch.full((batch_size,), style_level, device=device)

        # Forward
        logits, _ = model(
            midi_tokens=midi_tokens,
            chord_ids=chord_ids,
            style_level=style_levels,
            chord_types=chord_types,
            use_cache=False
        )

        # Loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='JazzFlow-RT QLoRA Fine-tuning')

    # 데이터
    parser.add_argument('--data_dir', type=str, required=True,
                       help='재즈 MIDI 데이터 디렉토리')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation 비율')

    # 모델
    parser.add_argument('--base_model', type=str, default=None,
                       help='사전학습 모델 경로 (None이면 랜덤 초기화)')

    # LoRA
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank (8, 16, 32)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    parser.add_argument('--use_qlora', action='store_true',
                       help='4-bit 양자화 사용 (QLoRA)')

    # 학습
    parser.add_argument('--epochs', type=int, default=30,
                       help='학습 에포크')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='학습률')
    parser.add_argument('--seq_len', type=int, default=512,
                       help='시퀀스 길이')

    # 재즈
    parser.add_argument('--style_level', type=int, default=6,
                       help='재즈 스타일 레벨 (0-8, 6=Bebop)')

    # 출력
    parser.add_argument('--output_dir', type=str, default='./models/jazzflow_finetuned',
                       help='출력 디렉토리')
    parser.add_argument('--save_every', type=int, default=5,
                       help='N 에포크마다 저장')

    args = parser.parse_args()

    print("🎺 JazzFlow-RT QLoRA 파인튜닝 시작\n")
    print(f"설정:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()

    # 디바이스
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"디바이스: {device}\n")

    # 데이터셋
    print("📂 데이터셋 로딩...")
    full_dataset = JazzMIDIDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len
    )

    # Train/Val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples\n")

    # 모델
    print("🎹 모델 생성...")
    if args.base_model:
        print(f"  사전학습 모델 로드: {args.base_model}")
        base_model = JazzFlowRT()  # 아키텍처만
        checkpoint = torch.load(args.base_model, map_location='cpu')
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  랜덤 초기화 (처음부터 학습)")
        base_model = JazzFlowRT(
            midi_vocab_size=512,
            chord_vocab_size=256,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            ff_dim=2048
        )

    # QLoRA 적용
    model = setup_qlora_model(
        base_model=base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        quantization=args.use_qlora
    )

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # 학습
    print(f"\n🚀 학습 시작 ({args.epochs} epochs)\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            style_level=args.style_level,
            epoch=epoch
        )

        # Validate
        val_loss = evaluate(
            model=model,
            val_loader=val_loader,
            device=device,
            style_level=args.style_level
        )

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        # Best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✅ New best! Saving...")

            # LoRA weights만 저장 (작은 용량)
            model.save_pretrained(output_dir / "best_lora")

            # 설정 저장
            config = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'style_level': args.style_level,
                'lora_rank': args.lora_rank,
            }

            with open(output_dir / "best_config.json", 'w') as f:
                json.dump(config, f, indent=2)

        # 주기적 저장
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(output_dir / f"checkpoint_epoch_{epoch+1}")

        print()

    print("🎉 파인튜닝 완료!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  LoRA weights 저장: {output_dir / 'best_lora'}")

    # 사용 방법 안내
    print("\n📖 사용 방법:")
    print(f"""
from peft import PeftModel
from jazzflow_rt import JazzFlowRT

# Base model
base_model = JazzFlowRT()

# LoRA weights 로드
model = PeftModel.from_pretrained(base_model, "{output_dir / 'best_lora'}")

# 생성
output = model.generate_realtime(
    chord_progression=["Dm7", "G7", "Cmaj7"],
    style_level={args.style_level}
)
    """)


if __name__ == "__main__":
    main()
