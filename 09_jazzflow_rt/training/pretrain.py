"""
JazzFlow-RT 사전학습 스크립트
MAESTRO 데이터셋으로 사전학습
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
import sys
import json
from tqdm import tqdm
import wandb
import time

sys.path.append(str(Path(__file__).parent.parent))

from architecture.jazzflow_rt import JazzFlowRT
from data_processing.dataset import create_dataloaders
from data_processing.midi_tokenizer import REMITokenizer


class Trainer:
    """사전학습 Trainer"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        self.use_wandb = config.get('use_wandb', False)

    def train_epoch(self) -> dict:
        """1 에포크 학습"""
        self.model.train()

        total_loss = 0
        total_tokens = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Data to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            chord_ids = batch.get('chord_ids', torch.zeros_like(input_ids)).to(self.device)
            style_level = batch.get('style_level', torch.tensor([6] * input_ids.size(0))).to(self.device)

            # Forward pass (with AMP)
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits, info = self.model(
                        midi_tokens=input_ids,
                        chord_ids=chord_ids,
                        style_level=style_level,
                        use_cache=False
                    )

                    # Loss
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1)
                    )

                # Backward (with AMP)
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Regular training (without AMP)
                logits, info = self.model(
                    midi_tokens=input_ids,
                    chord_ids=chord_ids,
                    style_level=style_level,
                    use_cache=False
                )

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Statistics
            batch_tokens = (target_ids != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': total_loss / total_tokens,
                'lr': f'{current_lr:.2e}'
            })

            # Wandb logging
            if self.use_wandb and self.global_step % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': current_lr,
                    'train/global_step': self.global_step
                })

            # Save checkpoint (every N steps)
            if self.global_step % self.config.get('save_every_steps', 5000) == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        avg_loss = total_loss / total_tokens

        return {
            'train_loss': avg_loss,
            'train_ppl': torch.exp(torch.tensor(avg_loss)).item()
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validation"""
        self.model.eval()

        total_loss = 0
        total_tokens = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            chord_ids = batch.get('chord_ids', torch.zeros_like(input_ids)).to(self.device)
            style_level = batch.get('style_level', torch.tensor([6] * input_ids.size(0))).to(self.device)

            # Forward
            logits, info = self.model(
                midi_tokens=input_ids,
                chord_ids=chord_ids,
                style_level=style_level,
                use_cache=False
            )

            # Loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            batch_tokens = (target_ids != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        avg_loss = total_loss / total_tokens

        return {
            'val_loss': avg_loss,
            'val_ppl': torch.exp(torch.tensor(avg_loss)).item()
        }

    def train(self, num_epochs: int):
        """전체 학습 루프"""
        print(f"\n{'='*60}")
        print(f"🚀 JazzFlow-RT 사전학습 시작")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            # Print results
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f} | PPL: {train_metrics['train_ppl']:.2f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f} | PPL: {val_metrics['val_ppl']:.2f}")

            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['train_loss'],
                    'train/epoch_ppl': train_metrics['train_ppl'],
                    'val/loss': val_metrics['val_loss'],
                    'val/ppl': val_metrics['val_ppl']
                })

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint("best.pt")
                print(f"  ✅ New best model! Val Loss: {self.best_val_loss:.4f}")

            # Save epoch checkpoint
            if (epoch + 1) % self.config.get('save_every_epochs', 10) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        print(f"\n🎉 학습 완료!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")

        # Save final model
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, filename: str):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        # print(f"💾 Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='JazzFlow-RT Pre-training')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='MAESTRO 데이터 디렉토리')
    parser.add_argument('--max_files', type=int, default=None,
                       help='최대 파일 수 (테스트용)')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ff_dim', type=int, default=2048)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true',
                       help='Mixed precision training')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/pretrain')
    parser.add_argument('--save_every_epochs', type=int, default=10)
    parser.add_argument('--save_every_steps', type=int, default=5000)

    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='jazzflow-rt')

    # Others
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"디바이스: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Wandb init
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )

    # Tokenizer
    print("🔤 Tokenizer 생성...")
    tokenizer = REMITokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}\n")

    # DataLoaders
    print("📂 데이터 로딩...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
        max_files=args.max_files
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")

    # Model
    print("🎹 모델 생성...")
    model = JazzFlowRT(
        midi_vocab_size=tokenizer.vocab_size,
        chord_vocab_size=256,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        chunk_size=64,
        max_seq_len=args.seq_len * 2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01
    )

    # Scheduler (cosine with warmup)
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / (args.epochs * len(train_loader) - args.warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Config
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'lr': args.lr,
        'warmup_steps': args.warmup_steps,
        'grad_clip': args.grad_clip,
        'use_amp': args.use_amp,
        'checkpoint_dir': args.checkpoint_dir,
        'save_every_epochs': args.save_every_epochs,
        'save_every_steps': args.save_every_steps,
        'use_wandb': args.use_wandb,
        'model': {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'ff_dim': args.ff_dim
        }
    }

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # Train!
    trainer.train(num_epochs=args.epochs)

    # Save config
    with open(Path(args.checkpoint_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ 학습 완료! 체크포인트: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
