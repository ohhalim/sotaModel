"""
학습 유틸리티
- 학습 루프
- 체크포인트 관리
- 학습률 스케줄러
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Callable, List
import numpy as np
from tqdm import tqdm
import json
import time


class Trainer:
    """
    범용 학습 클래스
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
        save_interval: int = 1000,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        # Loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        # 학습 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> Dict[str, float]:
        """
        1 에포크 학습
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 데이터 -> 디바이스
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            else:
                batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x
                             for x in batch)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(batch)

            # Loss 계산
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = self.criterion(outputs, batch)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 통계
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 로깅
            if self.global_step % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # 체크포인트 저장
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        avg_epoch_loss = epoch_loss / num_batches
        return {'train_loss': avg_epoch_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validation
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # 데이터 -> 디바이스
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            else:
                batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x
                             for x in batch)

            # Forward
            outputs = self.model(batch)

            # Loss
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = self.criterion(outputs, batch)

            val_loss += loss.item()
            num_batches += 1

        avg_val_loss = val_loss / num_batches
        return {'val_loss': avg_val_loss}

    def train(self, num_epochs: int):
        """
        전체 학습 루프
        """
        print(f"🚀 학습 시작: {num_epochs} epochs")
        print(f"디바이스: {self.device}")
        print(f"체크포인트 디렉토리: {self.checkpoint_dir}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['train_loss'])

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['val_loss'])

                # Best model 저장
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint("best.pt")
                    print(f"✅ Best model saved (val_loss: {self.best_val_loss:.4f})")

            # 에포크 결과 출력
            print(f"\nEpoch {epoch}: train_loss={train_metrics['train_loss']:.4f}", end="")
            if val_metrics:
                print(f", val_loss={val_metrics['val_loss']:.4f}", end="")
            print()

            # 주기적 저장
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")

        print("🎉 학습 완료!")
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, filename: str):
        """
        체크포인트 저장
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        # print(f"💾 체크포인트 저장: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        체크포인트 로드
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"✅ 체크포인트 로드: {checkpoint_path}")
        print(f"   Epoch: {self.current_epoch}, Step: {self.global_step}")


class EarlyStopping:
    """
    Early Stopping
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"⏹️ Early stopping triggered (patience={self.patience})")
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    """
    Cosine 학습률 스케줄러 (Warmup 포함)
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * num_cycles * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    모델 파라미터 수 계산
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
    }


def set_seed(seed: int = 42):
    """
    재현성을 위한 시드 설정
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> str:
    """
    최적 디바이스 선택
    """
    if prefer_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 GPU 사용: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 버전: {torch.version.cuda}")
        print(f"   사용 가능 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("💻 CPU 사용")

    return device


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    seconds_per_batch: float
) -> Dict[str, float]:
    """
    학습 시간 추정
    """
    num_batches_per_epoch = num_samples // batch_size
    total_batches = num_batches_per_epoch * num_epochs
    total_seconds = total_batches * seconds_per_batch

    return {
        'batches_per_epoch': num_batches_per_epoch,
        'total_batches': total_batches,
        'estimated_hours': total_seconds / 3600,
        'estimated_days': total_seconds / (3600 * 24),
    }


# ===== 사용 예시 =====
if __name__ == "__main__":
    print("🎓 학습 유틸리티 테스트")

    # 간단한 모델
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    # 파라미터 수
    params = count_parameters(model)
    print(f"파라미터 수: {params}")

    # 디바이스
    device = get_device()

    print("✅ 학습 유틸리티 정상 작동")
