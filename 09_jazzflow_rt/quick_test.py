#!/usr/bin/env python3
"""
JazzFlow-RT Quick Test
빠른 작동 확인용 스크립트
"""

import torch
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from architecture.jazzflow_rt import JazzFlowRT
from data_processing.midi_tokenizer import REMITokenizer
from evaluation.metrics import JazzMetrics


def test_tokenizer():
    """MIDI Tokenizer 테스트"""
    print("\n" + "="*60)
    print("1️⃣  MIDI Tokenizer 테스트")
    print("="*60)

    tokenizer = REMITokenizer()
    print(f"✅ Tokenizer 생성 성공")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    print(f"   샘플 토큰: {list(tokenizer.vocab[:10])}")

    return tokenizer


def test_model(tokenizer):
    """모델 생성 및 forward 테스트"""
    print("\n" + "="*60)
    print("2️⃣  JazzFlow-RT 모델 테스트")
    print("="*60)

    # 작은 모델 (빠른 테스트용)
    model = JazzFlowRT(
        midi_vocab_size=tokenizer.vocab_size,
        chord_vocab_size=256,
        hidden_dim=256,  # 작게
        num_layers=2,    # 작게
        num_heads=4,
        ff_dim=512,
        chunk_size=64
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 모델 생성 성공")
    print(f"   파라미터 수: {total_params / 1e6:.2f}M")

    # Forward test
    batch_size = 2
    seq_len = 64

    midi_tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    chord_ids = torch.randint(0, 256, (batch_size, seq_len))
    style_level = torch.tensor([5, 7])

    print(f"\n   Forward pass 테스트...")
    print(f"   입력 shape: {midi_tokens.shape}")

    logits, info = model(midi_tokens, chord_ids, style_level)

    print(f"✅ Forward pass 성공")
    print(f"   출력 logits: {logits.shape}")
    print(f"   Swing ratio: {info['swing_ratio'].shape}")

    return model


def test_generation(model):
    """생성 테스트"""
    print("\n" + "="*60)
    print("3️⃣  실시간 생성 테스트")
    print("="*60)

    chord_progression = ["Dm7", "G7", "Cmaj7", "Am7"]

    print(f"   코드 진행: {' - '.join(chord_progression)}")
    print(f"   생성 중...")

    generated = model.generate_realtime(
        chord_progression=chord_progression,
        style_level=6,
        num_bars=4,
        max_notes=32
    )

    print(f"✅ 생성 성공")
    print(f"   생성된 토큰 수: {generated.shape[1]}")
    print(f"   처음 10개: {generated[0, :10].tolist()}")

    return generated


def test_metrics():
    """평가 메트릭 테스트"""
    print("\n" + "="*60)
    print("4️⃣  Jazz Metrics 테스트")
    print("="*60)

    # 더미 노트
    notes = []
    for i in range(16):
        notes.append({
            'pitch': 60 + (i % 12),
            'start': i * 0.5,
            'end': (i + 1) * 0.5,
            'velocity': 80
        })

    # 더미 코드 진행
    chord_progression = [
        {'type': 'm7', 'root': 2, 'start': 0.0, 'end': 4.0},
        {'type': '7', 'root': 7, 'start': 4.0, 'end': 8.0},
    ]

    metrics = JazzMetrics()
    results = metrics.evaluate_generation(notes, chord_progression)

    print(f"✅ Metrics 평가 성공")
    print(f"   Harmonic Consistency: {results['harmonic_consistency']:.4f}")
    print(f"   Swing Ratio Score: {results['swing_ratio_score']:.4f}")
    print(f"   Overall Score: {results['overall_score']:.4f}")


def main():
    print("\n🎺 JazzFlow-RT Quick Test")
    print("모든 컴포넌트가 올바르게 작동하는지 확인합니다.\n")

    try:
        # 1. Tokenizer
        tokenizer = test_tokenizer()

        # 2. Model
        model = test_model(tokenizer)

        # 3. Generation
        generated = test_generation(model)

        # 4. Metrics
        test_metrics()

        print("\n" + "="*60)
        print("🎉 모든 테스트 통과!")
        print("="*60)
        print("\n✅ JazzFlow-RT가 정상적으로 작동합니다!")
        print("\n다음 단계:")
        print("  1. 데이터셋 다운로드 (MAESTRO, PiJAMA)")
        print("  2. 사전학습: python training/pretrain.py --data_dir <path>")
        print("  3. 파인튜닝: python training/finetune_qlora.py --data_dir <path>")
        print("  4. 생성: python inference/live_jam.py --checkpoint <path>")

        return True

    except Exception as e:
        print("\n" + "="*60)
        print("❌ 테스트 실패!")
        print("="*60)
        print(f"\n에러: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
