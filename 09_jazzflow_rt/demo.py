#!/usr/bin/env python3
"""
JazzFlow-RT Demo
간단한 재즈 생성 데모
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from architecture.jazzflow_rt import JazzFlowRT
from data_processing.midi_tokenizer import REMITokenizer


def generate_jazz(
    chord_progression=["Dm7", "G7", "Cmaj7", "Am7"],
    style_level=6,
    num_bars=8,
    output_file="demo_output.mid",
    use_pretrained=None
):
    """
    재즈 생성 데모

    Args:
        chord_progression: 코드 진행
        style_level: 재즈 스타일 (0=클래식 ~ 8=Hard Bop)
        num_bars: 마디 수
        output_file: 출력 MIDI 파일
        use_pretrained: 사전학습 모델 경로 (None이면 랜덤 초기화)
    """
    print("\n🎺 JazzFlow-RT Demo")
    print("="*60)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"디바이스: {device}")

    # Tokenizer
    print("\n1. Tokenizer 생성...")
    tokenizer = REMITokenizer()
    print(f"   Vocabulary: {tokenizer.vocab_size} 토큰")

    # Model
    print("\n2. 모델 로드...")
    if use_pretrained:
        print(f"   사전학습 모델: {use_pretrained}")
        model = JazzFlowRT(
            midi_vocab_size=tokenizer.vocab_size,
            chord_vocab_size=256
        )
        checkpoint = torch.load(use_pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"   랜덤 초기화 (데모용)")
        model = JazzFlowRT(
            midi_vocab_size=tokenizer.vocab_size,
            chord_vocab_size=256,
            hidden_dim=256,  # 작은 모델
            num_layers=4,
            num_heads=4,
            ff_dim=1024
        )

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   파라미터: {total_params / 1e6:.2f}M")

    # Generate
    print(f"\n3. 재즈 생성 중...")
    print(f"   코드 진행: {' → '.join(chord_progression)}")
    print(f"   스타일 레벨: {style_level} (0=클래식, 6=Bebop, 8=Hard Bop)")
    print(f"   마디 수: {num_bars}")

    max_notes = num_bars * 16  # 마디당 16개 노트

    with torch.no_grad():
        generated_tokens = model.generate_realtime(
            chord_progression=chord_progression,
            style_level=style_level,
            num_bars=num_bars,
            tempo=140,
            max_notes=max_notes
        )

    print(f"\n✅ 생성 완료!")
    print(f"   생성된 토큰: {generated_tokens.shape[1]}개")

    # Decode to MIDI
    print(f"\n4. MIDI 변환 중...")
    try:
        tokenizer.decode(
            token_ids=generated_tokens[0].cpu().tolist(),
            output_path=output_file,
            tempo=140
        )
        print(f"✅ MIDI 저장 완료: {output_file}")
        print(f"\n🎵 MIDI 플레이어로 재생하세요:")
        print(f"   {output_file}")
    except Exception as e:
        print(f"⚠️  MIDI 변환 실패: {e}")

    print("\n" + "="*60)

    return generated_tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='JazzFlow-RT Demo')

    parser.add_argument('--chords', type=str,
                       default="Dm7,G7,Cmaj7,Am7",
                       help='코드 진행 (콤마로 구분)')
    parser.add_argument('--style', type=int, default=6,
                       help='재즈 스타일 (0-8)')
    parser.add_argument('--bars', type=int, default=8,
                       help='생성할 마디 수')
    parser.add_argument('--output', type=str, default='demo_output.mid',
                       help='출력 MIDI 파일')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='사전학습 모델 경로 (선택)')

    args = parser.parse_args()

    # Parse chords
    chords = [c.strip() for c in args.chords.split(',')]

    # Generate
    generate_jazz(
        chord_progression=chords,
        style_level=args.style,
        num_bars=args.bars,
        output_file=args.output,
        use_pretrained=args.checkpoint
    )
