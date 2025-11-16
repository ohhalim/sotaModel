"""
학습된 RNN 모델로 멜로디 생성
"""

import torch
import argparse
from pathlib import Path
import sys
import pretty_midi

sys.path.append(str(Path(__file__).parent.parent.parent))

from model import MelodyLSTM
from utils.midi_utils import notes_to_midi


def parse_args():
    parser = argparse.ArgumentParser(description='Generate melody with RNN')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='모델 체크포인트 경로')
    parser.add_argument('--length', type=int, default=100,
                       help='생성할 노트 개수')
    parser.add_argument('--start_pitch', type=int, default=60,
                       help='시작 음높이 (MIDI pitch)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='샘플링 온도 (낮을수록 보수적)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k 샘플링 (None=전체)')
    parser.add_argument('--output', type=str, default='generated.mid',
                       help='출력 MIDI 파일')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"🎵 멜로디 생성 시작\n")
    print(f"설정:")
    print(f"  체크포인트: {args.checkpoint}")
    print(f"  길이: {args.length}")
    print(f"  시작 음: {args.start_pitch}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print()

    # 디바이스
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 로드
    model = MelodyLSTM(
        vocab_size=128,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ 모델 로드 완료\n")

    # 시작 시퀀스
    start_sequence = torch.tensor([[args.start_pitch]])

    # 생성
    print("🎹 생성 중...")
    with torch.no_grad():
        generated = model.generate(
            start_sequence=start_sequence,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

    # MIDI 변환
    pitches = generated.squeeze(0).cpu().numpy()
    print(f"생성된 음높이 (처음 10개): {pitches[:10]}")

    # MIDI 파일로 저장
    notes = []
    current_time = 0.0
    note_duration = 0.5  # 기본 duration

    for pitch in pitches:
        if 0 <= pitch <= 127:  # 유효한 MIDI pitch
            notes.append({
                'pitch': int(pitch),
                'start': current_time,
                'end': current_time + note_duration,
                'velocity': 80
            })
            current_time += note_duration

    # MIDI 저장
    notes_to_midi(notes, args.output)
    print(f"\n✅ 생성 완료: {args.output}")
    print(f"   총 노트 수: {len(notes)}")
    print(f"   길이: {current_time:.1f}초")


if __name__ == "__main__":
    main()
