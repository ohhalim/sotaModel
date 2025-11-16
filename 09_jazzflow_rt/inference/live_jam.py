"""
JazzFlow-RT Live Jam Session
실시간 재즈 잼 세션 시스템
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import time
import sys
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from architecture.jazzflow_rt import JazzFlowRT
from peft import PeftModel

# 오디오 출력 (실제로는 pyaudio, sounddevice 등 사용)
try:
    import pretty_midi
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  오디오 라이브러리 없음. MIDI 파일만 생성됩니다.")


class LiveJamSession:
    """
    실시간 재즈 잼 세션
    """

    def __init__(
        self,
        model: nn.Module,
        chord_progression: List[str],
        style_level: int = 6,
        tempo: int = 140,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()

        self.chord_progression = chord_progression
        self.style_level = style_level
        self.tempo = tempo
        self.device = device

        # 코드 토크나이저 (간단화 버전)
        self.chord_to_id = self._build_chord_vocab()

        # 생성 상태
        self.current_bar = 0
        self.current_chord_idx = 0
        self.generated_notes = []

        # 타이밍
        self.bar_duration = 60.0 / tempo * 4  # 4/4 박자 가정
        self.start_time = None

        # 실시간 캐시 초기화
        self.model.reset_cache()

        print(f"🎺 Live Jam Session 초기화 완료")
        print(f"  코드 진행: {' - '.join(chord_progression)}")
        print(f"  스타일: Level {style_level}")
        print(f"  템포: {tempo} BPM")

    def _build_chord_vocab(self) -> dict:
        """간단한 코드 vocabulary"""
        # 실제로는 더 복잡한 tokenizer 필요
        chords = [
            # Major 7th
            "Cmaj7", "Dbmaj7", "Dmaj7", "Ebmaj7", "Emaj7", "Fmaj7",
            "Gbmaj7", "Gmaj7", "Abmaj7", "Amaj7", "Bbmaj7", "Bmaj7",
            # Dominant 7th
            "C7", "Db7", "D7", "Eb7", "E7", "F7",
            "Gb7", "G7", "Ab7", "A7", "Bb7", "B7",
            # Minor 7th
            "Cm7", "Dbm7", "Dm7", "Ebm7", "Em7", "Fm7",
            "Gbm7", "Gm7", "Abm7", "Am7", "Bbm7", "Bm7",
            # Half-diminished
            "Cm7b5", "Dbm7b5", "Dm7b5", "Ebm7b5", "Em7b5", "Fm7b5",
            "Gbm7b5", "Gm7b5", "Abm7b5", "Am7b5", "Bbm7b5", "Bm7b5",
        ]

        return {chord: i for i, chord in enumerate(chords)}

    @torch.no_grad()
    def generate_bar(self, bar_idx: int) -> List[dict]:
        """
        한 마디 생성 (실시간)

        Returns:
            [{pitch, start, duration, velocity}, ...]
        """
        # 현재 코드
        chord_idx = bar_idx % len(self.chord_progression)
        chord_name = self.chord_progression[chord_idx]

        if chord_name not in self.chord_to_id:
            print(f"⚠️  알 수 없는 코드: {chord_name}, 기본값 사용")
            chord_id = 0
        else:
            chord_id = self.chord_to_id[chord_name]

        chord_tensor = torch.tensor([[chord_id]], device=self.device)
        style_tensor = torch.tensor([self.style_level], device=self.device)

        # 한 마디에 생성할 노트 수 (템포에 따라 조절)
        notes_per_bar = 8 if self.tempo < 120 else 16

        # 시작 토큰 (이전 노트가 있으면 마지막 노트, 없으면 코드 루트)
        if len(self.generated_notes) > 0:
            start_token = self.generated_notes[-1]['pitch']
        else:
            start_token = 60  # C4

        current_token = torch.tensor([[start_token]], device=self.device)

        bar_notes = []

        for note_idx in range(notes_per_bar):
            # Forward (실시간 모드)
            logits, info = self.model(
                midi_tokens=current_token,
                chord_ids=chord_tensor,
                style_level=style_tensor,
                use_cache=True  # KV-cache 사용 (중요!)
            )

            # 다음 토큰 샘플링
            next_logits = logits[:, -1, :]

            # Temperature 조절 (스타일에 따라)
            temperature = 0.8 + (self.style_level / 18.0)  # 0.8 ~ 1.2
            next_logits = next_logits / temperature

            # Top-k sampling
            top_k = 20
            top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            next_pitch = top_k_indices[0, sampled_idx].item()

            # MIDI pitch 범위 체크
            if next_pitch < 21 or next_pitch > 108:
                next_pitch = np.clip(next_pitch, 21, 108)

            # Swing ratio (from model output)
            swing = info['swing_ratio'][0, -1, 0].item() if 'swing_ratio' in info else 1.5

            # 노트 타이밍 계산
            note_duration = self.bar_duration / notes_per_bar
            note_duration *= swing  # Swing 적용

            note_start = bar_idx * self.bar_duration + note_idx * (self.bar_duration / notes_per_bar)

            # 노트 정보
            note = {
                'pitch': next_pitch,
                'start': note_start,
                'duration': note_duration * 0.8,  # 약간 짧게 (staccato)
                'velocity': np.random.randint(70, 100),
                'bar': bar_idx,
                'chord': chord_name
            }

            bar_notes.append(note)

            # 다음 스텝 준비
            current_token = torch.tensor([[next_pitch]], device=self.device)

        return bar_notes

    def start_session(
        self,
        num_bars: int = 32,
        save_midi: Optional[str] = None,
        visualize: bool = True
    ):
        """
        잼 세션 시작!

        Args:
            num_bars: 생성할 마디 수
            save_midi: MIDI 저장 경로
            visualize: 실시간 시각화 (콘솔)
        """
        print(f"\n{'='*60}")
        print(f"🎺 JazzFlow-RT Live Jam Session 시작!")
        print(f"{'='*60}\n")

        self.start_time = time.time()
        all_notes = []

        for bar in range(num_bars):
            bar_start_time = time.time()

            # 한 마디 생성
            bar_notes = self.generate_bar(bar)
            all_notes.extend(bar_notes)

            # 생성 시간 측정
            generation_time = (time.time() - bar_start_time) * 1000  # ms

            # 실시간 성능 체크
            real_time_factor = self.bar_duration / (generation_time / 1000.0)

            # 진행 상황 출력
            chord_name = self.chord_progression[bar % len(self.chord_progression)]

            if visualize:
                # 콘솔 시각화
                bar_str = f"Bar {bar+1:3d}/{num_bars} | {chord_name:8s} |"

                # 생성된 노트 시각화 (간단한 ASCII)
                note_viz = ""
                for note in bar_notes:
                    pitch_class = note['pitch'] % 12
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                                 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note_viz += note_names[pitch_class] + " "

                bar_str += f" {note_viz[:30]:30s} |"

                # 성능 정보
                bar_str += f" {generation_time:5.1f}ms ({real_time_factor:4.1f}x RT)"

                # 실시간 여부
                if real_time_factor >= 1.0:
                    bar_str += " ✅"
                else:
                    bar_str += " ⚠️"

                print(bar_str)

            # 실시간 대기 (실제 템포에 맞춤)
            # 실제로는 오디오 재생과 동기화
            # time.sleep(self.bar_duration)

        total_time = time.time() - self.start_time

        print(f"\n{'='*60}")
        print(f"🎉 Jam Session 완료!")
        print(f"{'='*60}")
        print(f"총 시간: {total_time:.2f}초")
        print(f"생성 노트 수: {len(all_notes)}")
        print(f"평균 실시간 성능: {(num_bars * self.bar_duration) / total_time:.2f}x RT")

        # MIDI 저장
        if save_midi:
            self._save_midi(all_notes, save_midi)
            print(f"💾 MIDI 저장: {save_midi}")

        return all_notes

    def _save_midi(self, notes: List[dict], output_path: str):
        """생성된 노트를 MIDI 파일로 저장"""
        if not AUDIO_AVAILABLE:
            print("⚠️  pretty_midi 없음. MIDI 저장 불가")
            return

        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        for note_dict in notes:
            note = pretty_midi.Note(
                velocity=note_dict['velocity'],
                pitch=note_dict['pitch'],
                start=note_dict['start'],
                end=note_dict['start'] + note_dict['duration']
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        midi.write(output_path)


def main():
    parser = argparse.ArgumentParser(description='JazzFlow-RT Live Jam Session')

    # 모델
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='모델 체크포인트 경로')
    parser.add_argument('--lora_weights', type=str, default=None,
                       help='LoRA weights 경로 (파인튜닝 모델)')

    # 음악 설정
    parser.add_argument('--chords', type=str,
                       default="Dm7,G7,Cmaj7,Am7",
                       help='코드 진행 (콤마로 구분)')
    parser.add_argument('--style', type=int, default=6,
                       help='재즈 스타일 (0=클래식 ~ 8=Hard Bop)')
    parser.add_argument('--tempo', type=int, default=140,
                       help='템포 (BPM)')

    # 세션 설정
    parser.add_argument('--bars', type=int, default=32,
                       help='생성할 마디 수')
    parser.add_argument('--output', type=str, default='live_jam.mid',
                       help='출력 MIDI 파일')

    # 모드
    parser.add_argument('--demo', action='store_true',
                       help='데모 모드 (랜덤 초기화 모델)')

    args = parser.parse_args()

    print("🎺 JazzFlow-RT Live Jam\n")

    # 디바이스
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"디바이스: {device}\n")

    # 코드 진행 파싱
    chord_progression = args.chords.split(',')
    chord_progression = [c.strip() for c in chord_progression]

    # 모델 로드
    if args.demo:
        print("🔧 데모 모드: 랜덤 초기화 모델 사용\n")
        model = JazzFlowRT(
            midi_vocab_size=512,
            chord_vocab_size=256,
            hidden_dim=256,  # 작게
            num_layers=4,
            num_heads=4,
            ff_dim=1024
        )
    else:
        print(f"📂 모델 로드: {args.checkpoint}\n")
        model = JazzFlowRT()
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        # LoRA weights (있으면)
        if args.lora_weights:
            print(f"📂 LoRA weights 로드: {args.lora_weights}\n")
            model = PeftModel.from_pretrained(model, args.lora_weights)

    # Jam Session 시작!
    session = LiveJamSession(
        model=model,
        chord_progression=chord_progression,
        style_level=args.style,
        tempo=args.tempo,
        device=device
    )

    notes = session.start_session(
        num_bars=args.bars,
        save_midi=args.output,
        visualize=True
    )

    print("\n🎵 완료! 생성된 MIDI를 재생하세요:")
    print(f"   {args.output}")


if __name__ == "__main__":
    main()
