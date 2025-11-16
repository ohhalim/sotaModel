#!/usr/bin/env python3
"""
PiJAMA 재즈 데이터셋 다운로드 스크립트
~200시간 재즈 피아노
"""

import argparse
from pathlib import Path


def download_pijama(output_dir: str):
    """
    PiJAMA 재즈 데이터셋 다운로드

    Args:
        output_dir: 저장 디렉토리
    """
    print("🎺 PiJAMA 재즈 데이터셋 다운로드\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PiJAMA는 2025년 신규 데이터셋으로, 실제 URL은 논문/프로젝트 페이지 참고
    print("⚠️  PiJAMA는 2025년 신규 데이터셋입니다.")
    print("   실제 다운로드 URL은 아래 소스를 참고하세요:\n")

    print("📖 참고 자료:")
    print("   1. 논문: [PiJAMA 논문 링크]")
    print("   2. GitHub: [PiJAMA GitHub]")
    print("   3. 웹사이트: [PiJAMA 웹사이트]\n")

    print("📥 다운로드 방법:")
    print("   1. 위 웹사이트 방문")
    print("   2. 다운로드 신청 (연구 목적)")
    print("   3. 승인 후 다운로드 링크 받기")
    print(f"   4. 다운로드한 파일을 {output_dir}에 저장\n")

    print("💡 대안:")
    print("   - Weimar Jazz Database")
    print("   - iRealPro chord progressions")
    print("   - YouTube 재즈 MIDI 컬렉션")

    # 더미 데이터 생성 (테스트용)
    print(f"\n🔧 테스트용 더미 데이터 생성...")
    dummy_dir = output_dir / "dummy_jazz"
    dummy_dir.mkdir(parents=True, exist_ok=True)

    print(f"✅ 더미 디렉토리 생성: {dummy_dir}")
    print(f"   실제 데이터로 교체하세요!\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Download PiJAMA dataset')
    parser.add_argument('--output_dir', type=str, default='./datasets/PiJAMA',
                       help='출력 디렉토리')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_pijama(args.output_dir)
