#!/usr/bin/env python3
"""
MAESTRO 데이터셋 다운로드 스크립트
~200시간 고품질 피아노 MIDI + 오디오
"""

import argparse
import urllib.request
import zipfile
import os
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """다운로드 진행 표시"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """URL에서 파일 다운로드"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_maestro(output_dir: str, version: str = "v3.0.0"):
    """
    MAESTRO 데이터셋 다운로드

    Args:
        output_dir: 저장 디렉토리
        version: MAESTRO 버전 (v1.0.0, v2.0.0, v3.0.0)
    """
    print(f"🎹 MAESTRO {version} 다운로드 시작\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MAESTRO URL
    base_url = "https://storage.googleapis.com/magentadata/datasets/maestro"
    filename = f"maestro-{version}.zip"
    url = f"{base_url}/{version}/{filename}"

    zip_path = output_dir / filename
    extract_dir = output_dir / f"maestro-{version}"

    # 이미 다운로드 되었는지 확인
    if extract_dir.exists():
        print(f"✅ MAESTRO {version}이(가) 이미 존재합니다: {extract_dir}")
        print(f"   삭제하고 다시 다운로드하려면 디렉토리를 삭제하세요.")
        return

    # 다운로드
    print(f"📥 다운로드 중: {url}")
    print(f"   저장 위치: {zip_path}")
    print(f"   크기: ~100GB (MIDI + 오디오)\n")

    try:
        download_url(url, zip_path)
        print(f"\n✅ 다운로드 완료: {zip_path}")
    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")
        return

    # 압축 해제
    print(f"\n📦 압축 해제 중...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✅ 압축 해제 완료: {extract_dir}")
    except Exception as e:
        print(f"❌ 압축 해제 실패: {e}")
        return

    # ZIP 파일 삭제 (선택)
    delete_zip = input("\n원본 ZIP 파일을 삭제하시겠습니까? (y/n): ")
    if delete_zip.lower() == 'y':
        zip_path.unlink()
        print(f"🗑️  ZIP 파일 삭제: {zip_path}")

    # 통계
    midi_files = list(extract_dir.rglob("*.midi")) + list(extract_dir.rglob("*.mid"))
    print(f"\n📊 데이터셋 통계:")
    print(f"   MIDI 파일 수: {len(midi_files)}")
    print(f"   총 크기: ~100GB")
    print(f"   총 시간: ~200시간")

    print(f"\n🎉 MAESTRO 다운로드 완료!")
    print(f"\n사용 방법:")
    print(f"  python train.py --data_dir {extract_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Download MAESTRO dataset')
    parser.add_argument('--output_dir', type=str, default='./datasets/MAESTRO',
                       help='출력 디렉토리')
    parser.add_argument('--version', type=str, default='v3.0.0',
                       choices=['v1.0.0', 'v2.0.0', 'v3.0.0'],
                       help='MAESTRO 버전')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_maestro(args.output_dir, args.version)
