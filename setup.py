"""
🎵 음악 생성 AI - 설치 스크립트
SOTA 모델 학습 로드맵 (2025)
"""

from setuptools import setup, find_packages
import os

# README 읽기
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# requirements.txt 읽기
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="sota-music-generation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="SOTA 음악 생성 AI 학습 로드맵 - Music Informer, ImprovNet, Magenta RealTime",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sotaModel",
    packages=find_packages(exclude=["tests", "notebooks", "datasets"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "ui": [
            "gradio>=4.10.0",
            "streamlit>=1.29.0",
        ],
        "distributed": [
            "ray[tune]>=2.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "music-gen=utils.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
