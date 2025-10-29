#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Workspace - Dọn dẹp các file/thư mục không cần thiết trong dự án

- Xoá output tạm: output/clean_song_output, Audio_separator_ui/clean_song_output, temp_output/
- Xoá các thư mục *_mdx/ sinh ra bởi tách giọng
- Xoá __pycache__/ và *.pyc
- Giữ an toàn: mã nguồn trong src/, scripts/, tests/, assets/, config/, docs/, .git/

Chạy:
  python scripts/clean_workspace.py --dry-run   # chỉ liệt kê
  python scripts/clean_workspace.py             # thực thi xoá
"""

import os
import sys
import shutil
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAFE_DIRS = {
    'src', 'scripts', 'tests', 'assets', 'config', 'docs', '.git'
}

def remove_path(p: Path, dry_run: bool) -> None:
    if not p.exists():
        return
    if dry_run:
        print(f"DRY-RUN: would remove {p}")
        return
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"Removed: {p}")
    except Exception as e:
        print(f"Failed to remove {p}: {e}")

def clean(dry_run: bool = False) -> None:
    targets = []

    # 1) Thư mục output tạm
    targets += [
        PROJECT_ROOT / 'output' / 'clean_song_output',
        PROJECT_ROOT / 'Audio_separator_ui' / 'clean_song_output',
        PROJECT_ROOT / 'data' / 'temp_output',
        PROJECT_ROOT / 'temp_output',
    ]

    # 2) __pycache__ và *.pyc toàn dự án
    for p in PROJECT_ROOT.rglob('__pycache__'):
        targets.append(p)
    for p in PROJECT_ROOT.rglob('*.pyc'):
        targets.append(p)

    # 3) Thư mục *_mdx sinh ra bởi tách giọng trong clean_song_output
    for base in [
        PROJECT_ROOT / 'Audio_separator_ui' / 'clean_song_output',
        PROJECT_ROOT / 'output' / 'clean_song_output',
    ]:
        if base.exists():
            for sub in base.iterdir():
                if sub.is_dir() and sub.name.endswith('_mdx'):
                    targets.append(sub)

    # 4) Các file trung gian slice/export trong clean_song_output
    for base in [
        PROJECT_ROOT / 'Audio_separator_ui' / 'clean_song_output',
        PROJECT_ROOT / 'output' / 'clean_song_output',
    ]:
        if base.exists():
            for f in base.glob('*'):
                if f.is_file() and any(f.suffix.lower() == ext for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']):
                    targets.append(f)

    # Lọc trùng
    unique_targets = []
    seen = set()
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique_targets.append(t)

    # Bảo vệ: không xoá nhầm thư mục gốc/safe
    for t in unique_targets:
        try:
            rel = t.relative_to(PROJECT_ROOT)
        except ValueError:
            # ngoài project root
            continue
        parts = set(rel.parts)
        if any(part in SAFE_DIRS for part in parts) and rel.name not in {'clean_song_output', '__pycache__'}:
            # cho phép xoá clean_song_output và __pycache__ bên trong các SAFE_DIRS
            pass

    print(f"Cleaning {len(unique_targets)} target(s) (dry_run={dry_run})...")
    for t in unique_targets:
        remove_path(t, dry_run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Chỉ liệt kê, không xoá')
    args = parser.parse_args()
    clean(dry_run=args.dry_run)


