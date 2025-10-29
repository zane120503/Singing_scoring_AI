#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organize Files - Sắp xếp các file nằm ngoài vào đúng thư mục

- Di chuyển các file test ở root (test_*.py, *_test.py) vào thư mục tests/

Chạy:
  python scripts/organize_files.py --dry-run   # chỉ liệt kê
  python scripts/organize_files.py             # thực thi di chuyển
"""

import os
import shutil
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def move(src: Path, dst: Path, dry_run: bool):
    ensure_dir(dst.parent)
    if dry_run:
        print(f"DRY-RUN: would move {src.relative_to(PROJECT_ROOT)} -> {dst.relative_to(PROJECT_ROOT)}")
        return
    shutil.move(str(src), str(dst))
    print(f"Moved: {src.relative_to(PROJECT_ROOT)} -> {dst.relative_to(PROJECT_ROOT)}")


def organize(dry_run: bool = False):
    # 1) Di chuyển các file test ở root vào tests/
    tests_dir = PROJECT_ROOT / 'tests'
    ensure_dir(tests_dir)

    candidates = []
    for p in PROJECT_ROOT.glob('*.py'):
        name = p.name
        if name.startswith('test_') or name.endswith('_test.py'):
            candidates.append(p)

    # Loại trừ nếu file đã nằm trong tests/
    for src in candidates:
        dst = tests_dir / src.name
        if src.resolve().parent == tests_dir.resolve():
            continue
        move(src, dst, dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Chỉ liệt kê, không di chuyển')
    args = parser.parse_args()
    organize(dry_run=args.dry_run)


