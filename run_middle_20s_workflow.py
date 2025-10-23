#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI to run Optimized Full Audio Workflow (No Slicing)
Usage:
  python run_middle_20s_workflow.py "<karaoke_file>" "<beat_file>"
"""

import sys
import os
import json

from optimized_middle_workflow import run_workflow

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_middle_20s_workflow.py <karaoke_file> <beat_file>")
        sys.exit(1)

    karaoke_file = sys.argv[1]
    beat_file = sys.argv[2]

    out_dir = os.path.join(os.path.dirname(__file__), 'output', 'full_audio_workflow')
    result = run_workflow(karaoke_file, beat_file, duration=0.0, output_dir=out_dir)

    if not result.get('success'):
        print("\n=== KET QUA ===")
        print("FAIL")
        print("Error:", result.get('error'))
        sys.exit(1)

    print("\n=== KET QUA ===")
    print("SUCCESS")
    print("Vocals (src):", result['vocals_src'])
    print("Vocals (export):", result['vocals_export'])
    print("Vocals key:", result['vocals_key']['key'], "conf:", f"{result['vocals_key']['confidence']:.3f}")
    print("Beat key:", result['beat_key']['key'], "conf:", f"{result['beat_key']['confidence']:.3f}")
    print("Key match:", result['key_compare']['match'])
    print("Key similarity:", f"{result['key_compare']['similarity']:.3f}")
    print("Key score:", f"{result['key_compare']['score']:.2f}")

if __name__ == '__main__':
    main()


