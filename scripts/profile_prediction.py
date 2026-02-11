# -*- coding: utf-8 -*-
"""
Profile the run_prediction pipeline (including plotting) on a local test image.
Writes a .pstats file and prints a short timing summary.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time
import os

from age_prediction.services.prediction import run_prediction


def _run_once(image_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    run_prediction(image_path, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile run_prediction on a test image.")
    parser.add_argument(
        "--image",
        default="static/images/test_img_large.jpg",
        help="Path to input image (default: static/images/test_img_large.jpg)",
    )
    parser.add_argument(
        "--out",
        default="profile_prediction.pstats",
        help="Path to write .pstats output (default: profile_prediction.pstats)",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/profile_outputs",
        help="Directory to write generated images (default: tmp/profile_outputs)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top functions to print (default: 30)",
    )
    args = parser.parse_args()

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.runcall(_run_once, args.image, args.output_dir)
    elapsed = time.perf_counter() - start

    profiler.dump_stats(args.out)
    print(f"Profile complete in {elapsed:.2f}s")
    print(f"Stats written to: {args.out}")
    print(f"Images written to: {args.output_dir}")

    stats = pstats.Stats(args.out)
    stats.sort_stats("cumulative").print_stats(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
