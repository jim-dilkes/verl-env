#!/usr/bin/env python3
"""
predownload_model.py

Usage:
  python predownload_model.py --model Qwen/Qwen3-4B-Instruct-2507 [--revision main]
"""
import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Pre-download a HF model into cache.")
    parser.add_argument("--model", required=True, help="Model repo ID, e.g. Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default=None, help="Optional branch/tag/commit")
    parser.add_argument("--cache-dir", default=None, help="Override HF_HOME cache dir if desired")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        resume_download=True,
        local_files_only=False,
    )
    print(f"Cached files at: {path}")


if __name__ == "__main__":
    main()