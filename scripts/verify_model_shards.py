"""Verify that all expected safetensor shards are present for a local model checkpoint.

Usage:
    python scripts/verify_model_shards.py --model-dir models/qwen2.5-14b-Instruct-model
"""
import argparse
import json
import os
import sys
from pathlib import Path


def verify_shards(model_dir: Path) -> bool:
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        print(f"ERROR: {model_dir} does not exist or is not a directory.")
        return False

    # Case 1: sharded model with index file
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        total_bytes = index.get("metadata", {}).get("total_size", None)
        expected_shards = sorted(set(index["weight_map"].values()))

        print(f"Model dir : {model_dir}")
        print(f"Index     : {index_path.name}")
        if total_bytes:
            print(f"Total size: {total_bytes / 1e9:.2f} GB expected")
        print(f"Shards    : {len(expected_shards)} expected\n")

        missing = []
        present = []
        for shard in expected_shards:
            shard_path = model_dir / shard
            if shard_path.exists():
                size = shard_path.stat().st_size
                present.append((shard, size))
                print(f"  [OK]     {shard}  ({size / 1e9:.2f} GB)")
            else:
                missing.append(shard)
                print(f"  [MISSING] {shard}")

        print()
        print(f"Result: {len(present)}/{len(expected_shards)} shards present")
        if missing:
            print(f"Missing  : {', '.join(missing)}")
            return False
        actual_bytes = sum(s for _, s in present)
        if total_bytes and abs(actual_bytes - total_bytes) > 1_000_000:
            print(f"WARNING: size mismatch — index says {total_bytes}, found {actual_bytes}")
            return False
        print("All shards present. Checkpoint looks complete.")
        return True

    # Case 2: single-file model
    single = model_dir / "model.safetensors"
    if single.exists():
        size = single.stat().st_size
        print(f"Model dir : {model_dir}")
        print(f"Single shard: {single.name}  ({size / 1e9:.2f} GB)")
        print("All shards present. Checkpoint looks complete.")
        return True

    print(f"ERROR: no model.safetensors.index.json or model.safetensors found in {model_dir}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Verify safetensor shards for a local model checkpoint.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Path to the model directory")
    args = parser.parse_args()

    ok = verify_shards(args.model_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
