"""
Download arXiv dataset from Kaggle and explore format
"""

import kagglehub
import json
import os
from pathlib import Path

def explore_dataset():
    """
    Download arXiv dataset and explore its format
    """
    print("="*60)
    print("DOWNLOADING ARXIV DATASET FROM KAGGLE")
    print("="*60)

    # Download latest version
    print("\nDownloading dataset...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")

    print(f"\n✅ Dataset downloaded to: {path}")

    # Explore the directory structure
    print("\n" + "="*60)
    print("EXPLORING DATASET STRUCTURE")
    print("="*60)

    dataset_path = Path(path)

    # List all files
    print(f"\nFiles in {dataset_path}:")
    all_files = list(dataset_path.glob("*"))
    for file in all_files[:20]:  # Show first 20 files
        size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
        file_type = "DIR" if file.is_dir() else "FILE"
        print(f"  [{file_type}] {file.name} ({size_mb:.2f} MB)")

    if len(all_files) > 20:
        print(f"  ... and {len(all_files) - 20} more files")

    # Look for JSON files
    json_files = list(dataset_path.glob("*.json"))
    jsonl_files = list(dataset_path.glob("*.jsonl"))

    print(f"\nFound {len(json_files)} JSON files")
    print(f"Found {len(jsonl_files)} JSONL files")

    # Examine first few papers from the first file
    print("\n" + "="*60)
    print("EXAMINING PAPER FORMAT")
    print("="*60)

    # Try to find and read a data file
    sample_file = None

    if jsonl_files:
        sample_file = jsonl_files[0]
        print(f"\nReading from JSONL: {sample_file.name}")

        with open(sample_file, 'r', encoding='utf-8') as f:
            print("\nFirst 3 papers:")
            for i, line in enumerate(f):
                if i >= 3:
                    break
                paper = json.loads(line)
                print(f"\n--- Paper {i+1} ---")
                print(f"Keys: {list(paper.keys())}")
                print(json.dumps(paper, indent=2)[:500] + "...")

    elif json_files:
        sample_file = json_files[0]
        print(f"\nReading from JSON: {sample_file.name}")

        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if isinstance(data, list):
                print(f"\nJSON contains a list with {len(data)} items")
                print("\nFirst 3 papers:")
                for i, paper in enumerate(data[:3]):
                    print(f"\n--- Paper {i+1} ---")
                    print(f"Keys: {list(paper.keys())}")
                    print(json.dumps(paper, indent=2)[:500] + "...")
            elif isinstance(data, dict):
                print(f"\nJSON contains a dict with keys: {list(data.keys())}")
                print(json.dumps(data, indent=2)[:1000] + "...")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset path: {path}")
    print(f"Total files: {len(all_files)}")
    print(f"JSON files: {len(json_files)}")
    print(f"JSONL files: {len(jsonl_files)}")

    if sample_file:
        print(f"\nSample file examined: {sample_file.name}")
        print(f"Size: {sample_file.stat().st_size / (1024 * 1024):.2f} MB")

    return path


if __name__ == "__main__":
    try:
        dataset_path = explore_dataset()
        print("\n✅ Exploration complete!")
        print(f"\nDataset location: {dataset_path}")
        print("\nNext steps:")
        print("1. Review the paper format above")
        print("2. Adapt the paper_evaluator to handle this format")
        print("3. Create evaluation script for arXiv papers")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
