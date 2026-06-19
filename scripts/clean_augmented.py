#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Clean all augmented files (*_augmented.mp4/wav/csv) from a dataset directory.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory to clean.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the files that would be deleted, without actually deleting them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"Error: Directory {args.dataset_dir} does not exist.")
        sys.exit(1)

    print(f"Scanning {args.dataset_dir} for augmented files...")
    
    # Locate all files ending with _augmented.mp4, _augmented.wav, _augmented.csv
    extensions = ["*.mp4", "*.wav", "*.csv"]
    augmented_files = []
    for ext in extensions:
        augmented_files.extend(list(args.dataset_dir.rglob(f"*_augmented{ext[1:]}")))

    if not augmented_files:
        print("No augmented files found.")
        sys.exit(0)

    print(f"Found {len(augmented_files)} augmented files.")

    if args.dry_run:
        print("\n[Dry Run] The following files would be deleted:")
        for file_path in sorted(augmented_files):
            print(f"  - {file_path}")
        print(f"\nDry run complete. No files were deleted.")
        sys.exit(0)

    deleted_count = 0
    for file_path in augmented_files:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    print(f"\nSuccessfully deleted {deleted_count} of {len(augmented_files)} augmented files.")


if __name__ == "__main__":
    main()
