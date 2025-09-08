import argparse
import csv
import glob
import os

# Paths
RAW_DIR = "raw"  # relative to script location
OUTPUT_DIR = "."  # current folder, i.e., /data/


def merge_files(pattern, output_file):
    """
    Merge all CSV files matching the pattern into a single CSV.
    Preserves all rows even if they contain commas inside quotes.
    """
    files = glob.glob(pattern)
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    print(f"Found files: {files}")

    with open(output_file, "w", newline="", encoding="utf-8") as out_f:
        writer = None

        for f in files:
            with open(f, newline="", encoding="utf-8") as in_f:
                reader = csv.reader(in_f, quotechar='"')
                headers = next(reader)

                if writer is None:
                    writer = csv.writer(
                        out_f, quotechar='"', quoting=csv.QUOTE_ALL
                    )
                    writer.writerow(headers)  # write header once

                for row in reader:
                    writer.writerow(row)

    print(f"Merged dataset saved to {output_file}.")


def merge_train_files(version):
    """Merge all train dataset files of a given version."""
    pattern = os.path.join(RAW_DIR, f"train_dataset-v{version}*.csv")
    output_file = os.path.join(OUTPUT_DIR, "train_dataset.csv")
    merge_files(pattern, output_file)


def merge_test_files(version):
    """Merge all test dataset files of a given version."""
    pattern = os.path.join(RAW_DIR, f"test_dataset-v{version}*.csv")
    output_file = os.path.join(OUTPUT_DIR, "test_dataset.csv")
    merge_files(pattern, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge dataset files (train & test) by version."
    )
    parser.add_argument("version", type=str, help="Version number (e.g., 2.0)")
    args = parser.parse_args()

    version = args.version

    # Merge train datasets
    merge_train_files(version)

    # Merge test datasets
    merge_test_files(version)
