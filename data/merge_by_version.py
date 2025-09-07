import argparse
import csv
import glob
import os
import shutil

# Paths
RAW_DIR = "raw"  # relative to script location
OUTPUT_DIR = "."  # current folder, i.e., /data/


def merge_train_files(version):
    """
    Merge all train dataset files that contain the given version into a single CSV,
    preserving all rows even if they contain commas inside quotes.
    """
    pattern = os.path.join(RAW_DIR, f"train_dataset-v{version}*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No train dataset files found for version v{version}.")
        return

    print(f"Found train files: {files}")

    output_file = os.path.join(OUTPUT_DIR, "train_dataset.csv")
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

    print(f"Merged train dataset saved to {output_file}.")


def copy_test_file(version):
    """
    Copy the test dataset file of a specific version to the output folder
     without version suffix.
    """
    pattern = os.path.join(RAW_DIR, f"test_dataset-v{version}*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No test dataset file found for version v{version}.")
        return
    src_file = files[0]  # pick the first match
    output_file = os.path.join(OUTPUT_DIR, "test_dataset.csv")
    shutil.copy(src_file, output_file)
    print(f"Test dataset copied to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy/merge dataset files by version."
    )
    parser.add_argument("version", type=str, help="Version number (e.g., 2.0)")
    args = parser.parse_args()

    version = args.version

    # Copy test dataset
    copy_test_file(version)

    # Merge train datasets of the given version
    merge_train_files(version)
