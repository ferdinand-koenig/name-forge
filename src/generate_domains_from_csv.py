# generate_domains_from_csv.py

import argparse

import pandas as pd

from domain_generator import generate_domains


def run_pipeline(input_csv: str, output_csv: str):
    """
    Read a CSV with business descriptions, generate domains, and save output.
    Expects a column named 'business_description' and an empty 'domain_names' column.
    """
    df = pd.read_csv(input_csv)

    # Generate domains for each description
    domains_generated = []
    for idx, row in df.iterrows():
        desc = row["business_description"]
        generated = generate_domains(desc)
        domains_generated.append(generated)
        print(f"{idx+1}/{len(df)}: {desc} -> {generated}")

    df["domain_names"] = domains_generated
    df.to_csv(output_csv, index=False)
    print(f"Domain generation completed. Output saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate domains from CSV")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="test_dataset_sample.csv",
        help="Path to the input CSV with business descriptions",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="test_dataset_sample_filled.csv",
        help="Path to save the output CSV with generated domains",
    )
    args = parser.parse_args()

    run_pipeline(args.input_csv, args.output_csv)
