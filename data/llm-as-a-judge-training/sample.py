import argparse

import pandas as pd


def sample_test_dataset(input_csv, output_csv, n_samples, random_seed=None):
    """
    Sample a sub-dataset from the test CSV without replacement and
    create empty columns for evaluation criteria.

    Args:
        input_csv (str): Path to the original test_dataset.csv
        output_csv (str): Path to save the sampled CSV
        n_samples (int): Number of rows to sample
        random_seed (int, optional): Seed for reproducibility
    """
    # Load the full test dataset
    df = pd.read_csv(input_csv)

    # Sample without replacement
    sampled_df = df.sample(
        n=n_samples, replace=False, random_state=random_seed
    ).copy()

    # Empty the domain_names column
    if "domain_names" in sampled_df.columns:
        sampled_df["domain_names"] = ""

    # Add empty columns for evaluation
    eval_columns = ["Memorability", "Clarity", "Brandability", "Safety"]
    for col in eval_columns:
        sampled_df[col] = ""

    # Save sampled dataset
    sampled_df.to_csv(output_csv, index=False)
    print(
        f"Sampled {n_samples} rows saved to {output_csv} with empty evaluation fields."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample sub-dataset from test_dataset.csv for evaluation."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="../test_dataset.csv",
        help="Path to test_dataset.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="test_dataset_sample.csv",
        help="Path to save sampled CSV",
    )
    parser.add_argument(
        "--n_samples", type=int, default=30, help="Number of samples to pick"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    sample_test_dataset(
        args.input_csv, args.output_csv, args.n_samples, args.seed
    )
