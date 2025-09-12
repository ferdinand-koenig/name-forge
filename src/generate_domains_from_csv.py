# generate_domains_from_csv.py

import argparse

import pandas as pd

from .domain_generator import DomainGenerator
from .lib.local_llm_wrapper import LocalTransformersLLM


def generate_domains_from_csv(
    input_csv: str,
    output_csv: str,
    model_path: str,
    max_length: int = 50,
    temperature: float = 0.7,
):
    """
    Read a CSV with business descriptions, generate domains using a specified model,
    and save output.

    Args:
        input_csv (str): Path to CSV with 'business_description' column.
        output_csv (str): Path to save CSV with generated 'domain_names'.
        model_path (str): Path to GGUF model or other supported model.
        max_length (int): Max tokens for generation.
        temperature (float): Sampling temperature for creativity.
    """
    # Load CSV
    df = pd.read_csv(input_csv)

    # Initialize LLM and generator
    llm = LocalTransformersLLM(
        model_name=model_path,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
    )
    generator = DomainGenerator(llm=llm)

    # Generate domains
    domains_generated = []
    for idx, row in df.iterrows():
        desc = row["business_description"]
        generated = generator.generate(desc)
        domains_generated.append(generated)
        print(f"{idx+1}/{len(df)}: {desc} -> {generated}")

    # Save results
    df["domain_names"] = domains_generated
    df.to_csv(output_csv, index=False)
    print(f"✅ Domain generation completed. Output saved to {output_csv}")


# -----------------------------
# CLI interface
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate domains from CSV using an LLM"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV with 'business_description' column",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save output CSV with 'domain_names'",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to GGUF model or local LLM",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for creativity",
    )
    args = parser.parse_args()

    generate_domains_from_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_path=args.model_path,
        max_length=args.max_length,
        temperature=args.temperature,
    )
