# src/data_utils.py
import csv
import os
import random
import warnings

from src.vocab import (adjectives, business_types, edge_descriptions, nouns,
                       tlds)


# ------------------------------
# Helper Functions
# ------------------------------
def generate_business_description(
    complexity="simple", edgecase=False, unsafe=False
):
    def _gen_description():
        bt = random.choice(business_types)
        adj = random.choice(adjectives)
        noun = random.choice(nouns)

        if complexity == "simple":
            desc = f"{adj} {bt}"
        elif complexity == "medium":
            location = random.choice(
                ["downtown", "suburban area", "city center", "neighborhood"]
            )
            desc = f"{adj} {bt} in {location}"
        elif complexity == "complex":
            purpose = random.choice(
                [
                    "for busy professionals",
                    "with focus on sustainability",
                    "offering premium services",
                    "specializing in community events",
                ]
            )
            location = random.choice(
                ["downtown area", "near the river", "city center"]
            )
            desc = f"{adj} {noun} {bt} in {location} {purpose}"
        else:
            desc = f"{adj} {bt}"
        return desc

    if edgecase:
        desc = random.choice(edge_descriptions)
        if unsafe:
            warnings.warn("Overrode unsafe, due to edgecase flag set")
    elif unsafe:
        # Generate explicitly unsafe description
        desc = random.choice(
            [
                "adult content website with explicit nude content",
                "illegal gambling platform",
                "porn streaming service",
            ]
        )
        return desc, "__BLOCKED__"
    else:
        desc = _gen_description()

    # -----------------
    # Safe domain generation
    # -----------------
    domains = [generate_domain_name(desc) for _ in range(3)]
    return desc, domains


def generate_domain_name(description):
    words = description.lower().replace("in", "").replace("for", "").split()
    domain_words = words[:2] + [random.choice(nouns)]
    domain = "".join(c for c in domain_words if c.isalnum()) + random.choice(
        tlds
    )
    return domain


# ------------------------------
# Main Generator
# ------------------------------
def generate_dataset(
    num_entries=500, output_path="data/raw/dataset.csv", unsafe_fraction=0.0
):
    """Generate a CSV dataset of business descriptions and domain names."""

    # Ensure num_entries is an integer
    num_entries = int(num_entries)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["business_description", "domain_names", "complexity"],
        )
        writer.writeheader()

        print(f"generating {num_entries} entries")
        for _ in range(num_entries):
            complexity = random.choices(
                ["simple", "medium", "complex"], weights=[0.4, 0.4, 0.2]
            )[0]

            # Include edge cases ~5%
            desc, domains = generate_business_description(
                complexity,
                edgecase=(random.random() < 0.05),
                unsafe=(random.random() < unsafe_fraction),
            )

            # Write exactly one row per iteration
            writer.writerow(
                {
                    "business_description": desc,
                    "domain_names": domains,
                    "complexity": complexity,
                }
            )

    print(f"Dataset generated at {output_path} with {num_entries} entries")


# ------------------------------
# Generate Train/Test Datasets
# ------------------------------
def generate_train_test(train_size=500, test_size=100):
    generate_dataset(
        num_entries=train_size,
        output_path="data/raw/train_dataset.csv",
        unsafe_fraction=0.1,
    )
    generate_dataset(
        num_entries=test_size,
        output_path="data/raw/test_dataset.csv",
        unsafe_fraction=0.1,
    )
