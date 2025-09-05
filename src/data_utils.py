# src/data_utils.py
import csv
import os
import random

from src.vocab import (FORBIDDEN_WORDS, adjectives, business_types,
                       edge_descriptions, nouns, tlds)


# ------------------------------
# Helper Functions
# ------------------------------
def generate_business_description(complexity="simple"):
    bt = random.choice(business_types)
    adj = random.choice(adjectives)
    random.choice(nouns)

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
        location = random.choice(["downtown area", "near the river", "city center"])
        desc = f"{adj} {bt} in {location} {purpose}"
    else:
        desc = f"{adj} {bt}"
    return desc


def generate_domain_name(description):
    words = description.lower().replace("in", "").replace("for", "").split()
    domain_words = words[:2] + [random.choice(nouns)]
    domain = "".join(c for c in domain_words if c.isalnum()) + random.choice(tlds)
    return domain


def is_safe_domain(domain):
    domain_lower = domain.lower()
    return not any(word in domain_lower for word in FORBIDDEN_WORDS)


# ------------------------------
# Main Generator
# ------------------------------
def generate_dataset(
    num_entries=500, output_path="data/raw/dataset.csv", unsafe_fraction=0.0
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["business_description", "domain_name", "complexity", "is_safe"],
        )
        writer.writeheader()

        for _ in range(num_entries):
            complexity = random.choices(
                ["simple", "medium", "complex"], weights=[0.4, 0.4, 0.2]
            )[0]
            desc = generate_business_description(complexity)

            # Include edge cases ~5%
            if random.random() < 0.05:
                desc = random.choice(edge_descriptions)

            # Unsafe injection (only if allow_unsafe=True)
            if random.random() < unsafe_fraction:
                random.choice(["xxxhotstuff.net", "gambling-now.io"])
                writer.writerow(
                    {
                        "business_description": desc,
                        "domain_name": "__BLOCKED__",  # ground truth expected output
                        "complexity": complexity,
                        "is_safe": False,
                    }
                )
                continue

            # Normal safe case
            domain = generate_domain_name(desc)

            # Safety check
            if not is_safe_domain(domain):
                # If the random generator accidentally
                # produces a bad word, sanitize
                domain = "__BLOCKED__"
                safe = False
            else:
                safe = True

            writer.writerow(
                {
                    "business_description": desc,
                    "domain_name": domain,
                    "complexity": complexity,
                    "is_safe": safe,
                }
            )

    print(f"Dataset generated at {output_path} with {num_entries} entries")


# ------------------------------
# Generate Train/Test Datasets
# ------------------------------
def generate_train_test(train_size=500, test_size=100):
    generate_dataset(num_entries=train_size, output_path="data/raw/train_dataset.csv")
    generate_dataset(
        num_entries=test_size,
        output_path="data/raw/test_dataset.csv",
        unsafe_fraction=0.1,
    )
