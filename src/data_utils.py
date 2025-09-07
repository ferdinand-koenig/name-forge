# src/data_utils.py
import csv
import os
import random

from src.vocab import (adjectives, business_type_keywords, business_types,
                       edge_descriptions, generic_business_types, locations,
                       nouns, purposes, rare_adjectives, rare_nouns, suffixes,
                       tlds, unsafe_categories)

# ------------------------------
# Grammar templates for domain generation
# ------------------------------
domain_grammar = [
    ["{adj}", "{noun}"],
    ["{adj}", "{business_type}"],
    ["{noun}", "{suffix}"],
    ["{adj}", "{noun}", "{suffix}"],
    ["{adj}", "{business_type}", "{suffix}"],
]

connectors = ["", "-", "hub", "corner", "works"]


# ------------------------------
# Helpers
# ------------------------------
def clean_word(word):
    return "".join(c for c in word.lower() if c.isalnum())


def generate_domain_from_template(template_words):
    domain_base = "".join(clean_word(w) for w in template_words)
    domain_suffix = random.choice(suffixes + [""])
    tld = random.choice(tlds)
    return domain_base + domain_suffix + tld


# ------------------------------
# Domain generator
# ------------------------------
def generate_domain_from_description(
    description, business_type=None, n_domains=3
):
    """
    Generate 2–3 relevant domain names using business_type_keywords.
    """
    words = description.lower().split()
    domains = set()
    attempts = 0

    # Determine business type if not given
    if not business_type or business_type not in business_type_keywords:
        for bt in business_type_keywords:
            if bt in description.lower():
                business_type = bt
                break
    if not business_type or business_type not in business_type_keywords:
        business_type = random.choice(list(business_type_keywords.keys()))

    bt_keywords = business_type_keywords[business_type]

    while len(domains) < n_domains and attempts < 50:
        # Pick 1 word from description + 1 from business type keywords
        main_word = random.choice(
            [w for w in words if w.isalpha()] + [random.choice(bt_keywords)]
        )
        secondary_word = random.choice(bt_keywords)
        connector = random.choice(connectors) if random.random() < 0.5 else ""

        # Build domain
        domain_base = (
            clean_word(main_word) + connector + clean_word(secondary_word)
        )
        if len(domain_base) > 25:
            domain_base = domain_base[:25]

        tld = random.choice(tlds)
        domains.add(domain_base + tld)
        attempts += 1

    return list(domains)


# ------------------------------
# Unsafe description generator
# ------------------------------
def generate_unsafe_description():
    """Mix-and-match unsafe categories to produce varied unsafe descriptions"""
    category = random.choice(list(unsafe_categories.keys()))
    phrase = random.choice(unsafe_categories[category])
    bt = random.choice(generic_business_types)
    return f"{phrase} {bt}"


# ------------------------------
# Business description generator
# ------------------------------
def generate_business_description(
    complexity="simple", unsafe=False, edgy_fraction=0.2
):
    """
    Generate a business description, business type, and domains.
    - complexity: simple / medium / complex
    - unsafe: True → blocked output
    - edgy_fraction: fraction of safe examples using rare adjectives/nouns
    """
    if unsafe:
        desc = generate_unsafe_description()
        domains = ["__BLOCKED__"] * 3
        return desc, domains

    # Edge cases ~5%
    if random.random() < 0.05:
        desc = random.choice(edge_descriptions)
        domains = generate_domain_from_description(desc)
        return desc, domains

    # Safe description
    adj_pool = adjectives + rare_adjectives
    noun_pool = nouns + rare_nouns

    if random.random() < edgy_fraction:
        adj = random.choice(rare_adjectives)
        noun = random.choice(rare_nouns)
    else:
        adj = random.choice(adj_pool)
        noun = random.choice(noun_pool)

    # Pick business type from merged list
    bt = random.choice(business_types)
    location = random.choice(locations)
    purpose = random.choice(purposes)

    # Build description based on complexity
    if complexity == "simple":
        desc = f"{adj} {bt}"
    elif complexity == "medium":
        desc = f"{adj} {bt} in {location}"
    else:  # complex
        desc = f"{adj} {noun} {bt} in {location} {purpose}"

    # Generate domains using business type
    domains = generate_domain_from_description(desc, business_type=bt)
    return desc, domains


# ------------------------------
# Dataset generation
# ------------------------------
def generate_dataset(
    num_entries=500, output_path="data/raw/dataset.csv", unsafe_fraction=0.1
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["business_description", "domain_names", "complexity"],
        )
        writer.writeheader()

        for _ in range(num_entries):
            complexity = random.choices(
                ["simple", "medium", "complex"], weights=[0.4, 0.4, 0.2]
            )[0]
            desc, domains = generate_business_description(
                complexity, unsafe=(random.random() < unsafe_fraction)
            )
            writer.writerow(
                {
                    "business_description": desc,
                    "domain_names": domains,
                    "complexity": complexity,
                }
            )

    print(f"Dataset generated at {output_path} ({num_entries} entries)")


# ------------------------------
# Train/test split
# ------------------------------
def generate_train_test(train_size=500, test_size=100):
    generate_dataset(
        train_size, "data/raw/train_dataset.csv", unsafe_fraction=0.1
    )
    generate_dataset(
        test_size, "data/raw/test_dataset.csv", unsafe_fraction=0.1
    )


# ------------------------------
# Run standalone
# ------------------------------
if __name__ == "__main__":
    generate_train_test()
