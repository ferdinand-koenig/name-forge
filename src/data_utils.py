# src/data_utils.py
import csv
import os
import random

from src.vocab import (adjectives, business_types, edge_descriptions,
                       generic_business_types, locations, nouns, purposes,
                       rare_adjectives, rare_nouns, suffixes, tlds,
                       unsafe_categories)

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


def generate_domain_from_description(description, n_domains=3):
    words = description.lower().split()
    adj_candidates = [
        w for w in words if w in adjectives + rare_adjectives
    ] or [random.choice(adjectives + rare_adjectives)]
    noun_candidates = [w for w in words if w in nouns + rare_nouns] or [
        random.choice(nouns + rare_nouns)
    ]
    bt_candidates = [
        w for w in words if w in business_types + generic_business_types
    ] or [random.choice(business_types + generic_business_types)]

    domains = set()
    attempts = 0
    while len(domains) < n_domains and attempts < 20:
        template = random.choice(domain_grammar)
        filled = []
        for token in template:
            if token == "{adj}":
                filled.append(random.choice(adj_candidates))
            elif token == "{noun}":
                filled.append(random.choice(noun_candidates))
            elif token == "{business_type}":
                filled.append(random.choice(bt_candidates))
            elif token == "{suffix}":
                filled.append(random.choice(suffixes))
        domain = generate_domain_from_template(filled)
        domains.add(domain)
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
    Generate a business description and domains.
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

    # Optionally use rare adjectives/nouns for variety
    if random.random() < edgy_fraction:
        adj = random.choice(rare_adjectives)
        noun = random.choice(rare_nouns)
    else:
        adj = random.choice(adj_pool)
        noun = random.choice(noun_pool)

    bt = random.choice(business_types)
    location = random.choice(locations)
    purpose = random.choice(purposes)

    if complexity == "simple":
        desc = f"{adj} {bt}"
    elif complexity == "medium":
        desc = f"{adj} {bt} in {location}"
    else:  # complex
        desc = f"{adj} {noun} {bt} in {location} {purpose}"

    domains = generate_domain_from_description(desc)
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
