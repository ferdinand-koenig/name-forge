import ast
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import wordninja  # pip install wordninja
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
input_folder = "./outputs"
output_folder = "./outputs/judged"
os.makedirs(output_folder, exist_ok=True)

# Load embedding model (CPU-friendly)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Weights for overall score (sum should be 1)
WEIGHTS = {"relevance": 1 / 3, "diversity": 1 / 3, "originality": 1 / 3}


def tokenize_domain(domain):
    """Split concatenated domains into words using wordninja"""
    base = domain.split(".")[0]  # remove TLD
    tokens = wordninja.split(base.lower())
    return set(tokens)


def lexical_overlap(desc, domain):
    """
    Compute lexical overlap between business description and domain.
    Returns a fraction of overlapping words (Jaccard similarity)
    """
    desc_tokens = set(re.findall(r"\w+", desc.lower()))
    domain_tokens = tokenize_domain(domain)
    overlap = len(desc_tokens & domain_tokens) / max(
        1, len(desc_tokens | domain_tokens)
    )
    return overlap


def compute_metrics(desc, domains):
    """
    desc: str
    domains: list[str]
    Returns:
        relevances: np.array of cosine similarities
        diversity: float
        originality: np.array of 1 - lexical overlap
    """
    desc_embedding = model.encode(desc, show_progress_bar=False)
    embeddings = model.encode(domains, show_progress_bar=False)

    # Relevance: cosine similarity
    relevances = cosine_similarity(embeddings, [desc_embedding]).flatten()

    # Diversity: avg pairwise cosine distance across all domains in this row
    n = len(domains)
    if n > 1:
        pairwise_sim = cosine_similarity(embeddings)
        diversity = np.sum(1 - pairwise_sim) / (n * (n - 1))
    else:
        diversity = 0.0

    # Originality: 1 - lexical overlap
    originality = np.array([1 - lexical_overlap(desc, d) for d in domains])

    return relevances, diversity, originality


summary_records = []

# Process each CSV file
for file_path in glob(os.path.join(input_folder, "*.csv")):
    df = pd.read_csv(file_path)
    expanded_rows = []

    for _, row in df.iterrows():
        desc = row["business_description"]
        try:
            domains = ast.literal_eval(row["domain_names"])
        except Exception:
            print(f"Could not parse domains in row: {row['domain_names']}")
            domains = []

        if not isinstance(domains, list):
            continue

        relevances, diversity, originality = compute_metrics(desc, domains)

        # Expand: one row per domain
        for domain, rel, orig in zip(domains, relevances, originality):
            overall_score = (
                WEIGHTS["relevance"] * rel
                + WEIGHTS["diversity"] * diversity
                + WEIGHTS["originality"] * orig
            )
            expanded_rows.append(
                {
                    "business_description": desc,
                    "domain_name": domain,
                    "complexity": row.get("complexity", None),
                    "relevance": float(rel),
                    "diversity": float(diversity),
                    "originality": float(orig),
                    "overall_score": float(overall_score),
                }
            )

    # Save judged CSV
    judged_df = pd.DataFrame(expanded_rows)
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    judged_df.to_csv(output_path, index=False)

    # Summary stats per file
    if not judged_df.empty:
        summary_records.append(
            {
                "model_csv": os.path.basename(file_path),
                "mean_relevance": judged_df["relevance"].mean(),
                "median_relevance": judged_df["relevance"].median(),
                "mean_diversity": judged_df["diversity"].mean(),
                "median_diversity": judged_df["diversity"].median(),
                "mean_originality": judged_df["originality"].mean(),
                "median_originality": judged_df["originality"].median(),
                "mean_overall_score": judged_df["overall_score"].mean(),
                "median_overall_score": judged_df["overall_score"].median(),
            }
        )

# Save global summary
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(output_folder, "summary.csv"), index=False)

print("Done! Judged CSVs saved in /outputs/judged and summary.csv created.")
