import re
from pathlib import Path

import pandas as pd

# Folder with your judged CSVs
base = Path("outputs/judged")

# Grab files
human_files = [f for f in base.glob("mistral_v*-human.csv")]
domain_files = [f for f in base.glob("mistral_7B_*.csv")]


def extract_version(name: str) -> str:
    """Extract version string (e.g. v1.0, v2.1) from filename."""
    match = re.search(r"v\d+\.\d+", name)
    return match.group(0) if match else None


results = []

for hfile in human_files:
    hname = hfile.stem
    hver = extract_version(hname)
    if not hver:
        continue

    human_df = pd.read_csv(hfile)

    # find matching domain file
    for dfile in domain_files:
        dname = dfile.stem
        dver = extract_version(dname)

        if dver != hver:
            continue  # only compare matching versions

        domain_df = pd.read_csv(dfile)

        merged = pd.merge(
            human_df,
            domain_df,
            on=["business_description", "domain_name"],
            suffixes=("_human", "_model"),
        )

        if merged.empty:
            continue

        for metric in ["relevance", "diversity", "originality"]:
            corr = merged[f"{metric}_human"].corr(merged[f"{metric}_model"])
            results.append(
                {"version": hver, "metric": metric, "pearson_corr": corr}
            )

# Build results DataFrame
results_df = pd.DataFrame(results)

# Pivot to wide format: versions as rows, metrics as columns
summary = results_df.pivot(
    index="version", columns="metric", values="pearson_corr"
).reset_index()

print(summary)

# Save
summary.to_csv(base / "correlation_summary_pivot.csv", index=False)
