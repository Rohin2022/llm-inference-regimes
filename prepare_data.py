import argparse
import json
from pathlib import Path

import pandas as pd


def report_to_text(report):
    """Recursively flatten a GovReport/CRS report into a string."""
    lines = []

    def add_section(section):
        title = section.get("section_title", "").strip()
        paragraphs = section.get("paragraphs", [])
        subsections = section.get("subsections", [])

        if title:
            lines.append(title)

        for paragraph in paragraphs:
            lines.append(paragraph.strip())

        for subsection in subsections:
            add_section(subsection)

    add_section(report)

    return "\n\n".join(lines)


def word_count(text):
    """Approximate word count using whitespace-separated tokens."""
    return len(text.split())


def load_crs_reports(reports_dir):
    """Load and process all CRS report JSON files into a DataFrame."""
    reports_dir = Path(reports_dir)
    rows = []

    for json_file in sorted(reports_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        report = report_to_text(data["reports"])
        summary = "\n".join(data["summary"])

        rows.append({
            "filename": json_file.name,
            "report": report,
            "summary": summary,
            "report_word_count": word_count(report),
            "summary_word_count": word_count(summary),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.set_index("filename")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Load and process CRS/GovReport JSON files."
    )
    parser.add_argument(
        "reports_path",
        type=str,
        help="Path to directory containing CRS report JSON files",
    )
    args = parser.parse_args()

    df = load_crs_reports(args.reports_path)

    # Save full dataset
    output_path = "crs_reports_parsed.csv"
    df.to_csv(output_path)

    # Select reports that are at least 10x longer than their summaries
    filtered_df = df[
        df["report_word_count"] >= 10 * df["summary_word_count"]
    ]

    # Randomly sample up to 100 reports
    sample_size = min(100, len(filtered_df))
    sample_df = filtered_df.sample(
        n=sample_size,
        random_state=42,
    )

    # Save smaller dataset
    sample_output_path = "crs_reports_sample_100.csv"
    sample_df.to_csv(sample_output_path)

    print(f"Loaded {len(df)} reports")
    print(f"Saved full dataset to {output_path}")

    print(
        f"Found {len(filtered_df)} reports with "
        "report >= 10x summary length"
    )
    print(f"Sampled {len(sample_df)} reports")
    print(f"Saved sample to {sample_output_path}")

    print("\nSample:")
    print(sample_df.head())


if __name__ == "__main__":
    main()