#!/usr/bin/env python3
"""
FACTR Repository Interrogation Script
------------------------------------
Generates a structured overview of the repository for documentation
and README generation.

Outputs:
- repo_summary.json
- repo_summary.md
"""

import os
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(".").resolve()

EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".idea", ".vscode"
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".log", ".tmp"
}

def walk_repo():
    tree = []
    stats = defaultdict(int)
    file_types = defaultdict(int)

    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        rel_root = Path(root).relative_to(REPO_ROOT)

        for f in files:
            ext = Path(f).suffix.lower()
            if ext in EXCLUDE_EXTENSIONS:
                continue

            path = rel_root / f
            size = (Path(root) / f).stat().st_size

            tree.append({
                "path": str(path),
                "extension": ext or "no_extension",
                "size_bytes": size
            })

            stats["total_files"] += 1
            stats["total_size_bytes"] += size
            file_types[ext or "no_extension"] += 1

    return tree, stats, file_types


def detect_key_files(tree):
    keywords = {
        "entry_points": ["app", "main", "streamlit"],
        "evaluation": ["eval", "metric", "score"],
        "kb": ["kb", "faiss", "embedding"],
        "notebooks": [".ipynb"],
        "docker": ["dockerfile"],
        "docs": ["readme", "technical", ".pdf"]
    }

    findings = defaultdict(list)

    for item in tree:
        p = item["path"].lower()
        for category, terms in keywords.items():
            if any(t in p for t in terms):
                findings[category].append(item["path"])

    return findings


def generate_markdown(stats, file_types, findings):
    lines = []

    lines.append("# FACTR Repository Interrogation Report\n")
    lines.append("## Repository Statistics\n")
    lines.append(f"- Total files: **{stats['total_files']}**")
    lines.append(f"- Total size: **{stats['total_size_bytes'] / 1_048_576:.2f} MB**\n")

    lines.append("## File Types Breakdown\n")
    for ext, count in sorted(file_types.items(), key=lambda x: -x[1]):
        lines.append(f"- `{ext}` : {count}")

    lines.append("\n## Detected Key Components\n")
    for section, items in findings.items():
        lines.append(f"### {section.replace('_', ' ').title()}")
        for i in sorted(set(items)):
            lines.append(f"- {i}")
        lines.append("")

    return "\n".join(lines)


def main():
    tree, stats, file_types = walk_repo()
    findings = detect_key_files(tree)

    # JSON output
    with open("repo_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "stats": stats,
            "file_types": file_types,
            "findings": findings,
            "files": tree
        }, f, indent=2)

    # Markdown output
    md = generate_markdown(stats, file_types, findings)
    with open("repo_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("✅ Repository interrogation complete.")
    print("Generated:")
    print(" - repo_summary.json")
    print(" - repo_summary.md")


if __name__ == "__main__":
    main()
