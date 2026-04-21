"""
agent.py — CLI entry point for Gemma 4 Smart Gallery.

Bypasses Gradio for direct workflow execution from the terminal.

Usage:
    python agent.py describe --folder ./images
    python agent.py search   --folder ./images --query "ocean"
"""

import argparse
import sys

import workflows


def cmd_describe(folder: str) -> None:
    print(f"Generating descriptions for images in: {folder}")
    results = workflows.run_describe_workflow(folder)
    if not results:
        print("No supported images found.")
        return
    for r in results:
        print(f"\n{r['path']}")
        print(f"  {r['description']}")
    print(f"\nDone — {len(results)} images described and cached.")


def cmd_search(folder: str, query: str) -> None:
    print(f'Searching for "{query}" in: {folder}')
    matches = workflows.run_search_workflow(folder, query)
    if not matches:
        print(f'No images matched "{query}".')
        return
    print(f"\n{len(matches)} match(es):")
    for path in matches:
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 Smart Gallery CLI")
    sub    = parser.add_subparsers(dest="command", required=True)

    desc_parser = sub.add_parser("describe", help="Generate descriptions for all images in a folder")
    desc_parser.add_argument("--folder", required=True, help="Path to image folder")

    search_parser = sub.add_parser("search", help="Search images by keyword")
    search_parser.add_argument("--folder", required=True, help="Path to image folder")
    search_parser.add_argument("--query",  required=True, help="Search keyword or phrase")

    args = parser.parse_args()

    if args.command == "describe":
        cmd_describe(args.folder)
    elif args.command == "search":
        cmd_search(args.folder, args.query)


if __name__ == "__main__":
    main()
