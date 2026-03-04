#!/usr/bin/env python3
"""
Clean golden_samples.jsonl: remove dot-leader noise (......), collapse spaces,
and strip control characters from the "text" field of each JSON object.
"""
import json
import re
import sys

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Replace runs of 3+ dots (dot leaders / ellipsis noise) with a single space
    # old: "...." or ".............."  new: " "
    text = re.sub(r"\.{3,}", " ", text)
    # Collapse multiple spaces (and tabs) to a single space
    # old: "  " or "   \t  "  new: " "
    text = re.sub(r"[ \t]+", " ", text)
    # Remove control characters (e.g. \u0007)
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\r\t")
    # Normalize newlines: optional trailing spaces before newline
    text = re.sub(r" +\n", "\n", text)
    # Strip leading/trailing whitespace from each line
    text = "\n".join(line.strip() for line in text.splitlines())
    # Strip leading/trailing from whole text
    return text.strip()


def main():
    input_path = "/Users/vishnu/Downloads/golden_samples.jsonl"
    output_path = "/Users/vishnu/Downloads/golden_samples_cleaned.jsonl"
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    cleaned = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skip invalid JSON: {e}", file=sys.stderr)
                continue
            if "text" in obj:
                old_text = obj["text"]
                obj["text"] = clean_text(old_text)
                if old_text != obj["text"]:
                    cleaned += 1
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Cleaned {cleaned} records. Output: {output_path}")


if __name__ == "__main__":
    main()
