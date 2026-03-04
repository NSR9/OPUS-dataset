#!/usr/bin/env python3
"""Convert golden_samples.jsonl to a mobile-friendly HTML page."""
import json
import html
import re
from collections import defaultdict
from pathlib import Path

JSONL = Path(__file__).resolve().parent.parent / "examples" / "golden_samples.jsonl"
OUT   = Path(__file__).resolve().parent.parent / "examples" / "golden_samples.html"

# Map tags → human-readable domain names (all 37 tags from golden_samples.jsonl)
TAG_TO_DOMAIN = {
    # Math (14 samples)
    "math_reasoning": "Math",
    "math_competition": "Math",
    "math_hard": "Math",
    "science_math": "Math",
    # Code (14 samples)
    "code_generation": "Code",
    "software_engineering": "Code",
    # Knowledge (14 samples)
    "general_knowledge": "Knowledge",
    "science": "Knowledge",
    "benchmark_qa": "Knowledge",
    "general_qa": "Knowledge",
    # Preference & Reasoning (14 samples)
    "reasoning": "Preference & Reasoning",
    "general_preference": "Preference & Reasoning",
    "post_training": "Preference & Reasoning",
    "general_conversation": "Preference & Reasoning",
    "human_preferred": "Preference & Reasoning",
    "high_reasoning": "Preference & Reasoning",
    # Tool Use (14 samples)
    "function_calling": "Tool Use",
    "tool_use": "Tool Use",
    # Instruction Following (14 samples)
    "instruction_following": "Instruction Following",
    # Truthfulness (14 samples)
    "truthfulness": "Truthfulness",
    # Indic (16 samples)
    "indic_hindi": "Indic",
    "indic_tamil": "Indic",
    "indic_telugu": "Indic",
    "indic_bengali": "Indic",
    "indic_kannada": "Indic",
    "indic_instruction": "Indic",
    "indic_hindi_qa": "Indic",
    "indic_tamil_qa": "Indic",
    "indic_telugu_qa": "Indic",
    "indic_hindi_nlu": "Indic",
    "indic_tamil_nlu": "Indic",
    "indic_cultural_knowledge": "Indic",
    "indic_fairness": "Indic",
    # Long Context (14 samples)
    "long_context_qa": "Long Context",
    "long_context_narrative": "Long Context",
    "long_context_retrieval": "Long Context",
    "long_context_multihop": "Long Context",
}

def tag_color(domain: str) -> str:
    colors = {
        "Math": "#4CAF50",
        "Code": "#2196F3",
        "Knowledge": "#FF9800",
        "Preference & Reasoning": "#9C27B0",
        "Tool Use": "#F44336",
        "Instruction Following": "#00BCD4",
        "Truthfulness": "#795548",
        "Indic": "#E91E63",
        "Long Context": "#607D8B",
    }
    return colors.get(domain, "#999")

def format_text(text: str) -> str:
    """Format the raw text, preserving [USER]/[ASSISTANT] structure."""
    escaped = html.escape(text)
    # Highlight [USER] and [ASSISTANT] markers
    escaped = re.sub(
        r'\[USER\]',
        '<span class="role-marker user-marker">[USER]</span>',
        escaped,
    )
    escaped = re.sub(
        r'\[ASSISTANT\]',
        '<span class="role-marker asst-marker">[ASSISTANT]</span>',
        escaped,
    )
    return escaped

def main():
    samples = []
    with open(JSONL) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # Group by domain
    domains = defaultdict(list)
    for s in samples:
        domain = TAG_TO_DOMAIN.get(s["tag"], "Other")
        domains[domain].append(s)

    domain_order = [
        "Math", "Code", "Knowledge", "Preference & Reasoning",
        "Tool Use", "Instruction Following", "Truthfulness",
        "Indic", "Long Context",
    ]

    # Build domain sections
    domain_html = []
    for domain in domain_order:
        items = domains.get(domain, [])
        if not items:
            continue
        color = tag_color(domain)
        cards = []
        for s in items:
            cards.append(f'''
        <div class="card" style="border-left-color:{color}">
          <div class="field">
            <span class="field-label">ID</span>
            <span class="field-value id-value">{s["id"]}</span>
          </div>
          <div class="field">
            <span class="field-label">Tag</span>
            <span class="field-value"><span class="tag" style="background:{color}22;color:{color};border:1px solid {color}44">{s["tag"]}</span></span>
          </div>
          <div class="field text-field">
            <span class="field-label">Text</span>
            <div class="field-value text-value">{format_text(s["text"])}</div>
          </div>
        </div>''')

        domain_html.append(f'''
    <div class="domain-section">
      <details class="domain" open>
        <summary>
          <span class="domain-badge" style="background:{color}">{domain}</span>
          <span class="count">{len(items)} samples</span>
        </summary>
        <div class="domain-body">{"".join(cards)}</div>
      </details>
    </div>''')

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OPUS Golden Dataset — 128 Samples</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5; color: #333; padding: 12px; max-width: 800px; margin: 0 auto;
    -webkit-text-size-adjust: 100%;
  }}
  h1 {{ font-size: 1.3rem; text-align: center; padding: 16px 0 4px; }}
  .subtitle {{ text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 16px; }}
  .stats {{
    display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;
    margin-bottom: 20px;
  }}
  .stat {{
    background: #fff; border-radius: 8px; padding: 8px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 0.8rem; text-align: center;
  }}
  .stat b {{ display: block; font-size: 1.2rem; color: #1976D2; }}
  .domain-section {{ margin-bottom: 12px; }}
  .domain > summary {{
    list-style: none; cursor: pointer; display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; background: #fff; border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .domain > summary::-webkit-details-marker {{ display: none; }}
  .domain > summary::before {{ content: "▶"; font-size: 0.7rem; transition: transform 0.2s; }}
  .domain[open] > summary::before {{ transform: rotate(90deg); }}
  .domain-badge {{
    color: #fff; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;
  }}
  .count {{ font-size: 0.75rem; color: #888; margin-left: auto; }}
  .domain-body {{ padding: 8px 0 0 0; }}
  .card {{
    margin: 10px 0; background: #fff; border-radius: 10px;
    border-left: 4px solid #ddd; padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .field {{
    display: flex; align-items: baseline; gap: 10px;
    margin-bottom: 10px; padding-bottom: 8px;
    border-bottom: 1px solid #f0f0f0;
  }}
  .field:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
  .text-field {{ flex-direction: column; gap: 6px; }}
  .field-label {{
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: #888; min-width: 32px; flex-shrink: 0;
    background: #f5f5f5; padding: 2px 8px; border-radius: 4px;
  }}
  .field-value {{ font-size: 0.85rem; }}
  .id-value {{ font-weight: 700; color: #333; font-family: monospace; }}
  .tag {{
    font-size: 0.75rem; padding: 3px 10px; border-radius: 10px; font-weight: 600;
    white-space: nowrap; display: inline-block;
  }}
  .text-value {{
    font-size: 0.85rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word;
    background: #fafafa; padding: 12px; border-radius: 8px; border: 1px solid #eee;
    max-height: 400px; overflow-y: auto;
  }}
  .role-marker {{
    font-weight: 700; font-size: 0.8rem; display: inline-block;
    padding: 2px 8px; border-radius: 4px; margin: 4px 0;
  }}
  .user-marker {{ background: #E3F2FD; color: #1565C0; }}
  .asst-marker {{ background: #F1F8E9; color: #33691E; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a1a; color: #e0e0e0; }}
    .stat, .domain > summary, .card {{ background: #2a2a2a; }}
    .field {{ border-bottom-color: #3a3a3a; }}
    .field-label {{ background: #333; color: #aaa; }}
    .text-value {{ background: #222; border-color: #3a3a3a; }}
    .user-marker {{ background: #1a2a3a; color: #64B5F6; }}
    .asst-marker {{ background: #1a2a1a; color: #81C784; }}
    .subtitle {{ color: #aaa; }}
    .id-value {{ color: #ddd; }}
    .count {{ color: #888; }}
  }}
</style>
</head>
<body>
<h1>OPUS Golden Dataset</h1>
<p class="subtitle">128 proxy-signal samples across 9 domains / 31 datasets</p>
<div class="stats">
  <div class="stat"><b>{len(samples)}</b>Samples</div>
  <div class="stat"><b>{len(domains)}</b>Domains</div>
  <div class="stat"><b>{len(set(s["tag"] for s in samples))}</b>Tags</div>
  <div class="stat"><b>31</b>Datasets</div>
</div>
{"".join(domain_html)}
</body>
</html>'''

    OUT.write_text(page)
    print(f"Written {OUT}  ({OUT.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
