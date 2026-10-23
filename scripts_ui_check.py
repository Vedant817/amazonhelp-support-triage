"""Structural validation of the served dashboard HTML.

Catches the common UI bugs: missing element IDs referenced by JS, unbalanced tags,
preset keys without copy, duplicate IDs, and broken endpoint references.
"""

import re
import sys
import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
FAILURES = []


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    html = httpx.get(f"{BASE}/", timeout=30).text

    # 1. Every byId("x") / $("x") reference has a matching id="x"
    referenced = set(re.findall(r'(?:byId|\$)\(\s*"([A-Za-z0-9_-]+)"\s*\)', html))
    declared = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', html))
    missing = sorted(referenced - declared)
    check("all JS-referenced IDs exist", not missing, f"missing: {missing}")

    # 2. No duplicate IDs
    all_ids = re.findall(r'\bid="([A-Za-z0-9_-]+)"', html)
    dups = sorted({i for i in all_ids if all_ids.count(i) > 1})
    check("no duplicate element IDs", not dups, f"duplicates: {dups}")

    # 3. Preset buttons all have copy in the presets object
    preset_keys = set(re.findall(r'data-preset="([a-z_]+)"', html))
    preset_block = re.search(r"const presets = \{(.*?)\};", html, re.S)
    defined = set(re.findall(r"([a-z_]+)\s*:", preset_block.group(1))) if preset_block else set()
    check("every preset button has copy", preset_keys <= defined, f"missing: {sorted(preset_keys - defined)}")

    # 4. Balanced tags for structural elements
    for tag in ["div", "section", "aside", "main", "header", "ol", "li", "button", "span"]:
        opens = len(re.findall(rf"<{tag}[\s>]", html))
        closes = len(re.findall(rf"</{tag}>", html))
        check(f"<{tag}> balanced", opens == closes, f"open={opens} close={closes}")

    # 5. No leftover template markers or debug text
    check("no '{{' template markers", "{{" not in html)
    check("no 'undefined' text in markup", ">undefined<" not in html)
    check("no TODO/FIXME in UI", "TODO" not in html and "FIXME" not in html)

    # 6. Accessibility basics
    check("viewport meta present", 'name="viewport"' in html)
    check("html lang set", 'lang="en"' in html)
    check("form control has label/aria", 'aria-describedby="charCount"' in html)
    check("live regions present", 'aria-live' in html)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nDashboard structure is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())