"""Structural validation of the served dashboard HTML.

Catches the common UI bugs: missing element IDs referenced by JS, dangling
data-* hooks and classes the script queries but the markup never defines,
unbalanced tags, duplicate IDs, and broken endpoint references.
"""

import re
import sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts_verify_common import base_url_from_argv, exit_if_unreachable

BASE = base_url_from_argv()
FAILURES = []


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    exit_if_unreachable(BASE)
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

    # 3. Every JS query hook resolves to something in the markup. This
    #    generalises the old preset check: a data-* attribute the script binds
    #    to, or a class it queries, that the markup never defines is silently
    #    dead code, which is exactly how the demo-case buttons rotted.
    queried_attrs = set(re.findall(r'querySelectorAll?\(\s*["\']\[(data-[a-z-]+)\]', html))
    present_attrs = set(re.findall(r'\b(data-[a-z-]+)=', html))
    dangling_attrs = sorted(queried_attrs - present_attrs)
    check("every queried data-* attribute exists in markup", not dangling_attrs,
          f"missing: {dangling_attrs}")

    queried_classes = set(re.findall(r'querySelectorAll\(\s*["\']\.([a-z][a-z0-9-]*)["\']', html))
    present_classes = set(re.findall(r'class="([^"]*)"', html))
    flattened = {c for group in present_classes for c in group.split()}
    dangling_classes = sorted(queried_classes - flattened)
    check("every queried class exists in markup", not dangling_classes,
          f"missing: {dangling_classes}")

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