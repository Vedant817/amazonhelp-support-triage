"""Summarize pip-audit JSON: which vulnerabilities are fixable and to what."""
import json
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pip_audit", "-r", "requirements.txt", "--format", "json", "--progress-spinner", "off"],
    capture_output=True,
    text=True,
)
if not result.stdout.strip():
    print("pip-audit produced no JSON. stderr:", result.stderr[:500])
    raise SystemExit(1)

data = json.loads(result.stdout)

total = 0
fixable = []
unfixable = []

for dep in data.get("dependencies", []):
    vulns = dep.get("vulns", [])
    if not vulns:
        continue
    total += len(vulns)
    fixes = sorted({f for v in vulns for f in (v.get("fix_versions") or [])})
    line = f"{dep['name']}=={dep['version']}: {len(vulns)} vuln(s)"
    if fixes:
        line += f" -> fixed in {', '.join(fixes)}"
        fixable.append((dep["name"], dep["version"], fixes, [v["id"] for v in vulns]))
    else:
        line += " -> NO FIX RELEASED"
        unfixable.append((dep["name"], dep["version"], [v["id"] for v in vulns]))
    print(line)

print()
print(f"Total vulnerabilities: {total}")
print(f"Dependencies with a fix available: {len(fixable)}")
print(f"Dependencies with no fix released yet: {len(unfixable)}")

if unfixable:
    print("\nUnfixable (monitor):")
    for name, version, ids in unfixable:
        print(f"  {name}=={version}: {', '.join(ids)}")