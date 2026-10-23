"""Assemble a clean, upload-ready copy of the static site in dist/space/.

    python scripts/build_space.py
Excludes node_modules, tests, package files and parity fixtures (only needed for local testing).
"""

import shutil
from pathlib import Path

SRC = Path("web")
OUT = Path("dist/space")
FILES = ["index.html", "sim.html", "home.css", "home.js", "style.css", "README.md"]
DIRS = ["src", "vendor", "results", "media"]

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for f in FILES:
    shutil.copy(SRC / f, OUT / f)
for d in DIRS:
    shutil.copytree(SRC / d, OUT / d)
(OUT / "models").mkdir()
shutil.copy(SRC / "models" / "policies.json", OUT / "models" / "policies.json")

total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print(f"{OUT}: {sum(1 for p in OUT.rglob('*') if p.is_file())} files, {total / 1e6:.1f} MB")
big = [(p.stat().st_size, p) for p in OUT.rglob("*") if p.is_file() and p.stat().st_size > 1_000_000]
for size, p in sorted(big, reverse=True):
    print(f"  {size / 1e6:.1f} MB  {p.relative_to(OUT)}")
