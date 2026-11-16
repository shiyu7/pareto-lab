#!/usr/bin/env python3
"""Build the website's reproducible Skill ZIP from tracked source files."""

import argparse
from pathlib import Path
import subprocess
import zipfile


def package(output):
    root = Path(__file__).resolve().parents[1]
    skill = root / "skills" / "pareto-serving-benchmark"
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "skills/pareto-serving-benchmark"], cwd=root
    ).decode().split("\0")
    paths = sorted(root / name for name in tracked if name)
    if skill / "SKILL.md" not in paths:
        raise ValueError("Track the Skill source files in Git before packaging")
    for path in paths:
        path.resolve().relative_to(skill.resolve())
        if not path.is_file():
            raise ValueError(f"Missing Skill source: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            name = path.relative_to(root / "skills").as_posix()
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes(), compresslevel=9)
    print(f"Packaged {len(paths)} files: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/downloads/pareto-serving-benchmark.zip",
    )
    package(parser.parse_args().output)
