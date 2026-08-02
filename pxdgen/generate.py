#!/usr/bin/env python3
"""Regenerate (or verify) the committed Cython pxd files under src/pcl/pxd/.

Usage:
    python pxdgen/generate.py            # rewrite src/pcl/pxd/*.pxd
    python pxdgen/generate.py --check    # exit 1 if committed pxd are stale

Requires cppast2autopxd:  pip install git+https://github.com/sirokujira/cppast2autopxd
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys

try:
    from cppast2autopxd import load_config
    from cppast2autopxd.generator import generate_job
except ImportError:  # pragma: no cover
    print(
        "error: cppast2autopxd is not installed.\n"
        "  pip install git+https://github.com/sirokujira/cppast2autopxd",
        file=sys.stderr,
    )
    sys.exit(2)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "pcl_headers.toml")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail if committed pxd files differ from "
        "freshly generated output",
    )
    args = ap.parse_args(argv)

    cfg = load_config(CONFIG)
    stale = []
    for job in cfg.headers:
        result = generate_job(cfg, job)
        for w in result.warnings:
            print(f"warning [{os.path.basename(job.path)}]: {w}", file=sys.stderr)

        rel = os.path.relpath(job.output, os.path.dirname(HERE))
        if args.check:
            try:
                with open(job.output, "r", encoding="utf-8") as fh:
                    committed = fh.read()
            except FileNotFoundError:
                committed = ""
            if committed != result.text:
                stale.append(rel)
                diff = difflib.unified_diff(
                    committed.splitlines(keepends=True),
                    result.text.splitlines(keepends=True),
                    fromfile=f"{rel} (committed)",
                    tofile=f"{rel} (generated)",
                )
                sys.stderr.writelines(diff)
        else:
            os.makedirs(os.path.dirname(job.output), exist_ok=True)
            with open(job.output, "w", encoding="utf-8") as fh:
                fh.write(result.text)
            print(f"wrote {rel}")

    if stale:
        print(
            "\nerror: stale pxd files (run `python pxdgen/generate.py`): "
            + ", ".join(stale),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
