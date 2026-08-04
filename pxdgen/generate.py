#!/usr/bin/env python3
"""Regenerate (or verify) the committed Cython pxd files under src/pcl/pxd/.

Usage:
    python pxdgen/generate.py            # rewrite src/pcl/pxd/**/*.pxd
    python pxdgen/generate.py --check    # exit 1 if committed pxd are stale

The base config (pxdgen/pcl_headers.toml) carries the [generator] settings
and typemap substitutions; per-module header lists live in
pxdgen/modules/*.toml so the PCL surface can grow without one unreadable
file. Module files contain ONLY [[headers]] entries, and their relative
paths resolve against pxdgen/ — the same base directory as the main
config — so an entry reads the same wherever it is written.

Requires cppast2autopxd:  pip install git+https://github.com/sirokujira/cppast2autopxd
"""

from __future__ import annotations

import argparse
import difflib
import glob
import os
import sys

try:
    from cppast2autopxd import load_config
    from cppast2autopxd.config import HeaderJob
    from cppast2autopxd.generator import generate_job
except ImportError:  # pragma: no cover
    print(
        "error: cppast2autopxd is not installed.\n"
        "  pip install git+https://github.com/sirokujira/cppast2autopxd",
        file=sys.stderr,
    )
    sys.exit(2)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "pcl_headers.toml")
MODULE_GLOB = os.path.join(HERE, "modules", "*.toml")


def _resolve(path: str) -> str:
    """Resolve a config-relative path against pxdgen/."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(HERE, path))


def _load_module_jobs(path: str) -> list:
    """Read the [[headers]] entries of a per-module config file."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    for key in data:
        if key != "headers":
            raise SystemExit(
                f"error: {os.path.relpath(path, HERE)} defines [{key}]; "
                "module configs may only contain [[headers]]. Generator "
                "settings and typemap substitutions belong in "
                "pcl_headers.toml so they cannot drift between modules."
            )

    jobs = []
    for h in data.get("headers", []):
        jobs.append(
            HeaderJob(
                path=_resolve(h["path"]),
                output=_resolve(h["output"]),
                extern_from=h.get("extern_from"),
                namespaces=list(h.get("namespaces", [])),
                include_names=list(h.get("include", [])),
                exclude_names=list(h.get("exclude", [])),
                extra_cimports=list(h.get("extra_cimports", [])),
                language=h.get("language"),
                pyx_scaffold=(
                    _resolve(h["pyx_scaffold"]) if h.get("pyx_scaffold") else None
                ),
                pxd_module=h.get("pxd_module"),
            )
        )
    return jobs


def load_all():
    """Base config plus every pxdgen/modules/*.toml header list."""
    cfg = load_config(CONFIG)
    for module_path in sorted(glob.glob(MODULE_GLOB)):
        cfg.headers.extend(_load_module_jobs(module_path))

    # Two headers writing the same pxd would silently lose one of them —
    # the failure mode the flat src/pcl/pxd/ layout used to invite.
    seen = {}
    for job in cfg.headers:
        if job.output in seen:
            raise SystemExit(
                "error: two headers generate "
                f"{os.path.relpath(job.output, os.path.dirname(HERE))} "
                f"({seen[job.output]} and {job.path}); give one of them a "
                "different output path."
            )
        seen[job.output] = job.path
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail if committed pxd files differ from "
        "freshly generated output",
    )
    args = ap.parse_args(argv)

    cfg = load_all()
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
