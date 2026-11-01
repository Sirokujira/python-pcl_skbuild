#!/usr/bin/env python3
"""Regenerate (or verify) the committed Cython pxd files under src/pcl/pxd/.

Usage:
    python pxdgen/generate.py            # rewrite src/pcl/pxd/**/*.pxd
    python pxdgen/generate.py --check    # exit 1 if committed pxd are stale
    python pxdgen/generate.py --backend cppast   # same, through the C++ tool

The committed pxd files are the libclang backend's output. `--backend
cppast` writes what cppast2autopxd's C++ tool (cppast_autopxd) generates
for the same config instead, so the two can be diffed; it does not combine
with --check, because the two backends differ cosmetically (block-level vs
per-function `nogil`, cimport order) and a byte comparison would always be
"stale". Cross-backend parity is tested where it belongs, in
tests/test_pxd_pipeline.py::test_cppast_backend_parity.

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
    ap.add_argument(
        "--backend",
        choices=["libclang", "cppast"],
        default="libclang",
        help="parser to generate with: libclang (cppast2autopxd's own; what "
        "the committed files are) or cppast (delegate every header to the "
        "cppast_autopxd binary, found via CPPAST2AUTOPXD_CPP_TOOL, PATH or "
        "the cppast_autopxd_native wheel). Not combinable with --check",
    )
    args = ap.parse_args(argv)

    if args.backend == "cppast":
        if args.check:
            print(
                "error: --check compares against the committed pxd, which are "
                "libclang output; the cppast backend differs cosmetically, so "
                "parity is tested by tests/test_pxd_pipeline.py::"
                "test_cppast_backend_parity instead",
                file=sys.stderr,
            )
            return 2
        try:
            from cppast2autopxd.generator import generate_job_cppast
        except ImportError:
            print(
                "error: the installed cppast2autopxd predates its cppast "
                "backend's batch mode; upgrade:\n"
                "  pip install -U git+https://github.com/sirokujira/cppast2autopxd",
                file=sys.stderr,
            )
            return 2
        generate = generate_job_cppast
    else:
        generate = generate_job

    cfg = load_all()
    stale = []
    for job in cfg.headers:
        try:
            result = generate(cfg, job)
        except (RuntimeError, ValueError) as err:
            # a missing/failing cppast_autopxd binary, or a job option the
            # cppast backend cannot honor: say which, do not write anything
            print(f"error [{os.path.basename(job.path)}]: {err}", file=sys.stderr)
            return 2
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
