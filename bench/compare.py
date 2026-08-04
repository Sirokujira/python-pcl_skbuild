"""Diff two `name<TAB>seconds` result files into a readable table."""

from __future__ import annotations

import sys


def read(path: str) -> dict[str, float]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            name, value = line.split("\t")
            out[name] = float(value)
    return out


def main(argv: list[str]) -> int:
    cpp = read(argv[1])
    py = read(argv[2])
    n = int(argv[3]) if len(argv) > 3 else 0

    width = max(len(k) for k in cpp)
    print(f"{'operation'.ljust(width)}  {'C++':>12}  {'Python':>12}  "
          f"{'ratio':>8}  {'overhead':>12}")
    print("-" * (width + 52))
    for name in cpp:
        if name not in py:
            continue
        c, p = cpp[name], py[name]
        ratio = p / c if c else float("inf")
        delta = p - c
        per_point = f"{delta / n * 1e9:.0f} ns/pt" if n else ""
        print(f"{name.ljust(width)}  {c * 1e3:>9.3f} ms  {p * 1e3:>9.3f} ms  "
              f"{ratio:>7.2f}x  {per_point:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
