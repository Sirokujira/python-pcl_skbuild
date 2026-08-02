"""Pipeline tests that run WITHOUT PCL installed.

They validate the pxd generation chain end to end:
  mirror headers -> cppast2autopxd -> committed pxd -> cython transpile
"""

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cppast2autopxd = pytest.importorskip(
    "cppast2autopxd",
    reason="cppast2autopxd not installed "
    "(pip install git+https://github.com/sirokujira/cppast2autopxd)",
)


def test_committed_pxd_up_to_date():
    """`pxdgen/generate.py --check` must pass: committed pxd == regenerated."""
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "pxdgen", "generate.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "committed pxd files are stale; run `python pxdgen/generate.py`\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.skipif(shutil.which("cython") is None, reason="cython missing")
def test_wrapper_pyx_transpiles(tmp_path):
    """The Cython wrapper must transpile against the committed pxd files."""
    out = tmp_path / "_pointcloud.cpp"
    proc = subprocess.run(
        [
            sys.executable, "-m", "cython", "--cplus", "-3",
            "-I", os.path.join(ROOT, "src"),
            "-o", str(out),
            os.path.join(ROOT, "src", "pcl", "_pointcloud.pyx"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"cython failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.exists()


def test_generated_pxd_reference_real_pcl_headers():
    """pxd must `cdef extern from` the REAL pcl/ header paths (not mirrors)."""
    pxd_dir = os.path.join(ROOT, "src", "pcl", "pxd")
    with open(os.path.join(pxd_dir, "point_types.pxd")) as fh:
        assert 'cdef extern from "pcl/point_types.h"' in fh.read()
    with open(os.path.join(pxd_dir, "point_cloud.pxd")) as fh:
        assert 'cdef extern from "pcl/point_cloud.h"' in fh.read()
