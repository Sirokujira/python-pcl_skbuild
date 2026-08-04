"""Pipeline tests that run WITHOUT PCL installed.

They validate the pxd generation chain end to end:
  mirror headers -> cppast2autopxd -> committed pxd -> cython transpile
"""

import glob
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
    """Every Cython wrapper must transpile against the committed pxd files."""
    sources = sorted(glob.glob(os.path.join(ROOT, "src", "pcl", "*.pyx")))
    assert sources, "no .pyx modules found"

    for source in sources:
        name = os.path.basename(source)
        out = tmp_path / (name[: -len(".pyx")] + ".cpp")
        proc = subprocess.run(
            [
                sys.executable, "-m", "cython", "--cplus", "-3",
                "-I", os.path.join(ROOT, "src"),
                "-o", str(out),
                source,
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f"cython failed on {name}:\n{proc.stdout}\n{proc.stderr}"
        )
        assert out.exists()


def test_every_pyx_module_is_registered_with_cmake():
    """A .pyx nobody builds is a module that silently does not ship."""
    with open(os.path.join(ROOT, "src", "pcl", "CMakeLists.txt")) as fh:
        cmake = fh.read()
    for source in glob.glob(os.path.join(ROOT, "src", "pcl", "*.pyx")):
        module = os.path.basename(source)[: -len(".pyx")]
        assert module in cmake, (
            f"{module}.pyx is not listed in src/pcl/CMakeLists.txt"
        )


def test_generated_pxd_reference_real_pcl_headers():
    """pxd must `cdef extern from` the REAL pcl/ header paths (not mirrors)."""
    pxd_dir = os.path.join(ROOT, "src", "pcl", "pxd")
    expected = {
        "point_types.pxd": "pcl/point_types.h",
        "point_cloud.pxd": "pcl/point_cloud.h",
        os.path.join("io", "pcd_io.pxd"): "pcl/io/pcd_io.h",
        os.path.join("filters", "voxel_grid.pxd"): "pcl/filters/voxel_grid.h",
        os.path.join("kdtree", "kdtree_flann.pxd"): "pcl/kdtree/kdtree_flann.h",
        os.path.join("segmentation", "sac_segmentation.pxd"):
            "pcl/segmentation/sac_segmentation.h",
    }
    for rel, header in expected.items():
        with open(os.path.join(pxd_dir, rel)) as fh:
            assert f'cdef extern from "{header}"' in fh.read(), rel


def test_pxd_subpackages_are_importable():
    """Each generated pxd directory needs both __init__ files: __init__.py
    to make it a Python subpackage the wheel ships, __init__.pxd so
    `cimport pcl.pxd.<group>.<module>` resolves."""
    pxd_dir = os.path.join(ROOT, "src", "pcl", "pxd")
    for dirpath, _dirnames, filenames in os.walk(pxd_dir):
        if not any(f.endswith(".pxd") and f != "__init__.pxd"
                   for f in filenames):
            continue
        rel = os.path.relpath(dirpath, ROOT)
        assert "__init__.py" in filenames, f"{rel} is missing __init__.py"
        assert "__init__.pxd" in filenames, f"{rel} is missing __init__.pxd"
