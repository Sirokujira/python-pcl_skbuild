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


# --- cross-backend parity (auto-skips without the C++ tool) ------------------
#
# cppast2autopxd ships two implementations; the committed pxd files are the
# libclang one's output. This gate runs every header job through the cppast
# backend (the `cppast_autopxd` binary) and holds it to the same API surface
# and to the real cython compiler, the way tests/test_real_pcl.py in that
# project holds the generator to a real PCL install: it skips where the tool
# is absent and gates where it is present.

import re


def _cppast_tool():
    try:
        from cppast2autopxd.cppast_backend import find_cppast_tool
        from cppast2autopxd.generator import generate_job_cppast  # noqa: F401
    except ImportError:
        return None
    return find_cppast_tool()


def _declared_names(text):
    """The API surface of a pxd: every declared identifier, keyed by the
    extern block it sits in.

    Deliberately blind to the two backends' cosmetic differences — a
    by-value parameter's `const`, a namespace-scope constant's initializer,
    enum member values, parameter names on a function-pointer typedef,
    `nogil`/`except +` spelling, cimport order — and sharp on what matters:
    a declaration dropped, renamed or emitted in the wrong block.
    """
    names = set()
    block = None
    for line in text.splitlines():
        t = line.strip()
        if not t or t.startswith("#") or t.startswith(("from ", "cimport ")):
            continue
        if t.startswith("cdef extern from"):
            block = t.replace('" nogil:', '":')
            continue
        # block headers of nested scopes: cppclass/struct/enum/union
        m = re.match(r"(?:cdef )?(?:cppclass|struct|enum|union)\s+([A-Za-z_]\w*)", t)
        if m:
            names.add((block, m.group(1)))
            continue
        if t.startswith("ctypedef"):
            body = t[len("ctypedef"):]
            fp = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", body)
            ident = fp.group(1) if fp else re.findall(r"[A-Za-z_]\w*", body)[-1]
            names.add((block, ident))
            continue
        # a callable: the identifier just before the parameter list
        # (a member template's `[T]` sits between the name and the `(`)
        m = re.search(r"([A-Za-z_]\w*|operator\S*?)\s*(?:\[[^\]]*\])?\s*\(", t)
        if m and not t.startswith("pass"):
            names.add((block, m.group(1)))
            continue
        # a field, a constant, or an enum member: the first identifier that
        # is not part of the type, i.e. the last one before `=` or the end
        head = t.split("=", 1)[0]
        idents = re.findall(r"[A-Za-z_]\w*", head)
        if idents:
            names.add((block, idents[-1]))
    return names


@pytest.mark.skipif(_cppast_tool() is None,
                    reason="cppast_autopxd binary not available "
                           "(CPPAST2AUTOPXD_CPP_TOOL, PATH, or the "
                           "cppast_autopxd_native wheel)")
def test_cppast_backend_parity(tmp_path):
    """Every header job, generated through the cppast backend, must declare
    the same API surface as the committed libclang output and compile."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pxdgen_generate", os.path.join(ROOT, "pxdgen", "generate.py")
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    from cppast2autopxd.generator import generate_job_cppast

    cfg = gen.load_all()
    assert cfg.headers, "no header jobs loaded"

    out_root = tmp_path / "gen"
    problems = []
    for job in cfg.headers:
        result = generate_job_cppast(cfg, job)
        rel = os.path.relpath(job.output, os.path.join(ROOT, "src"))
        skipped = [w for w in result.warnings if w.startswith("skipped:")]
        if skipped:
            problems.append(f"{rel}: declarations skipped: {skipped}")
        with open(job.output, encoding="utf-8") as fh:
            committed = _declared_names(fh.read())
        generated = _declared_names(result.text)
        if committed - generated:
            problems.append(f"{rel}: cppast backend lacks {sorted(committed - generated)}")
        if generated - committed:
            problems.append(f"{rel}: cppast backend adds {sorted(generated - committed)}")
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.text)
    assert not problems, "\n".join(problems)

    # each generated pxd must compile with its generated siblings on the path
    if shutil.which("cython") is None:
        pytest.skip("cython missing")
    for d, _, _ in os.walk(out_root / "pcl"):
        open(os.path.join(d, "__init__.pxd"), "a").close()
    failures = []
    for job in cfg.headers:
        rel = os.path.relpath(job.output, os.path.join(ROOT, "src"))
        proc = subprocess.run(
            [sys.executable, "-m", "cython", "--cplus", "-3",
             "-I", str(out_root), "-o", str(tmp_path / "x.cpp"),
             str(out_root / rel)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            failures.append(f"{rel}:\n{proc.stderr.strip().splitlines()[-1]}")
    assert not failures, "\n".join(failures)
