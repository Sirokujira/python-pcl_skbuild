"""Runtime tests for the remaining feature estimators (`_features_extra`).

Skipped when the package is not built (requires PCL at build time):
pip install . && pytest tests/
"""

import numpy as np
import pytest

pcl = pytest.importorskip(
    "pcl", reason="pcl package not built (requires PCL; pip install .)"
)


def _sphere_points(n, seed=0):
    rng = np.random.RandomState(seed)
    theta = rng.rand(n) * 2 * np.pi
    phi = np.arccos(2 * rng.rand(n) - 1)
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ]).astype(np.float32)


def _normals(points):
    values = np.zeros((len(points), 4), dtype=np.float32)
    values[:, :3] = points / np.linalg.norm(points, axis=1, keepdims=True)
    return pcl.PointCloud_Normal(values)


@pytest.fixture
def sphere():
    points = _sphere_points(1500)
    return pcl.PointCloud(points), _normals(points)


@pytest.fixture
def small_sphere():
    points = _sphere_points(40, seed=3)
    return pcl.PointCloud(points), _normals(points)


def _rgba(points, seed=1):
    rng = np.random.RandomState(seed)
    colours = np.zeros((len(points), 4), dtype=np.uint8)
    colours[:, :3] = (np.clip(points * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    colours[:, 3] = 255
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_rgba_array(points, colours)
    return cloud


def _xyzi(points):
    values = np.zeros((len(points), 4), dtype=np.float32)
    values[:, :3] = points
    values[:, 3] = (points[:, 0] + 1.0) * 100.0
    return pcl.PointCloud_PointXYZI(values)


def _check(arr, n, width):
    assert arr.dtype == np.float32
    assert arr.shape == (n, width)


# --- OpenMP / colour variants ------------------------------------------------


def test_fpfh_omp_matches_fpfh(sphere):
    cloud, normals = sphere
    omp = cloud.make_FPFHEstimationOMP()
    omp.set_InputNormals(normals)
    omp.set_RadiusSearch(0.2)
    omp.set_NumberOfThreads(2)
    parallel = omp.compute()
    single = cloud.make_FPFHEstimation()
    single.set_InputNormals(normals)
    single.set_RadiusSearch(0.2)
    _check(parallel, cloud.size, 33)
    assert parallel == pytest.approx(single.compute(), abs=1e-3)


def test_shot_omp_matches_shot(sphere):
    cloud, normals = sphere
    omp = pcl.SHOTEstimationOMP(cloud, threads=2)
    omp.set_InputNormals(normals)
    omp.set_RadiusSearch(0.3)
    single = cloud.make_SHOTEstimation()
    single.set_InputNormals(normals)
    single.set_RadiusSearch(0.3)
    parallel = omp.compute()
    _check(parallel, cloud.size, 352)
    assert np.allclose(parallel, single.compute(), atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("cls", ["SHOTColorEstimation", "SHOTColorEstimationOMP"])
def test_shot_color(sphere, cls):
    cloud, normals = sphere
    coloured = _rgba(cloud.to_array())
    est = getattr(coloured, "make_" + cls)()
    est.set_InputNormals(normals)
    est.set_RadiusSearch(0.3)
    out = est.compute()
    _check(out, cloud.size, 1344)
    finite = out[np.isfinite(out).all(axis=1)]
    assert len(finite) > cloud.size * 0.9
    # Both halves carry information.
    assert finite[:, :352].sum() > 0 and finite[:, 352:].sum() > 0


@pytest.mark.parametrize("shape, colour, width", [(True, False, 352),
                                                  (False, True, 992)])
def test_shot_color_halves_can_be_switched_off(sphere, shape, colour, width):
    """PCL packs the enabled half at the front; only it comes back."""
    cloud, normals = sphere
    est = pcl.SHOTColorEstimation(_rgba(cloud.to_array()),
                                  describe_shape=shape, describe_color=colour)
    est.set_InputNormals(normals)
    est.set_RadiusSearch(0.3)
    out = est.compute()
    _check(out, cloud.size, width)
    assert np.nansum(out) > 0
    if shape:
        # Shape only is exactly SHOT; with colour the whole 1344-vector is
        # normalised, so the shape half is the same direction, shorter.
        plain = cloud.make_SHOTEstimation()
        plain.set_InputNormals(normals)
        plain.set_RadiusSearch(0.3)
        assert np.allclose(out, plain.compute(), atol=1e-5, equal_nan=True)
        both = pcl.SHOTColorEstimation(_rgba(cloud.to_array()))
        both.set_InputNormals(normals)
        both.set_RadiusSearch(0.3)
        half = both.compute()[:, :352]
        half = half / np.linalg.norm(half, axis=1, keepdims=True)
        assert np.allclose(out, half, atol=1e-4, equal_nan=True)


def test_shot_color_needs_a_half():
    with pytest.raises(ValueError):
        pcl.SHOTColorEstimation(describe_shape=False, describe_color=False)


def test_pfhrgb(small_sphere):
    cloud, normals = small_sphere
    est = _rgba(cloud.to_array()).make_PFHRGBEstimation()
    est.set_InputNormals(normals)
    est.set_RadiusSearch(0.8)
    out = est.compute()
    _check(out, cloud.size, 250)
    assert np.isfinite(out).all()


# --- shape contexts ------------------------------------------------------------


def test_shape_context_3d(sphere):
    cloud, normals = sphere
    est = cloud.make_ShapeContext3DEstimation()
    est.set_InputNormals(normals)
    est.set_MinimalRadius(0.05)
    est.set_PointDensityRadius(0.1)
    est.set_RadiusSearch(0.4)
    out = est.compute()
    _check(out, cloud.size, 1980)
    assert np.nansum(out) > 0
    assert est.get_MinimalRadius() == pytest.approx(0.05)


def test_shape_context_radius_must_exceed_minimal(sphere):
    cloud, normals = sphere
    est = cloud.make_ShapeContext3DEstimation()
    est.set_InputNormals(normals)
    est.set_MinimalRadius(0.5)
    est.set_RadiusSearch(0.4)
    with pytest.raises(ValueError, match="minimal radius"):
        est.compute()


def test_unique_shape_context(sphere):
    cloud, _ = sphere
    est = cloud.make_UniqueShapeContext()
    est.set_MinimalRadius(0.05)
    est.set_PointDensityRadius(0.1)
    est.set_LocalRadius(0.3)
    est.set_RadiusSearch(0.4)
    out = est.compute()
    _check(out, cloud.size, 1960)
    assert np.nansum(out) > 0
    assert est.get_LocalRadius() == pytest.approx(0.3)


# --- point pair features ---------------------------------------------------------


def test_ppf_covers_every_ordered_pair(small_sphere):
    cloud, normals = small_sphere
    est = cloud.make_PPFEstimation()
    est.set_InputNormals(normals)
    out = est.compute()
    n = cloud.size
    _check(out, n * n, 5)
    diagonal = out[np.arange(n) * n + np.arange(n)]
    assert np.isnan(diagonal).all()
    off = np.delete(out, np.arange(n) * n + np.arange(n), axis=0)
    assert np.isfinite(off).all()
    # f4 is the pair's distance.
    points = cloud.to_array()
    assert out[0 * n + 1, 3] == pytest.approx(
        np.linalg.norm(points[1] - points[0]), abs=1e-5)


def test_ppfrgb_and_cppf(small_sphere):
    cloud, normals = small_sphere
    coloured = _rgba(cloud.to_array())
    n = cloud.size
    ppfrgb = coloured.make_PPFRGBEstimation()
    ppfrgb.set_InputNormals(normals)
    _check(ppfrgb.compute(), n * n, 8)
    cppf = coloured.make_CPPFEstimation()
    cppf.set_InputNormals(normals)
    out = cppf.compute()
    _check(out, n * n, 11)
    assert np.isfinite(np.delete(
        out, np.arange(n) * n + np.arange(n), axis=0)).all()


# --- local reference frames -------------------------------------------------------


def _orthonormal(frames):
    good = frames[np.isfinite(frames).all(axis=1)].reshape(-1, 3, 3)
    eye = np.einsum("nij,nkj->nik", good, good)
    return len(good), np.abs(eye - np.eye(3)).max() if len(good) else 0


@pytest.mark.parametrize("cls", ["SHOTLocalReferenceFrameEstimation",
                                 "SHOTLocalReferenceFrameEstimationOMP"])
def test_shot_lrf_frames_are_orthonormal(sphere, cls):
    cloud, _ = sphere
    est = getattr(cloud, "make_" + cls)()
    est.set_RadiusSearch(0.3)
    frames = est.compute()
    _check(frames, cloud.size, 9)
    count, error = _orthonormal(frames)
    assert count > cloud.size * 0.9
    assert error < 1e-3


def test_board_lrf_z_is_the_normal(sphere):
    cloud, normals = sphere
    est = cloud.make_BOARDLocalReferenceFrameEstimation()
    est.set_InputNormals(normals)
    est.set_RadiusSearch(0.3)
    est.set_FindHoles(True)
    assert est.get_FindHoles() is True
    frames = est.compute()
    _check(frames, cloud.size, 9)
    count, error = _orthonormal(frames)
    assert count > cloud.size * 0.9 and error < 1e-3
    z = frames[:, 6:9]
    dots = np.abs(np.sum(z * normals.to_array()[:, :3], axis=1))
    assert np.nanmedian(dots) > 0.95


def test_flare_lrf(sphere):
    cloud, normals = sphere
    est = cloud.make_FLARELocalReferenceFrameEstimation()
    est.set_InputNormals(normals)
    est.set_RadiusSearch(0.2)
    with pytest.raises(RuntimeError, match="TangentRadius"):
        est.compute()
    est.set_TangentRadius(0.4)
    frames = est.compute()
    _check(frames, cloud.size, 9)
    count, error = _orthonormal(frames)
    assert count > cloud.size * 0.9 and error < 1e-3
    distances = est.get_SignedDistancesFromHighestPoints()
    assert distances.shape == (cloud.size,)


# --- intensity descriptors -----------------------------------------------------------


def test_intensity_gradient_follows_the_ramp(sphere):
    cloud, normals = sphere
    est = _xyzi(cloud.to_array()).make_IntensityGradientEstimation()
    est.set_InputNormals(normals)
    est.set_RadiusSearch(0.2)
    gradient = est.compute()
    _check(gradient, cloud.size, 3)
    # Intensity grows with x; on the equator facing +y the tangent-plane
    # gradient points along +x.
    points = cloud.to_array()
    side = (np.abs(points[:, 1]) > 0.9)
    assert np.nanmedian(gradient[side, 0]) > 0


def test_intensity_spin_and_rift(sphere):
    cloud, normals = sphere
    xyzi = _xyzi(cloud.to_array())
    spin = xyzi.make_IntensitySpinEstimation()
    spin.set_RadiusSearch(0.2)
    spin.set_Bins(5, 4)
    assert (spin.get_NrDistanceBins(), spin.get_NrIntensityBins()) == (5, 4)
    _check(spin.compute(), cloud.size, 20)
    with pytest.raises(ValueError):
        spin.set_Bins(5, 5)

    gradient_est = xyzi.make_IntensityGradientEstimation()
    gradient_est.set_InputNormals(normals)
    gradient_est.set_RadiusSearch(0.2)
    rift = xyzi.make_RIFTEstimation()
    rift.set_InputGradient(gradient_est.compute())
    rift.set_RadiusSearch(0.2)
    out = rift.compute()
    _check(out, cloud.size, 32)
    assert np.nansum(out) > 0
    with pytest.raises(ValueError):
        rift.set_Bins(4, 9)
    with pytest.raises(ValueError):
        rift.set_InputGradient(np.zeros((3, 3), np.float32))


def test_intensity_estimators_need_their_inputs(sphere):
    cloud, _ = sphere
    xyzi = _xyzi(cloud.to_array())
    with pytest.raises(RuntimeError):
        xyzi.make_IntensityGradientEstimation().compute()
    with pytest.raises(RuntimeError):
        xyzi.make_RIFTEstimation().compute()
    with pytest.raises(RuntimeError):
        xyzi.make_IntensitySpinEstimation().compute()


# --- normals ---------------------------------------------------------------------


def test_normal_based_signature_is_not_exposed():
    """Left out on purpose: PCL's returns only NaN (README)."""
    assert not hasattr(pcl, "NormalBasedSignatureEstimation")


def _organized_wave(h=30, w=40, f=40.0):
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 2.0 + 0.1 * np.sin(u / 6.0)
    xyz = np.dstack([(u - w / 2) * z / f, (v - h / 2) * z / f, z])
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz.astype(np.float32))
    return cloud


def test_linear_least_squares_normals_face_the_camera():
    cloud = _organized_wave()
    est = cloud.make_LinearLeastSquaresNormalEstimation()
    est.set_NormalSmoothingSize(5.0)
    est.set_MaxDepthChangeFactor(0.05)
    normals = est.compute()
    assert isinstance(normals, pcl.PointCloud_Normal)
    assert normals.size == cloud.size
    n = normals.to_array()[:, :3]
    good = np.isfinite(n).all(axis=1)
    assert good.mean() > 0.5
    assert np.median(np.abs(n[good, 2])) > 0.8
    with pytest.raises(ValueError):
        pcl.LinearLeastSquaresNormalEstimation(
            pcl.PointCloud(np.zeros((5, 3), np.float32)))


# --- organized edges ---------------------------------------------------------------


def _step_cloud(h=30, w=40, f=40.0):
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = np.where(u < w / 2, 1.0, 2.0).astype(np.float32)
    return np.dstack([(u - w / 2) * z / f, (v - h / 2) * z / f, z]).astype(
        np.float32)


def test_organized_edges_find_the_depth_step():
    xyz = _step_cloud()
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz)
    edges = cloud.make_OrganizedEdgeBase()
    edges.set_DepthDisconThreshold(0.02)
    labels, indices = edges.compute()
    assert labels.shape == (30, 40)
    assert len(indices) == 5
    occluding = labels & pcl.EDGELABEL_OCCLUDING != 0
    occluded = labels & pcl.EDGELABEL_OCCLUDED != 0
    # Near side of the step is column 19, far side column 20.
    assert occluding[:, 19].mean() > 0.8
    assert occluded[:, 20].mean() > 0.8
    assert set(indices[1].tolist()) == set(np.flatnonzero(occluding.ravel()))


def test_organized_edges_from_normals_and_rgb():
    xyz = _step_cloud()
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz)
    normal_est = cloud.make_IntegralImageNormalEstimation()
    normals = normal_est.compute_cloud()
    with_normals = cloud.make_OrganizedEdgeFromNormals()
    with_normals.set_InputNormals(normals)
    with_normals.set_EdgeType(pcl.EDGELABEL_OCCLUDING |
                              pcl.EDGELABEL_HIGH_CURVATURE)
    labels, indices = with_normals.compute()
    assert labels.shape == (30, 40)
    assert with_normals.get_EdgeType() == (pcl.EDGELABEL_OCCLUDING |
                                           pcl.EDGELABEL_HIGH_CURVATURE)

    rgba = np.zeros((30, 40, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    rgba[10:20, 10:30, :3] = 250
    coloured = pcl.PointCloud_PointXYZRGBA()
    coloured.from_organized_rgba_array(xyz, rgba)
    rgb = coloured.make_OrganizedEdgeFromRGB()
    rgb.set_EdgeType(pcl.EDGELABEL_RGB_CANNY)
    labels, indices = rgb.compute()
    assert (labels & pcl.EDGELABEL_RGB_CANNY != 0).sum() > 0
    assert len(indices[4]) > 0

    both = coloured.make_OrganizedEdgeFromRGBNormals()
    both.set_InputNormals(normals)
    labels, _ = both.compute()
    assert labels.shape == (30, 40)


def test_organized_edges_need_an_organized_cloud():
    flat = pcl.PointCloud(np.zeros((10, 3), np.float32))
    with pytest.raises(ValueError):
        pcl.OrganizedEdgeBase(flat)
    with pytest.raises(RuntimeError):
        pcl.OrganizedEdgeFromNormals().compute()


# --- global and mesh descriptors -----------------------------------------------------


def test_gfpfh_describes_a_labelled_cloud():
    rng = np.random.RandomState(4)
    points = rng.rand(800, 3).astype(np.float32)
    labels = (points[:, 0] > 0.5).astype(np.int64) + 1
    est = pcl.GFPFHEstimation()
    est.set_InputCloud(pcl.PointCloud(points), labels)
    est.set_OctreeLeafSize(0.1)
    est.set_NumberOfClasses(2)
    descriptor = est.compute()
    assert descriptor.shape == (16,)
    assert np.isfinite(descriptor).all()
    assert descriptor.sum() > 0
    with pytest.raises(ValueError):
        est.set_InputCloud(pcl.PointCloud(points), labels[:5])


def test_rops_on_a_mesh():
    u, v = np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 1, 15))
    z = 0.1 * np.sin(3 * u) * np.cos(3 * v)
    points = np.column_stack([u.ravel(), v.ravel(), z.ravel()]).astype(
        np.float32)
    triangles = []
    for r in range(14):
        for c in range(14):
            i = r * 15 + c
            triangles += [(i, i + 1, i + 16), (i, i + 16, i + 15)]
    cloud = pcl.PointCloud(points)
    rops = cloud.make_ROPSEstimation(triangles)
    rops.set_SupportRadius(0.25)
    rops.set_NumberOfPartitionBins(5)
    assert rops.get_NumberOfRotations() == 3
    centre = [7 * 15 + 7, 6 * 15 + 8]
    out = rops.compute(centre)
    _check(out, 2, 135)
    assert np.isfinite(out).all()
    with pytest.raises(ValueError):
        rops.set_Triangles([(0, 1, 2, 3)])
    with pytest.raises(IndexError):
        rops.compute([10 ** 6])


def test_multiscale_interest_regions(sphere):
    cloud, _ = sphere
    extractor = cloud.make_StatisticalMultiscaleInterestRegionExtraction()
    extractor.set_ScalesVector([0.1, 0.2])
    assert extractor.get_ScalesVector() == pytest.approx([0.1, 0.2])
    regions = extractor.compute_RegionsOfInterest()
    assert isinstance(regions, list)
    for region in regions:
        assert region.dtype == np.int32
        assert np.all((region >= 0) & (region < cloud.size))
    with pytest.raises(ValueError):
        extractor.set_ScalesVector([])


def test_multiscale_feature_persistence(sphere):
    cloud, normals = sphere
    persistence = cloud.make_MultiscaleFeaturePersistence(normals)
    persistence.set_ScalesVector([0.1, 0.15, 0.2])
    persistence.set_Alpha(0.8)
    persistence.set_DistanceMetric(pcl.NORM_L2)
    features, indices = persistence.determine_PersistentFeatures()
    assert features.shape == (len(indices), 33)
    assert np.all((indices >= 0) & (indices < cloud.size))
    with pytest.raises(ValueError):
        persistence.set_DistanceMetric(99)


# --- functions ---------------------------------------------------------------------------


def test_compute_approximate_normals_of_a_tetrahedron_face():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      dtype=np.float32)
    normals = pcl.compute_approximate_normals(pcl.PointCloud(points),
                                              [(0, 1, 2)])
    assert normals.size == 4
    n = normals.to_array()[:3, :3]
    assert np.abs(n[:, 2]) == pytest.approx(np.ones(3), abs=1e-5)
    with pytest.raises(IndexError):
        pcl.compute_approximate_normals(pcl.PointCloud(points), [(0, 1, 9)])


def test_compute_pair_features():
    f1, f2, f3, f4 = pcl.compute_pair_features(
        (0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, 1))
    assert f4 == pytest.approx(1.0)
    assert all(np.isfinite([f1, f2, f3]))
    with pytest.raises(ValueError):
        pcl.compute_pair_features((0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 1))
    with pytest.raises(ValueError):
        pcl.compute_pair_features((0, 0), (0, 0, 1), (1, 0, 0), (0, 0, 1))
