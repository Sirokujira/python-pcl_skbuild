"""pcl/common algorithms (PCA, intersections, norms, rigid motion,
projection matrix) and pcl/ml (k-means, SVM, DenseCrf)."""

import sys

import numpy as np
import pytest

pcl = pytest.importorskip("pcl")
pytest.importorskip("pcl._common")
pytest.importorskip("pcl._ml")

no_svm_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="pcl::SVM is not exported on Windows")


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# --- common -------------------------------------------------------------------


def test_pca_finds_the_long_axis():
    rng = np.random.RandomState(0)
    pts = rng.randn(2000, 3) * (5.0, 1.0, 0.2) + (1.0, 2.0, 3.0)
    pts = (pts @ _rotation((0, 0, 1), 0.5).T).astype(np.float32)
    pca = pcl.PCA(pcl.PointCloud(pts))
    assert np.allclose(pca.mean, pts.mean(axis=0), atol=1e-4)
    values, vectors = pca.eigenvalues, pca.eigenvectors
    assert values[0] > values[1] > values[2]
    assert np.allclose(values, np.linalg.eigvalsh(np.cov(pts.T))[::-1],
                       rtol=1e-3)
    long_axis = _rotation((0, 0, 1), 0.5) @ (1, 0, 0)
    assert abs(vectors[:, 0] @ long_axis) == pytest.approx(1, abs=1e-3)
    assert np.allclose(np.cross(vectors[:, 0], vectors[:, 1]), vectors[:, 2],
                       atol=1e-5)
    projected = pca.project(pcl.PointCloud(pts))
    assert np.allclose(projected.to_array(),
                       (pts - pca.mean) @ vectors, atol=1e-3)
    back = pca.reconstruct(projected)
    assert np.allclose(back.to_array(), pts, atol=1e-3)
    with pytest.raises(ValueError):
        pcl.PCA(pcl.PointCloud(pts[:2]))


def test_intersections():
    p = pcl.line_line_intersection([1, 2, 3, 1, 0, 0], [1, 2, 3, 0, 1, 1])
    assert np.allclose(p, (1, 2, 3), atol=1e-5)
    assert pcl.line_line_intersection([0, 0, 0, 1, 0, 0],
                                      [0, 1, 0, 1, 0, 0]) is None
    point, direction = pcl.plane_plane_intersection([0, 0, 1, 0],
                                                    [1, 0, 0, 0])
    assert abs(direction[1]) == pytest.approx(1) and abs(point[0]) < 1e-6
    assert pcl.plane_plane_intersection([0, 0, 1, 0], [0, 0, 1, -1]) is None
    corner = pcl.three_planes_intersection([1, 0, 0, -1], [0, 1, 0, -2],
                                           [0, 0, 1, -3])
    assert np.allclose(corner, (1, 2, 3))
    assert pcl.three_planes_intersection([1, 0, 0, -1], [1, 0, 0, -2],
                                         [0, 0, 1, -3]) is None


def test_histogram_distances():
    a = np.array([0.1, 0.4, 0.3, 0.2], np.float32)
    b = np.array([0.3, 0.2, 0.3, 0.2], np.float32)
    assert pcl.histogram_distance(a, b, "L1") == pytest.approx(0.4)
    assert pcl.histogram_distance(a, b, "L2") == pytest.approx(
        np.linalg.norm(a - b), rel=1e-5)
    assert pcl.histogram_distance(a, b, "L2_SQR") == pytest.approx(
        np.sum((a - b) ** 2), rel=1e-5)
    assert pcl.histogram_distance(a, b, "LINF") == pytest.approx(0.2)
    for norm in ("JM", "B", "SUBLINEAR", "CS", "DIV", "PF", "K", "KL", "HIK"):
        same = pcl.histogram_distance(a, a, norm)
        other = pcl.histogram_distance(a, b, norm)
        assert np.isfinite(other), norm
        if norm != "HIK":           # a similarity, largest for equal input
            assert abs(same) <= abs(other) + 1e-6, norm
    with pytest.raises(ValueError):
        pcl.histogram_distance(a, b[:3])


def test_transformation_from_correspondences():
    rng = np.random.RandomState(2)
    src = rng.rand(50, 3).astype(np.float32)
    rot = _rotation((1, 2, 3), 0.7)
    dst = (src @ rot.T + (0.5, -0.2, 1.0)).astype(np.float32)
    m = pcl.transformation_from_correspondences(pcl.PointCloud(src),
                                                pcl.PointCloud(dst))
    assert np.allclose(m[:3, :3], rot, atol=1e-4)
    assert np.allclose(m[:3, 3], (0.5, -0.2, 1.0), atol=1e-4)
    # a zero weight leaves an outlier out entirely
    dst[0] += 5
    m = pcl.transformation_from_correspondences(
        pcl.PointCloud(src), pcl.PointCloud(dst),
        weights=np.r_[0, np.ones(49)])
    assert np.allclose(m[:3, :3], rot, atol=1e-4)


def test_projection_matrix_of_a_pinhole_cloud():
    h, w = 60, 80
    k = np.array([[500.0, 0, 40], [0, 480, 30], [0, 0, 1]])
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    # not a plane: a planar scene leaves the projection undetermined
    z = 2.0 + 0.3 * np.sin(u / 7.0) + 0.2 * np.cos(v / 5.0)
    xyz = np.dstack([(u - 40) * z / 500, (v - 30) * z / 480, z])
    cloud = pcl.PointCloud()
    cloud.from_organized_array(xyz.astype(np.float32))
    p, camera, residual = pcl.estimate_projection_matrix(cloud)
    assert p.shape == (3, 4) and residual < 1e-3
    assert np.allclose(camera, k, rtol=2e-3, atol=0.1)
    with pytest.raises(ValueError, match="organized"):
        pcl.estimate_projection_matrix(pcl.PointCloud(xyz.reshape(-1, 3)
                                                      .astype(np.float32)))


def _matches(n_good=6, n_bad=4, seed=7):
    """Good matches agree on one rigid motion; bad ones point anywhere.
    Interleaved, best (lowest distance) first."""
    rng = np.random.RandomState(seed)
    true = np.eye(4)
    true[:3, :3] = _rotation((1, 1, 0), 0.6)
    true[:3, 3] = (0.5, -1.0, 2.0)
    model = rng.rand(n_good + n_bad, 3) * 4
    scene = model @ true[:3, :3].T + true[:3, 3]
    poses = np.repeat(true[None], n_good + n_bad, axis=0)
    bad = np.arange(n_good + n_bad) % 3 == 2           # every third
    bad[n_good + n_bad - (n_bad - bad.sum()):] |= bad.sum() < n_bad
    scene[bad] = rng.rand(bad.sum(), 3) * 4
    for i in np.flatnonzero(bad):
        poses[i][:3, :3] = _rotation(rng.randn(3), rng.rand() * 3)
    distances = np.linspace(0.1, 0.9, n_good + n_bad)
    return scene, model, poses, distances, true, bad


@pytest.mark.parametrize("size", [1, 2, 3])
def test_poses_from_matches(size):
    scene, model, poses, distances, true, bad = _matches()
    estimates = pcl.poses_from_matches(scene, model, poses, distances,
                                       size=size)
    assert estimates
    for pose, score, used in estimates:
        assert len(used) == size and pose.shape == (4, 4)
        assert score == pytest.approx(distances[used].mean(), rel=1e-5)
        if not bad[used].any():
            assert np.allclose(pose, true, atol=1e-3), used
    if size == 1:
        assert [u for _, _, u in estimates] == [[i] for i in range(len(bad))]
    else:
        # every all-good combination passes the distance checks (and
        # max_results=-1 keeps them all, which PCL's own -1 does not);
        # a bad one only passes by chance
        n_good = (~bad).sum()
        good = [u for _, _, u in estimates if not bad[u].any()]
        assert len(good) == (n_good * (n_good - 1) // 2 if size == 2
                             else n_good * (n_good - 1) * (n_good - 2) // 6)
        assert len(good) > len(estimates) / 2
    capped = pcl.poses_from_matches(scene, model, poses, distances,
                                    size=size, max_results=2)
    assert len(capped) == 2
    with pytest.raises(ValueError):
        pcl.poses_from_matches(scene, model, poses[:3], distances, size=size)


# --- ml ------------------------------------------------------------------------


def test_kmeans_separates_blobs():
    rng = np.random.RandomState(3)
    centres = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
    data = np.vstack([c + rng.randn(30, 2) for c in centres])
    order = rng.permutation(len(data))
    found, labels = pcl.kmeans(data[order], 3)
    assert found.shape == (3, 2) and labels.shape == (90,)
    for c in centres:
        assert np.min(np.linalg.norm(found - c, axis=1)) < 0.6
    # each blob ends up in one cluster
    truth = (np.arange(90) // 30)[order]
    for blob in range(3):
        assert len(set(labels[truth == blob])) == 1
    with pytest.raises(ValueError):
        pcl.kmeans(data, 0)


def _two_classes(n=60, seed=4):
    rng = np.random.RandomState(seed)
    labels = np.arange(n) % 2
    features = rng.randn(n, 2) * 0.6 + np.where(labels[:, None], 2.0, -2.0)
    return features.astype(np.float32), labels


@no_svm_on_windows
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_svm_classifies(kernel, tmp_path):
    features, labels = _two_classes()
    svm = pcl.SVM(kernel=kernel, probability=True)
    with pytest.raises(RuntimeError):
        svm.classify(features)
    svm.add_TrainingSamples(features, labels)
    svm.train()
    test, truth = _two_classes(seed=5)
    assert np.mean(svm.classify(test) == truth) > 0.95
    predicted, probabilities = svm.classify(test, probability=True)
    assert probabilities.shape == (len(test), 2)
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert sorted(svm.labels) == [0, 1]
    path = tmp_path / "model.svm"
    svm.save_Model(path)
    loaded = pcl.SVM()
    loaded.load_Model(path)
    assert np.array_equal(loaded.classify(test), svm.classify(test))


@no_svm_on_windows
def test_svm_reports_bad_parameters_instead_of_exiting():
    features, labels = _two_classes()
    svm = pcl.SVM(svm_type="nu_svc", nu=-1)
    svm.add_TrainingSamples(features, labels)
    with pytest.raises(ValueError, match="nu"):
        svm.train()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only refusal")
def test_svm_refused_on_windows():
    with pytest.raises(RuntimeError, match="Windows"):
        pcl.SVM()


def test_dense_crf_smooths_noisy_labels():
    n = 60
    truth = (np.arange(n) >= n // 2).astype(int)
    rng = np.random.RandomState(6)
    noisy = np.where(rng.rand(n) < 0.25, 1 - truth, truth)
    probability = np.where(np.arange(2)[None, :] == noisy[:, None], 0.65, 0.35)
    crf = pcl.DenseCrf(n, 2)
    with pytest.raises(RuntimeError):
        crf.add_PairwiseGaussian(3, 3, 3, 3)
    crf.set_Positions(np.c_[np.arange(n), np.zeros(n), np.zeros(n)])
    crf.set_UnaryEnergy(-np.log(probability))
    crf.add_PairwiseGaussian(3, 3, 3, 5)
    labels = crf.map_inference(10)
    assert np.mean(labels == truth) > np.mean(noisy == truth)
    assert np.mean(labels == truth) > 0.9
    marginals = crf.inference(10)
    assert marginals.shape == (n, 2)
    assert np.allclose(marginals.sum(axis=1), 1, atol=1e-5)
    with pytest.raises(RuntimeError, match="Colours"):
        crf.add_PairwiseBilateral(3, 3, 3, 10, 10, 10, 1)
