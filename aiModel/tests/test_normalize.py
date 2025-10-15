import numpy as np
from aiModel.run_webcam import normalize_keypoints

def test_normalize_keypoints_shape_and_finite():
    pts = np.zeros((33, 3), dtype=float)
    # set shoulders and hips to create non-zero scale and center
    pts[11] = [0.5, 0.0, 0.0]   # left_shoulder
    pts[12] = [-0.5, 0.0, 0.0]  # right_shoulder
    pts[23] = [0.0, 1.0, 0.0]   # left_hip
    pts[24] = [0.0, -1.0, 0.0]  # right_hip
    # small deterministic noise so values are not all identical
    rng = np.random.RandomState(0)
    pts += rng.randn(*pts.shape) * 0.01

    out = normalize_keypoints(pts)
    assert out.shape == (33, 2)
    assert np.isfinite(out).all()