"""Optional DenseCRF post-processing (pydensecrf).

pydensecrf has no reliable wheels for modern Python, so this module degrades
gracefully: ``CRF_AVAILABLE`` is False and callers skip CRF refinement stages
with a warning instead of failing the pipeline.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on environment
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    CRF_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure disables the feature
    dcrf = None
    unary_from_softmax = None
    CRF_AVAILABLE = False


def dense_crf_inference(image_rgb: np.ndarray, probs: np.ndarray, iterations: int = 10,
                        gaussian_sxy: float = 3.0, gaussian_compat: float = 3.0,
                        bilateral_sxy: float = 80.0, bilateral_srgb: float = 13.0,
                        bilateral_compat: float = 10.0) -> np.ndarray:
    """Run DenseCRF2D over ``probs`` (C,H,W, summing to <=1) given the RGB image."""
    if not CRF_AVAILABLE:
        raise RuntimeError(
            "pydensecrf is not installed. Install with: pip install pydensecrf2  "
            "(or pip install cython && pip install git+https://github.com/lucasb-eyer/pydensecrf.git)"
        )
    h, w = image_rgb.shape[:2]
    num_labels = probs.shape[0]
    model = dcrf.DenseCRF2D(w, h, num_labels)

    unary = unary_from_softmax(np.ascontiguousarray(probs.astype(np.float64)))
    model.setUnaryEnergy(unary)
    model.addPairwiseGaussian(sxy=gaussian_sxy, compat=gaussian_compat)
    model.addPairwiseBilateral(
        sxy=bilateral_sxy, srgb=bilateral_srgb, rgbim=np.ascontiguousarray(image_rgb),
        compat=bilateral_compat,
    )
    inference = model.inference(iterations)
    return np.array(inference).reshape((num_labels, h, w))
