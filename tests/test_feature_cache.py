import cv2
import numpy as np
import pytest

from lecture2doc import pages
from lecture2doc.vision import Config, analyze, measure_features, small


def test_pages_reuse_features_computed_during_analysis(tmp_path, monkeypatch):
    frames = []
    images = []

    for index in range(5):
        image = np.full((180, 320, 3), 245, dtype=np.uint8)
        cv2.putText(
            image,
            f"lecture {index}",
            (12, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
        )
        cv2.line(image, (15, 110 + index * 3), (280, 110 + index * 3), (30, 30, 30), 2)
        relative_path = f"frame-{index:06d}.png"
        assert cv2.imwrite(str(tmp_path / relative_path), image)
        images.append(image)
        frames.append(
            {
                "id": f"f{index:06d}",
                "time": index * 0.5,
                "pts": index,
                "time_base": "1/2",
                "image": relative_path,
                "width": 320,
                "height": 180,
            }
        )

    config = Config(board="whiteboard", analysis_width=160)
    analyze(tmp_path, frames, config, duration=2.5)

    expected_ink = []
    expected_mask = []
    expected_sharp = []
    for image in images:
        measured = measure_features(small(image, config), config.board)
        expected_ink.append(np.packbits(measured["ink"]))
        expected_mask.append(np.packbits(measured["foreground"]))
        expected_sharp.append(measured["ink_sharpness"])

    def unexpected_recompute(*args, **kwargs):
        pytest.fail("pages.observations recomputed features instead of reusing A's cache")

    monkeypatch.setattr(pages, "measure_features", unexpected_recompute)
    result = pages.observations(tmp_path, tmp_path / "page-stage", frames, config)

    assert np.array_equal(result["ink"], np.asarray(expected_ink))
    assert np.array_equal(result["mask"], np.asarray(expected_mask))
    assert np.allclose(result["sharp"], np.asarray(expected_sharp, dtype=np.float32))
    assert result["shape"] == expected_ink[0].shape[0] * (8,), (
        "packed arrays should be retained alongside the two-dimensional image shape"
    )
