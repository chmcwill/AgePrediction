from pathlib import Path

import pytest
from PIL import Image

from age_prediction.services import models
from age_prediction.services import prediction as pred


def _model_weights_available() -> bool:
    root = Path(__file__).resolve().parents[1]
    weights = [models.FACE_EMBEDDOR_WEIGHTS, *models.ENSEMBLE_WEIGHT_FILES]
    return all((root / "best_models" / name).exists() for name in weights)


@pytest.mark.integration
def test_run_prediction_with_real_model(test_image_path, tmp_path):
    assert _model_weights_available(), "Model weights not available in best_models/"

    result, generated = pred.run_prediction(test_image_path, str(tmp_path))

    assert result.big_fig_path is not None
    assert Path(result.big_fig_path).exists()
    assert len(result.fig_paths) == 2
    for path in generated:
        assert Path(path).exists()


@pytest.mark.integration
def test_run_prediction_no_faces_raises(tmp_path):
    img_path = tmp_path / "blank.jpg"
    # Use a size larger than MTCNN min_face_size to avoid internal detect_face errors.
    Image.new("RGB", (200, 200), color="white").save(img_path)

    with pytest.raises(pred.NoFacesFoundError):
        pred.run_prediction(str(img_path), str(tmp_path))


@pytest.mark.integration
def test_run_prediction_non_image_file_raises(tmp_path):
    img_path = tmp_path / "not_an_image.jpg"
    img_path.write_text("this is not an image")

    with pytest.raises(pred.InvalidImageError):
        pred.run_prediction(str(img_path), str(tmp_path))


@pytest.mark.integration
def test_run_prediction_corrupt_image_raises(tmp_path):
    img_path = tmp_path / "corrupt.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"corrupt-data")

    with pytest.raises(pred.InvalidImageError):
        pred.run_prediction(str(img_path), str(tmp_path))
