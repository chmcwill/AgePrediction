from pathlib import Path

import pytest

from age_prediction.services import prediction as pred
from age_prediction.services import models


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
