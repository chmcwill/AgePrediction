import numpy as np
from PIL import Image

from age_prediction.services import plotting


def test_plot_image_and_pred_returns_figure():
    img = Image.new("RGB", (160, 160), color="white")
    softmax = np.ones((1, 61), dtype=np.float32) / 61.0
    data = plotting.FacePlotData(
        image=img,
        prediction=25.0,
        output_softmax=softmax,
        image_name="test",
        img_input_size=160,
        classification=True,
    )
    fig = plotting.plot_image_and_pred(data)
    assert fig is not None
    fig.clf()


def test_overlay_preds_on_img_returns_fig_and_ax():
    img = Image.new("RGB", (200, 200), color="white")
    overlays = [
        plotting.OverlayItem(bounds=np.array([10, 10, 60, 60]), label=np.array([25])),
        plotting.OverlayItem(bounds=np.array([80, 80, 120, 120]), label="error"),
    ]
    fig, ax = plotting.overlay_preds_on_img(img, overlays)
    assert fig is not None
    assert ax is not None
    fig.clf()
