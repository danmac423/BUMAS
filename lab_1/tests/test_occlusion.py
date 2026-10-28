import pytest
import torch

from lab_1.data import get_device, load_images, load_labels, load_model
from lab_1.occlusion import box, compute_deltas, heatmap, make_occluded_batch


@pytest.fixture
def img():
    g = torch.Generator().manual_seed(0)
    return torch.rand(3, 64, 64, generator=g) * 0.4  # values < 0.5, so fill is distinguishable


def test_batch_shape_and_count(img):
    batch, coords = make_occluded_batch(img, win=20, stride=2)
    assert batch.shape == (23 * 23, 3, 64, 64)
    assert coords.shape == (23 * 23, 2)


@pytest.mark.parametrize("win", [16, 20, 24])
def test_windows_cover_whole_image_and_stay_inside(img, win):
    _, coords = make_occluded_batch(img, win=win, stride=2)
    assert coords.min() == 0
    assert coords.max() == 64 - win


def test_coords_order_is_row_major_xy(img):
    _, coords = make_occluded_batch(img, win=20, stride=2)
    assert coords[0].tolist() == [0, 0]
    assert coords[1].tolist() == [2, 0]  # x advances first
    assert coords[23].tolist() == [0, 2]  # then y
    assert coords[-1].tolist() == [44, 44]


def test_each_mask_is_exact_square(img):
    win = 20
    batch, coords = make_occluded_batch(img, win=win, stride=2)
    for k in (0, 1, 23, 100, len(coords) - 1):
        x, y = coords[k].tolist()
        changed = (batch[k] != img).any(dim=0)  # (H, W)
        assert changed.sum() == win * win
        assert changed[y : y + win, x : x + win].all()
        assert (batch[k][:, y : y + win, x : x + win] == 0.5).all()


def test_pixels_outside_window_unchanged(img):
    batch, coords = make_occluded_batch(img, win=20, stride=2)
    x, y = coords[150].tolist()
    keep = torch.ones(64, 64, dtype=torch.bool)
    keep[y : y + 20, x : x + 20] = False
    assert torch.equal(batch[150][:, keep], img[:, keep])


def test_custom_fill_and_input_not_modified(img):
    before = img.clone()
    batch, _ = make_occluded_batch(img, win=16, stride=4, fill=0.9)
    assert torch.equal(img, before)
    assert (batch[0][:, :16, :16] == 0.9).all()


DEVICE = get_device()


class _MeanOfRegion(torch.nn.Module):
    """Two 'classes': class 0 = mean over the top-left 8x8, class 1 = constant. Easy to reason about."""

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        c0 = x[:, :, :8, :8].mean(dim=(1, 2, 3))
        return torch.stack([c0, torch.full_like(c0, -1.0)], dim=1)


def test_deltas_shape_and_toy_model_values():
    img = torch.ones(3, 64, 64)
    delta = compute_deltas(_MeanOfRegion(), img, win=16, stride=4, fill=0.0)
    assert delta.shape == (13, 13)
    # window at (0,0) zeroes the whole 8x8 region -> drop of 1.0; window at x>=8 or y>=8 changes nothing
    assert delta[0, 0] == pytest.approx(1.0)
    assert delta[3, 3] == pytest.approx(0.0)  # x=y=12 is outside the 8x8 region


def test_deltas_toy_model_partial_overlap():
    img = torch.ones(3, 64, 64)
    delta = compute_deltas(_MeanOfRegion(), img, win=16, stride=4, fill=0.0)
    # x=4: window columns 4..19 hide cols 4..7 of the 8x8 region -> half of it, rows fully hidden -> 0.5
    assert delta[0, 1] == pytest.approx(0.5)
    assert delta[1, 1] == pytest.approx(0.25)  # y=4 too -> a quarter


def test_deltas_chunking_does_not_change_result():
    img = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(1))
    a = compute_deltas(_MeanOfRegion(), img, win=16, stride=4, chunk=7)
    b = compute_deltas(_MeanOfRegion(), img, win=16, stride=4, chunk=10_000)
    assert torch.allclose(a, b)


def test_deltas_match_manual_computation_on_real_model():
    model = load_model(DEVICE)
    labels = load_labels("train")
    img = load_images("train", labels.id[:1])[0].to(DEVICE)
    delta = compute_deltas(model, img, win=20, stride=2)
    assert delta.shape == (23, 23)

    x, y = 14, 6  # window index (i=7, j=3)
    manual = img.clone()
    manual[:, y : y + 20, x : x + 20] = 0.5
    with torch.no_grad():
        z = model(torch.stack([img, manual]))
    k = z[0].argmax()
    assert delta[3, 7].item() == pytest.approx((z[0, k] - z[1, k]).item(), abs=1e-3)


def test_heatmap_shape():
    assert heatmap(torch.ones(23, 23), win=20, stride=2).shape == (64, 64)
    assert heatmap(torch.ones(13, 13), win=16, stride=4).shape == (64, 64)


def test_heatmap_constant_delta_is_constant():
    assert torch.allclose(heatmap(torch.full((23, 23), 3.0)), torch.full((64, 64), 3.0))


def test_heatmap_single_window_in_corner():
    delta = torch.zeros(23, 23)
    delta[0, 0] = 5.0
    h = heatmap(delta)
    assert h[0, 0] == 5.0  # only window (0,0) covers this pixel
    assert h[63, 63] == 0.0
    assert h.max() == 5.0


def test_heatmap_averages_overlap():
    delta = torch.zeros(23, 23)
    delta[0, 0], delta[0, 1] = 2.0, 4.0  # windows at x=0 and x=2, y=0
    h = heatmap(delta)
    assert h[0, 0] == pytest.approx(2.0)  # only the first covers x=0,1
    assert h[0, 3] == pytest.approx(3.0)  # x=3 is covered by exactly the windows at x=0 and x=2 -> (2+4)/2
    assert h[0, 5] == pytest.approx(2.0)  # x=5 also by the zero window at x=4 -> (2+4+0)/3


def test_box_rectangle_max_exclusive():
    heat = torch.zeros(64, 64)
    heat[10:30, 5:25] = 1.0  # rows 10..29, cols 5..24
    assert box(heat, 0.4) == (5, 10, 25, 30)


def test_box_threshold_is_relative_to_max():
    heat = torch.zeros(64, 64)
    heat[10:20, 10:20] = 10.0
    heat[20:25, 10:20] = 5.0  # 50% of max
    assert box(heat, 0.4) == (10, 10, 20, 25)
    assert box(heat, 0.6) == (10, 10, 20, 20)


def test_box_single_pixel_and_edge():
    heat = torch.zeros(64, 64)
    heat[63, 63] = 1.0
    assert box(heat) == (63, 63, 64, 64)


def test_box_empty_map_returns_none():
    assert box(torch.zeros(64, 64)) == (0, 0, 0, 0)


def test_box_ignores_non_contiguous_pixels():
    heat = torch.zeros(64, 64)
    heat[5, 5] = 1.0
    heat[40, 50] = 1.0
    assert box(heat) == (5, 5, 51, 41)
