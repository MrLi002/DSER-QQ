"""Behavior checks for paper Sec. III-A and Eqs. (4)-(8); no dataset needed."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from model.ourNet import MyNet
from model.patch_attention import DetailsSup, TransformerBlock
from util.utils_func import custom_erode, mask_reverse, process_mask, pyramid_patch_mask


torch.set_num_threads(1)


@pytest.mark.parametrize("center,neighbors,expected", [(0, 6, 0), (0, 7, 1), (1, 2, 1), (1, 1, 0)])
def test_custom_erosion_requires_more_than_six_disagreements(center, neighbors, expected):
    mask = torch.zeros(1, 5, 5)
    neighbor_positions = [(y, x) for y in range(1, 4) for x in range(1, 4) if (y, x) != (2, 2)]
    for y, x in neighbor_positions[:neighbors]:
        mask[0, y, x] = 1
    mask[0, 2, 2] = center
    original = mask.clone()
    assert custom_erode(mask)[0, 2, 2].item() == expected
    assert torch.equal(mask, original)


def test_event_mask_removes_isolated_noise_and_uses_five_pixel_dilation():
    counts = torch.zeros(4, 15, 15)
    counts[0, 6:9, 6:9] = 1
    counts[1, 1, 1] = 1
    original = counts.clone()
    expected = torch.zeros(1, 15, 15)
    expected[:, 4:11, 4:11] = 1
    torch.testing.assert_close(process_mask(counts), expected)
    assert torch.equal(counts, original)
    batched = torch.stack((counts, torch.zeros_like(counts)))
    actual = process_mask(batched)
    torch.testing.assert_close(actual[0], expected)
    assert torch.count_nonzero(actual[1]) == 0


def test_hcs_counts_binary_event_supported_pixels_before_thresholding():
    endpoint = torch.zeros(1, 3, 32, 96)
    coarse = endpoint.clone()
    events = torch.zeros(1, 1, 32, 96)
    # Three supported pixels fail beta*p = 3.2, however large the RGB error.
    coarse[:, :, 0, :3] = 1
    events[:, :, 0, :3] = 1
    # Four supported pixels pass even with a very small nonzero RGB error.
    coarse[:, :, 0, 32:36] = 1e-6
    events[:, :, 0, 32:36] = 1
    # A changed region and an event elsewhere in its patch do not intersect.
    coarse[:, :, 0, 64:80] = 1
    events[:, :, 1, 64:80] = 1
    pyramid = pyramid_patch_mask(endpoint, coarse, events)
    expected = torch.zeros_like(events)
    expected[:, :, :, 32:64] = 1
    for scale, selected in enumerate(pyramid):
        reference = F.interpolate(expected, scale_factor=1 / (2 ** scale), mode="nearest")
        torch.testing.assert_close(selected, reference)


def test_hcs_threshold_includes_exact_equality():
    endpoint = torch.zeros(1, 3, 32, 32)
    coarse = endpoint.clone()
    coarse[:, :, 0, :4] = 1
    events = coarse[:, :1].clone()
    # beta*p = 4: Eq. (6) uses >=, not >.
    assert torch.all(pyramid_patch_mask(endpoint, coarse, events, beta=0.125)[0] == 1)


def test_asymmetric_selection_routes_opposite_keyframe_support():
    h0 = torch.zeros(1, 1, 4, 4)
    h1 = torch.zeros_like(h0)
    h0[:, :, :2, :2] = 1
    h1[:, :, 2:, 2:] = 1
    frame0_mask, frame1_mask = mask_reverse(h0, h1)
    assert torch.equal(frame0_mask, h1)
    assert torch.equal(frame1_mask, h0)
    assert torch.equal(frame0_mask + frame1_mask, h0 + h1)


def make_attention():
    torch.manual_seed(12)
    return TransformerBlock(patch_size=2, dim=8, channels=4, num_heads=2, dropout=0).eval()


def test_attention_never_reads_unselected_patches_or_writes_unselected_queries():
    block = make_attention()
    support = torch.randn(1, 4, 4, 6, requires_grad=True)
    query = torch.randn_like(support, requires_grad=True)
    mask = torch.zeros(1, 1, 4, 6)
    mask[:, :, :2, :2] = 1
    mask[:, :, 2:, 4:] = 1
    output = block(support, query, mask)
    outside = (mask == 0).expand_as(output)
    assert torch.count_nonzero(output[outside]) == 0
    modified_support = support.detach() + outside * 100
    modified_query = query.detach() - outside * 100
    torch.testing.assert_close(output, block(modified_support, modified_query, mask))
    assert output[~outside].abs().sum() > 0
    output.square().sum().backward()
    for feature in (support, query):
        assert torch.isfinite(feature.grad).all()
        assert torch.count_nonzero(feature.grad[outside]) == 0
        assert feature.grad[~outside].abs().sum() > 0
    assert block.img0_to_kv.weight.grad.abs().sum() > 0
    assert block.img1_to_qv.weight.grad.abs().sum() > 0


def test_empty_attention_and_mixed_batch_are_finite():
    block = make_attention()
    support = torch.randn(2, 4, 4, 4)
    query = torch.randn_like(support, requires_grad=True)
    mask = torch.zeros(2, 1, 4, 4)
    empty = block(support, query, mask)
    assert torch.count_nonzero(empty) == 0
    empty.sum().backward()
    assert torch.isfinite(query.grad).all()
    mask[1, :, :2, :2] = 1
    mixed = block(support, query, mask)
    assert torch.count_nonzero(mixed[0]) == 0
    torch.testing.assert_close(mixed[1:], block(support[1:], query[1:], mask[1:]))
    assert torch.isfinite(mixed).all()


def test_details_module_preserves_global_baseline_outside_selected_regions():
    torch.manual_seed(9)
    block = DetailsSup(patch_size=2, dim=8, channels=8, heads=2).eval()
    feature = torch.randn(2, 8, 4, 4)
    frame0 = torch.randn_like(feature)
    frame1 = torch.randn_like(feature)
    baseline = torch.randn(2, 32, 8, 8, requires_grad=True)
    mask0 = torch.zeros(2, 1, 4, 4)
    mask1 = torch.zeros_like(mask0)
    assert block(feature, frame0, frame1, baseline, mask0, mask1) is baseline
    mask0[1, :, :2, :2] = 1
    output = block(feature, frame0, frame1, baseline, mask0, mask1)
    gate = F.interpolate(mask0, size=(8, 8), mode="nearest").bool().expand_as(output)
    assert torch.equal(output[~gate], baseline[~gate])
    assert not torch.equal(output[gate], baseline[gate])
    output.square().mean().backward()
    assert torch.isfinite(baseline.grad).all()
    assert block.fusion_block[0].weight.grad.abs().sum() > 0


def test_network_forward_and_backward_on_rectangular_input():
    torch.manual_seed(3)
    model = MyNet(SimpleNamespace(bins=8, beta=0.1, mask_patch_size=32))
    images = torch.rand(1, 9, 64, 96)
    voxels = torch.randn(1, 24, 64, 96)
    mask = torch.zeros(1, 1, 64, 96)
    mask[:, :, :32, 32:64] = 1
    outputs = model(images, voxels, mask, 8)
    pure, reference, coarse, predictions, selected, _ = outputs
    assert pure.shape == reference.shape == coarse.shape == (1, 3, 64, 96)
    assert [prediction.shape[-2:] for prediction in predictions] == [(64, 96), (32, 48), (16, 24)]
    assert torch.equal(selected, mask)
    loss = sum(image.square().mean() for image in [reference, coarse, *predictions])
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.event_rec.conv1.weight.grad.abs().sum() > 0
    assert model.sys.refine_block0.fusion_block[0].weight.grad.abs().sum() > 0
    assert model.sys.refine_block1.fusion_block[0].weight.grad.abs().sum() > 0
