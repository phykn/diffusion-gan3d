from unittest.mock import patch

import pytest
import torch

from src.data.augment import CriticAugment


def test_all_square_symmetries_are_distinct() -> None:
    maps = CriticAugment().get_index_maps(torch.device("cpu"), 3, 3)
    actual = {tuple(mapping.tolist()) for mapping in maps}

    assert len(actual) == 8


@pytest.mark.parametrize("planes", [True, False, "isotropic", "anisotropic"])
def test_global_augmentation_presets_are_rejected(planes):
    with pytest.raises(ValueError, match="augmentation.planes"):
        CriticAugment(planes=planes)


def test_disabled_augmentation_returns_original_tensors():
    image = torch.randn(2, 3, 4, 4)
    assert CriticAugment().apply_together((image,))[0] is image


def test_pair_shares_one_transform_and_preserves_gradients() -> None:
    previous = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)
    previous.requires_grad_()
    current = previous + 100.0
    augment = CriticAugment(planes={"xy": {"flip_axes": ["x", "y"], "rotate_90": True}})

    with patch.object(
        augment,
        "sample_transforms",
        return_value=torch.tensor([1, 6]),
    ):
        transformed_previous, transformed_current = augment.apply_together(
            (previous, current),
            plane="xy",
        )

    assert torch.equal(
        transformed_current - transformed_previous,
        torch.full_like(transformed_previous, 100.0),
    )
    transformed_previous.sum().backward()
    assert previous.grad is not None
    assert torch.equal(previous.grad, torch.ones_like(previous))


def test_triplet_real_and_fake_share_the_same_transform() -> None:
    real = torch.arange(54, dtype=torch.float32).reshape(2, 3, 1, 3, 3)
    fake = real + 1000.0
    augment = CriticAugment(planes={"xy": {"flip_axes": ["x", "y"], "rotate_90": True}})

    with patch.object(
        augment,
        "sample_transforms",
        return_value=torch.tensor([4, 3]),
    ):
        transformed_real, transformed_fake = augment.apply_together(
            (real, fake),
            plane="xy",
        )

    assert torch.equal(
        transformed_fake - transformed_real,
        torch.full_like(transformed_real, 1000.0),
    )


def test_zero_probability_returns_original_tensors() -> None:
    first = torch.randn(2, 3, 4, 4)
    second = torch.randn(2, 3, 4, 4)
    augment = CriticAugment(planes={"xy": {"flip_axes": ["x", "y"]}}, prob=0.0)

    actual = augment.apply_together((first, second))

    assert actual[0] is first
    assert actual[1] is second


def test_rectangular_inputs_use_only_shape_preserving_transforms() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    augment = CriticAugment(planes={"xy": {"flip_axes": ["x", "y"], "rotate_90": True}})

    with patch("torch.randint", return_value=torch.tensor([0, 2])):
        transforms = augment.sample_transforms(
            2,
            device=inputs.device,
            plane="xy",
            square=False,
        )

    assert all(int(index) in (0, 2, 4, 6) for index in transforms)
    actual = augment.apply_transforms(inputs, transforms)
    assert actual.shape == inputs.shape


def test_rectangular_inputs_reject_quarter_turns() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    augment = CriticAugment(planes={"xy": {"flip_axes": ["x", "y"], "rotate_90": True}})

    with pytest.raises(ValueError, match="shape-preserving"):
        augment.apply_transforms(
            inputs,
            torch.tensor([1, 0]),
        )


def test_horizontal_flip_policy_preserves_pair_alignment_and_gradients() -> None:
    previous = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    previous.requires_grad_()
    current = previous + 100.0
    augment = CriticAugment(planes={"xy": {"flip_axes": ["x"]}}, prob=0.5)

    with patch("torch.rand", return_value=torch.tensor([1.0, 0.0])):
        transforms = augment.sample_transforms(2, device=previous.device, plane="xy")
    assert transforms.tolist() == [0, 4]

    with patch.object(augment, "sample_transforms", return_value=transforms):
        transformed_previous, transformed_current = augment.apply_together(
            (previous, current),
            plane="xy",
        )

    assert transformed_previous.shape == previous.shape
    assert torch.equal(transformed_previous[0], previous[0])
    assert torch.equal(transformed_previous[1], previous[1].flip(-1))
    assert torch.equal(
        transformed_current - transformed_previous,
        torch.full_like(transformed_previous, 100.0),
    )
    transformed_previous.sum().backward()
    assert previous.grad is not None
    assert torch.equal(previous.grad, torch.ones_like(previous))


@pytest.mark.parametrize("plane,flip", [("xy", "x"), ("xz", "x"), ("yz", "y")])
def test_physical_horizontal_flips_preserve_rows_and_transform_conditions_together(
    plane, flip
):
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4).requires_grad_()
    depth = torch.arange(3).reshape(1, 1, 3, 1).expand(2, 1, 3, 4)
    aug = CriticAugment(
        planes={plane: {"flip_axes": [flip], "rotate_90": False}}, preserve_height=True
    )
    image, coordinates, mask = aug.apply_together(
        (inputs, depth, depth > 0), plane=plane
    )
    assert torch.equal(image, inputs.flip(-1))
    assert torch.equal(coordinates, depth)
    assert torch.equal(mask, depth > 0)
    image.sum().backward()
    assert torch.equal(inputs.grad, torch.ones_like(inputs))


def test_mixed_plane_triplets_use_each_planes_physical_axes():
    data = torch.arange(3 * 3 * 2 * 4 * 4).reshape(3, 3, 2, 4, 4).float()
    aug = CriticAugment(
        planes={
            "xy": {"flip_axes": ["y"]},
            "xz": {"flip_axes": ["x"]},
            "yz": {"flip_axes": []},
        },
        preserve_height=True,
    )
    real, fake = aug.apply_together((data, data + 1000), plane=torch.tensor([0, 1, 2]))
    assert torch.equal(real[0], data[0].flip(-2))
    assert torch.equal(real[1], data[1].flip(-1))
    assert torch.equal(real[2], data[2])
    assert torch.equal(fake - real, torch.full_like(data, 1000))


@pytest.mark.parametrize(
    "plane,policy",
    [
        ("xz", {"flip_axes": ["z"]}),
        ("yz", {"flip_axes": ["z"]}),
        ("xz", {"rotate_90": True}),
        ("yz", {"rotate_90": True}),
    ],
)
def test_height_reversals_and_axis_exchanges_are_rejected(plane, policy):
    with pytest.raises(ValueError, match="preserve height"):
        CriticAugment(planes={plane: policy}, preserve_height=True)
    assert CriticAugment(planes={plane: policy}).plane_transforms


def test_plane_rotation_policy_preserves_rectangular_shapes():
    aug = CriticAugment(
        planes={"xy": {"flip_axes": [], "rotate_90": True}}, preserve_height=True
    )
    data = torch.arange(12).reshape(1, 1, 3, 4)
    (out,) = aug.apply_together((data,), plane="xy")
    assert torch.equal(out, data.flip((-2, -1)))


def test_plane_policy_does_not_allow_an_implicit_axis_or_global_mode():
    aug = CriticAugment(planes={"xy": {"flip_axes": []}})
    with pytest.raises(ValueError, match="plane is required"):
        aug.apply_together((torch.zeros(2, 1, 3, 3),))
    with pytest.raises(TypeError, match="mode"):
        CriticAugment(mode="isotropic", planes={"xy": {"flip_axes": []}})
