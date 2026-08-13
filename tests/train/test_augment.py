from unittest.mock import patch

import pytest
import torch

from src.train.augment import (
    ALL_TRANSFORMS,
    ANISOTROPIC_TRANSFORMS,
    SHAPE_PRESERVING_TRANSFORMS,
    CriticAugment,
)


def test_augment_presets_select_expected_transforms() -> None:
    isotropic = CriticAugment("isotropic")
    anisotropic = CriticAugment("anisotropic")

    assert isotropic.allowed_transforms(0) == ALL_TRANSFORMS
    assert anisotropic.allowed_transforms(2) == ANISOTROPIC_TRANSFORMS


def test_all_square_symmetries_are_distinct() -> None:
    image = torch.arange(9).reshape(1, 1, 3, 3)

    actual = {
        tuple(CriticAugment.transform(image, index).flatten().tolist())
        for index in ALL_TRANSFORMS
    }

    assert len(actual) == 8


def test_true_is_rejected_instead_of_guessing_a_preset() -> None:
    with pytest.raises(ValueError, match="use isotropic"):
        CriticAugment(True)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (False, None),
        ("isotropic", "isotropic"),
        ("anisotropic", "anisotropic"),
    ],
)
def test_augment_presets_are_parsed(mode: bool | str, expected: str | None) -> None:
    assert CriticAugment(mode).mode == expected


@pytest.mark.parametrize(
    "mode",
    ("transverse_0", "transverse_1", "transverse_2", "directional"),
)
def test_removed_augment_presets_are_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="anisotropic, isotropic"):
        CriticAugment(mode)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), True])
def test_augment_probability_is_validated(probability: float) -> None:
    with pytest.raises(ValueError, match="augment_prob"):
        CriticAugment("isotropic", probability)


def test_pair_shares_one_transform_and_preserves_gradients() -> None:
    previous = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)
    previous.requires_grad_()
    current = previous + 100.0
    augment = CriticAugment("isotropic", 1.0)

    with patch.object(
        augment,
        "sample_transforms",
        return_value=torch.tensor([1, 6]),
    ):
        transformed_previous, transformed_current = augment.apply_pair(
            previous,
            current,
            axis=0,
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
    axes = torch.tensor([0, 2])
    augment = CriticAugment("isotropic", 1.0)

    with patch.object(
        augment,
        "sample_transforms",
        return_value=torch.tensor([4, 3]),
    ):
        transformed_real, transformed_fake = augment.apply_together(
            (real, fake),
            axes,
        )

    assert torch.equal(
        transformed_fake - transformed_real,
        torch.full_like(transformed_real, 1000.0),
    )


def test_zero_probability_returns_original_tensors() -> None:
    first = torch.randn(2, 3, 4, 4)
    second = torch.randn(2, 3, 4, 4)
    augment = CriticAugment("isotropic", 0.0)

    actual = augment.apply_together((first, second), axis=1)

    assert actual[0] is first
    assert actual[1] is second


def test_rectangular_inputs_use_only_shape_preserving_transforms() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    augment = CriticAugment("isotropic", 1.0)

    with patch("torch.randint", return_value=torch.tensor([1, 3])):
        transforms = augment.sample_transforms(
            torch.tensor([0, 2]),
            square=False,
        )

    assert all(int(index) in SHAPE_PRESERVING_TRANSFORMS for index in transforms)
    actual = augment.apply_transforms(inputs, transforms)
    assert actual.shape == inputs.shape


def test_rectangular_inputs_reject_quarter_turns() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    augment = CriticAugment("isotropic", 1.0)

    with pytest.raises(ValueError, match="shape-preserving"):
        augment.apply_transforms(
            inputs,
            torch.tensor([1, 0]),
        )


def test_anisotropic_pair_uses_only_left_right_flips_and_preserves_gradients() -> None:
    previous = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    previous.requires_grad_()
    current = previous + 100.0
    augment = CriticAugment("anisotropic", 1.0)

    with patch("torch.randint", return_value=torch.tensor([0, 1])):
        transforms = augment.sample_transforms(torch.tensor([0, 2]))
    assert transforms.tolist() == [0, 4]

    with patch.object(augment, "sample_transforms", return_value=transforms):
        transformed_previous, transformed_current = augment.apply_pair(
            previous,
            current,
            axis=0,
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
