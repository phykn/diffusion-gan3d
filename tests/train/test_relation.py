import torch
import torch.nn.functional as F

from src.anchor import PlaneAnchor, build_anchors
from src.train.relation import (
    RelationBank,
    RelationCurve,
    RelationTarget,
    morphology_descriptor,
    relation_curve,
    relation_interval_loss,
)


def test_relation_curve_removes_chance_phase_overlap_at_every_distance() -> None:
    probs = torch.full((2, 4, 3, 3), 0.5)
    roi = torch.ones(3, 3, dtype=torch.bool)

    curve = relation_curve(probs, axis=0, index=0, roi=roi)

    assert curve.values.shape == (3, 2, 2)
    assert curve.valid.tolist() == [True, True, True]
    torch.testing.assert_close(curve.values, torch.zeros_like(curve.values))


def test_persistent_phase_support_has_positive_diagonal_relation() -> None:
    checker = torch.tensor(
        ((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0))
    )
    labels = checker.repeat(4, 1, 1)
    probs = F.one_hot(labels, num_classes=2).movedim(-1, 0).to(torch.float32)

    curve = relation_curve(
        probs,
        axis=0,
        index=0,
        roi=torch.ones(4, 4, dtype=torch.bool),
    )

    expected = torch.tensor(((0.25, -0.25), (-0.25, 0.25)))
    torch.testing.assert_close(curve.values, expected.expand_as(curve.values))


def test_relation_gradient_changes_neighbors_but_not_the_center() -> None:
    logits = torch.randn(2, 4, 3, 3, requires_grad=True)
    probs = logits.softmax(dim=0)
    curve = relation_curve(
        probs,
        axis=0,
        index=1,
        roi=torch.ones(3, 3, dtype=torch.bool),
    )

    curve.values.square().sum().backward()

    assert logits.grad is not None
    assert not bool(logits.grad[:, 1].any())
    assert bool(logits.grad[:, (0, 2, 3)].abs().sum() > 0.0)


def test_relation_curve_is_invariant_to_a_shared_spatial_permutation() -> None:
    probs = torch.randn(3, 5, 4, 4).softmax(dim=0)
    roi = torch.ones(4, 4, dtype=torch.bool)

    original = relation_curve(probs, axis=0, index=2, roi=roi)
    permuted = relation_curve(probs.flip((-2, -1)), axis=0, index=2, roi=roi)

    torch.testing.assert_close(original.values, permuted.values)
    torch.testing.assert_close(original.descriptor, permuted.descriptor)


def test_relation_interval_is_zero_inside_and_pushes_outside_values_back() -> None:
    valid = torch.tensor((True, True))
    weights = torch.tensor((0.75, 0.25))
    target = RelationTarget(
        lower=torch.full((2, 2, 2), -0.1),
        upper=torch.full((2, 2, 2), 0.1),
        weights=weights,
        valid=valid,
        ood_weight=1.0,
        source="shared",
    )
    inside_values = torch.zeros(2, 2, 2, requires_grad=True)
    inside = RelationCurve(
        inside_values,
        valid,
        torch.zeros(3),
        roi_pixels=16,
    )
    outside_values = torch.full((2, 2, 2), 0.2, requires_grad=True)
    outside = RelationCurve(
        outside_values,
        valid,
        torch.zeros(3),
        roi_pixels=16,
    )

    inside_loss = relation_interval_loss(inside, target)
    outside_loss = relation_interval_loss(outside, target)
    outside_loss.backward()

    assert float(inside_loss.detach()) == 0.0
    assert float(outside_loss.detach()) > 0.0
    assert outside_values.grad is not None
    assert bool((outside_values.grad > 0.0).all())


def test_frozen_bank_uses_domain_then_shared_fallback_without_storing_images() -> None:
    bank = _bank(axes=(0, 1))
    probs = _persistent_probs()
    for _ in range(3):
        bank.add(
            probs,
            condition_domain=0,
            source_domain=0,
            owned_axes=(0,),
        )

    assert bank.entry_count == 4
    assert bank.ready_bucket_count == 2
    assert bank.needs_data(
        condition_domain=0,
        source_domain=0,
        owned_axes=(1,),
    )
    stored = tuple(bank._domains[0][0].entries)
    assert all(entry.values.ndim == 3 for entry in stored)
    assert all(entry.descriptor.ndim == 1 for entry in stored)

    condition = _checker_condition()
    visible = torch.tensor((True,))
    domain_result = bank.loss(probs, condition, visible, domain=0)
    shared_result = bank.loss(probs, condition, visible, domain=1)

    assert domain_result.matches == domain_result.domain_matches == 1
    assert shared_result.matches == shared_result.shared_matches == 1
    assert float(domain_result.loss) == 0.0
    assert float(shared_result.loss) == 0.0


def test_relation_bank_rejects_global_ood_anchor_and_hidden_anchor() -> None:
    bank = _bank(axes=(0,))
    probs = _persistent_probs()
    for _ in range(2):
        bank.add(
            probs,
            condition_domain=0,
            source_domain=0,
            owned_axes=(0,),
        )
    ood = build_anchors(
        (PlaneAnchor(torch.zeros(4, 4, dtype=torch.uint8), axis=0, index=1),),
        batch_size=1,
        num_phases=2,
        volume_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert ood is not None

    rejected = bank.loss(probs, ood, torch.tensor((True,)), domain=1)
    hidden = bank.loss(probs, _checker_condition(), torch.tensor((False,)), domain=0)

    assert rejected.queries == 1
    assert rejected.matches == 0
    assert float(rejected.loss) == 0.0
    assert hidden.queries == hidden.matches == 0
    assert float(hidden.loss) == 0.0


def test_descriptor_ignores_pixels_outside_partial_roi() -> None:
    first = torch.randn(2, 4, 4).softmax(dim=0)
    second = first.clone()
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[1:3, 1:3] = True
    second[:, ~mask] = torch.randn_like(second[:, ~mask]).softmax(dim=0)

    torch.testing.assert_close(
        morphology_descriptor(first, mask),
        morphology_descriptor(second, mask),
    )


def _bank(*, axes: tuple[int, ...]) -> RelationBank:
    return RelationBank(
        num_domains=2,
        num_phases=2,
        axes=axes,
        capacity_per_axis=2,
        profiles_per_axis=1,
        neighbors=2,
        quantile_low=0.1,
        quantile_high=0.9,
    )


def _persistent_probs() -> torch.Tensor:
    checker = torch.tensor(
        ((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0))
    )
    labels = checker.repeat(4, 1, 1).unsqueeze(0)
    return F.one_hot(labels, num_classes=2).movedim(-1, 1).to(torch.float32)


def _checker_condition():
    checker = torch.tensor(
        ((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)),
        dtype=torch.uint8,
    )
    condition = build_anchors(
        (PlaneAnchor(checker, axis=0, index=1),),
        batch_size=1,
        num_phases=2,
        volume_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert condition is not None
    return condition
