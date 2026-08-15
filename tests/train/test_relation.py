import torch
import torch.nn.functional as F

from src.anchor import PlaneAnchor, build_anchors
from src.train.relation import (
    RelationBank,
    RelationLoss,
    _RelationEntry,
    _sample_profile_centers,
)
from src.train.relation_math import (
    RelationCurve,
    RelationPenalty,
    RelationTarget,
    _data_driven_distance_weights,
    _matched_directional_mean,
    _summarize_phase_target,
    _summarize_support_target,
    _support_statistics,
    matching_descriptor,
    morphology_descriptor,
    relation_curve,
    relation_interval_loss,
    relation_penalty,
    support_relation,
)


def test_empty_relation_loss_owns_zero_diagnostic_shapes() -> None:
    result = RelationLoss.empty(torch.tensor(1.0), distances=3, phases=2)

    assert float(result.loss) == 0.0
    assert result.distance_weights.shape == (3,)
    assert result.phase_distance_weights.shape == (3, 2)
    assert result.displacement_quantile.shape == (3, 2, 2)
    assert result.dilation_radius.shape == (3, 2)
    assert bool((result.displacement_quantile == -1.0).all())
    assert bool((result.dilation_radius == -1.0).all())
    assert result.queries == result.matches == 0


def test_relation_curve_removes_chance_phase_overlap_at_every_distance() -> None:
    probs = torch.full((2, 4, 3, 3), 0.5)
    roi = torch.ones(3, 3, dtype=torch.bool)

    curve = relation_curve(probs, axis=0, index=0, roi=roi)

    assert curve.values.shape == (2, 3, 2, 2)
    assert curve.valid.tolist() == [
        [False, False, False],
        [True, True, True],
    ]
    torch.testing.assert_close(curve.values, torch.zeros_like(curve.values))
    torch.testing.assert_close(
        curve.independent[1],
        torch.full_like(curve.independent[1], 0.25),
    )


def test_persistent_phase_support_has_positive_diagonal_relation() -> None:
    checker = torch.tensor(((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)))
    labels = checker.repeat(4, 1, 1)
    probs = F.one_hot(labels, num_classes=2).movedim(-1, 0).to(torch.float32)

    curve = relation_curve(
        probs,
        axis=0,
        index=0,
        roi=torch.ones(4, 4, dtype=torch.bool),
    )

    expected = torch.tensor(((0.25, -0.25), (-0.25, 0.25)))
    torch.testing.assert_close(
        curve.values[0],
        torch.zeros_like(curve.values[0]),
    )
    torch.testing.assert_close(
        curve.values[1],
        expected.expand_as(curve.values[1]),
    )


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
    empty_support = torch.empty(2, 0, 2, 2)
    target = RelationTarget(
        lower=torch.full((2, 2, 2), -0.1),
        upper=torch.full((2, 2, 2), 0.1),
        weights=weights,
        valid=valid,
        support_lower=empty_support,
        support_upper=empty_support,
        support_weights=torch.empty(2, 0),
        support_valid=torch.empty(2, 0, 2, 2, dtype=torch.bool),
        ood_weight=1.0,
        source="shared",
    )
    inside_values = torch.zeros(2, 2, 2, 2, requires_grad=True)
    inside = RelationCurve(
        inside_values,
        valid.expand(2, -1),
        torch.empty(2, 2, 0, 2, 2),
        torch.empty(2, 2, 0, 2, 2, dtype=torch.bool),
        torch.zeros(7),
        roi_pixels=16,
    )
    outside_values = torch.full((2, 2, 2, 2), 0.2, requires_grad=True)
    outside = RelationCurve(
        outside_values,
        valid.expand(2, -1),
        torch.empty(2, 2, 0, 2, 2),
        torch.empty(2, 2, 0, 2, 2, dtype=torch.bool),
        torch.zeros(7),
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
    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        bank.add(
            probs,
            condition_domain=0,
            source_domain=0,
            owned_axes=(0,),
            generator=generator,
        )

    assert bank.entry_count == 4
    assert bank.ready_bucket_count == 2
    assert bank.needs_data(
        condition_domain=0,
        source_domain=0,
        owned_axes=(1,),
    )
    stored = tuple(bank._domains[0][0].entries)
    assert all(entry.values.ndim == 4 for entry in stored)
    assert all(entry.support.ndim == 5 for entry in stored)
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


def test_matching_descriptor_uses_same_hard_morphology_for_soft_reference() -> None:
    labels = torch.tensor(((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)))
    hard = F.one_hot(labels, num_classes=2).movedim(-1, 0).to(torch.float32)
    soft = 0.6 * hard + 0.4 * (1.0 - hard)
    roi = torch.ones(4, 4, dtype=torch.bool)

    torch.testing.assert_close(
        matching_descriptor(soft, roi),
        matching_descriptor(hard, roi),
    )


def test_reference_bank_samples_the_observed_anchor_roi_geometry() -> None:
    bank = _bank(axes=(0,))
    bank.observe_shape(axis=0, height=2, width=3, volume_size=4)

    bank.add(
        _persistent_probs(),
        condition_domain=0,
        source_domain=0,
        owned_axes=(0,),
    )

    geometry = bank._domains[0][0].entries[0].descriptor[-4:].float()
    torch.testing.assert_close(
        geometry,
        torch.tensor((3 / 8, 1 / 2, 3 / 4, 0.0)),
    )


def test_one_sided_relation_failure_is_not_hidden_by_direction_averaging() -> None:
    expected = torch.tensor(((0.25, -0.25), (-0.25, 0.25)))
    curve_values = expected.expand(2, 1, 2, 2).clone()
    curve_values[1].zero_()
    curve = RelationCurve(
        values=curve_values,
        valid=torch.ones(2, 1, dtype=torch.bool),
        support=torch.empty(2, 1, 0, 2, 2),
        support_valid=torch.empty(2, 1, 0, 2, 2, dtype=torch.bool),
        descriptor=torch.zeros(7),
        roi_pixels=16,
    )
    target = RelationTarget(
        lower=(expected - 0.01).unsqueeze(0),
        upper=(expected + 0.01).unsqueeze(0),
        weights=torch.ones(1),
        valid=torch.ones(1, dtype=torch.bool),
        support_lower=torch.empty(1, 0, 2, 2),
        support_upper=torch.empty(1, 0, 2, 2),
        support_weights=torch.empty(1, 0),
        support_valid=torch.empty(1, 0, 2, 2, dtype=torch.bool),
        ood_weight=1.0,
        source="shared",
    )

    penalty: RelationPenalty = relation_penalty(
        curve,
        target,
        phase_weight=1.0,
        support_weight=0.0,
        direction_reduction="mean",
    )

    assert float(penalty.minus) == 0.0
    assert float(penalty.plus) > 0.0
    torch.testing.assert_close(penalty.loss, 0.5 * penalty.plus)


def test_support_relation_allows_a_one_voxel_lateral_shift() -> None:
    center_labels = torch.zeros(8, 8, dtype=torch.long)
    center_labels[2:6, 2:5] = 1
    neighbor_labels = torch.zeros_like(center_labels)
    neighbor_labels[2:6, 3:6] = 1
    center = F.one_hot(center_labels, num_classes=2).movedim(-1, 0).float()
    neighbor_logits = (
        F.one_hot(neighbor_labels, num_classes=2)
        .movedim(-1, 0)
        .float()
        .requires_grad_()
    )
    roi = torch.ones(8, 8, dtype=torch.bool)

    values, valid = support_relation(
        center,
        neighbor_logits.unsqueeze(0),
        roi,
        max_radius=1,
    )
    values[valid].sum().backward()

    # Radius slot zero is exact-coordinate support; slot one allows a
    # one-voxel Chebyshev displacement.
    assert bool(valid[0, 1, 1].all())
    assert bool((values[0, 1, 1] > 0.99).all())
    assert neighbor_logits.grad is not None
    assert bool(neighbor_logits.grad.abs().sum() > 0.0)


def test_long_range_teacher_dependence_keeps_far_distance_weight() -> None:
    statistics = _phase_statistics(
        torch.tensor(
            (
                (1.0, 1.0),
                (0.8, 0.8),
                (0.6, 0.6),
                (0.5, 0.5),
            )
        )
    )

    torch.testing.assert_close(
        statistics.weights.sum(dim=1),
        torch.ones(2, 2),
    )
    assert bool((statistics.weights[:, -1] > 0.1).all())


def test_short_range_teacher_dependence_suppresses_far_distance_weight() -> None:
    statistics = _phase_statistics(
        torch.tensor(
            (
                (1.0, 1.0),
                (0.2, 0.2),
                (0.01, 0.01),
                (0.0, 0.0),
            )
        )
    )

    assert bool((statistics.weights[:, -1] < 1e-6).all())
    assert bool((statistics.weights[:, 0] > statistics.weights[:, 1]).all())


def test_each_phase_learns_its_own_normalized_distance_profile() -> None:
    statistics = _phase_statistics(
        torch.tensor(
            (
                (1.0, 1.0),
                (0.8, 0.1),
                (0.6, 0.0),
                (0.5, 0.0),
            )
        )
    )

    torch.testing.assert_close(
        statistics.weights.sum(dim=1),
        torch.ones(2, 2),
    )
    assert bool((statistics.weights[:, -1, 0] > 0.1).all())
    assert bool((statistics.weights[:, -1, 1] == 0.0).all())


def test_zero_dependence_does_not_fall_back_to_uniform_distance_weights() -> None:
    weights = _data_driven_distance_weights(
        torch.zeros(3, 2),
        torch.ones(3, 2),
        torch.ones(3, 2, dtype=torch.bool),
        uncertainty_floor=0.01,
    )

    torch.testing.assert_close(weights, torch.zeros_like(weights))


def test_teacher_lateral_shift_increases_learned_dilation_radius() -> None:
    support = _shifted_support_statistics((0, 1, 3), max_radius=4)
    target = _support_target(support)

    radii = target.dilation_radius[0, :, 1]
    assert radii.tolist() == sorted(radii.tolist())
    assert int(radii[0]) == 0
    assert int(radii[-1]) >= 2
    torch.testing.assert_close(
        target.weights[:, :, 1].sum(dim=1),
        torch.ones(2),
    )


def test_displacement_overflow_disables_support_target() -> None:
    support = _shifted_support_statistics((6,), max_radius=2)
    target = _support_target(support)

    assert int(target.dilation_radius[0, 0, 1]) == -1
    assert not bool(target.valid[0, 0, 1].any())
    assert float(target.weights[0, 0, 1]) == 0.0


def test_learned_radius_does_not_penalize_matching_shifted_support() -> None:
    support = _shifted_support_statistics((2,), max_radius=3)
    learned = _support_target(support)
    distance_count = support.values.shape[0]
    phase_values = torch.zeros(2, distance_count, 2, 2)
    curve = RelationCurve(
        values=phase_values,
        valid=torch.ones(2, distance_count, dtype=torch.bool),
        support=support.values.unsqueeze(0).expand(2, -1, -1, -1, -1),
        support_valid=support.valid.unsqueeze(0).expand(2, -1, -1, -1, -1),
        descriptor=torch.zeros(7),
        roi_pixels=16 * 16,
        independent=torch.zeros_like(phase_values),
        phase_valid=torch.zeros(2, distance_count, 2, dtype=torch.bool),
    )
    target = RelationTarget(
        lower=torch.zeros(distance_count, 2, 2),
        upper=torch.zeros(distance_count, 2, 2),
        weights=torch.zeros(distance_count),
        valid=torch.zeros(distance_count, dtype=torch.bool),
        support_lower=learned.lower,
        support_upper=learned.upper,
        support_weights=learned.weights,
        support_valid=learned.valid,
        ood_weight=1.0,
        source="shared",
        phase_distance_weights=torch.zeros(distance_count, 2),
        phase_valid=torch.zeros(distance_count, 2, dtype=torch.bool),
        displacement_quantile=learned.displacement_quantile,
        dilation_radius=learned.dilation_radius,
        support_strength=learned.strength,
        support_uncertainty=learned.uncertainty,
    )

    penalty = relation_penalty(
        curve,
        target,
        phase_weight=0.0,
        support_weight=1.0,
        direction_reduction="mean",
    )

    torch.testing.assert_close(penalty.loss, torch.zeros_like(penalty.loss))
    assert int(target.dilation_radius[0, 0, 1]) >= 1


def test_rare_phase_below_minimum_mass_is_invalid() -> None:
    labels = torch.zeros(2, 16, 16, dtype=torch.long)
    labels[:, 4, 4:11] = 1
    probs = F.one_hot(labels, num_classes=2).movedim(-1, 0).float()

    curve = relation_curve(
        probs,
        axis=0,
        index=0,
        roi=torch.ones(16, 16, dtype=torch.bool),
        support_max_radius=1,
    )

    assert not bool(curve.phase_valid[1, 0, 1].any())
    assert not bool(curve.phase_valid[1, 0, :, 1].any())
    assert not bool(curve.support_valid[1, 0, :, 1].any())


def test_dominant_phase_chance_baseline_has_finite_values_and_gradients() -> None:
    center = (
        F.one_hot(
            torch.zeros(16, 16, dtype=torch.long),
            num_classes=2,
        )
        .movedim(-1, 0)
        .float()
    )
    neighbor = center.clone().requires_grad_()

    values, valid = support_relation(
        center,
        neighbor.unsqueeze(0),
        torch.ones(16, 16, dtype=torch.bool),
        max_radius=2,
    )
    values.sum().backward()

    assert bool(torch.isfinite(values).all())
    assert not bool(valid.any())
    assert neighbor.grad is not None
    assert bool(torch.isfinite(neighbor.grad).all())


def test_asymmetric_teacher_targets_keep_minus_and_plus_intervals_separate() -> None:
    minus = torch.tensor(((0.2, -0.2), (-0.2, 0.2)))
    plus = -minus
    samples = torch.stack((minus, plus)).view(1, 2, 1, 2, 2).expand(4, -1, -1, -1, -1)
    statistics = _summarize_phase_target(
        samples,
        torch.zeros_like(samples),
        torch.ones_like(samples, dtype=torch.bool),
        quantile_low=0.1,
        quantile_high=0.9,
        uncertainty_floor=0.01,
    )
    empty_curve_support = torch.empty(2, 1, 0, 2, 2)
    curve = RelationCurve(
        values=torch.stack((minus, plus)).unsqueeze(1),
        valid=torch.ones(2, 1, dtype=torch.bool),
        support=empty_curve_support,
        support_valid=torch.empty_like(empty_curve_support, dtype=torch.bool),
        descriptor=torch.zeros(7),
        roi_pixels=64,
        phase_valid=torch.ones(2, 1, 2, 2, dtype=torch.bool),
    )
    target = RelationTarget(
        lower=statistics.lower,
        upper=statistics.upper,
        weights=torch.ones(1),
        valid=torch.ones(1, dtype=torch.bool),
        support_lower=torch.empty(1, 0, 2, 2),
        support_upper=torch.empty(1, 0, 2, 2),
        support_weights=torch.empty(1, 0),
        support_valid=torch.empty(1, 0, 2, 2, dtype=torch.bool),
        ood_weight=1.0,
        source="shared",
        uncertainty=statistics.uncertainty,
        phase_distance_weights=statistics.weights,
        phase_valid=statistics.valid,
    )

    matching = relation_penalty(
        curve,
        target,
        phase_weight=1.0,
        support_weight=0.0,
        direction_reduction="mean",
    )
    mismatched_curve = RelationCurve(
        values=torch.stack((minus, minus)).unsqueeze(1),
        valid=curve.valid,
        support=curve.support,
        support_valid=curve.support_valid,
        descriptor=curve.descriptor,
        roi_pixels=curve.roi_pixels,
        phase_valid=curve.phase_valid,
    )
    mismatched = relation_penalty(
        mismatched_curve,
        target,
        phase_weight=1.0,
        support_weight=0.0,
        direction_reduction="mean",
    )

    assert float(matching.loss) == 0.0
    assert float(mismatched.minus) == 0.0
    assert float(mismatched.plus) > 0.0


def test_phase_interval_loss_is_invariant_to_relation_dimension_count() -> None:
    two_phase = _uniform_phase_violation(2)
    four_phase = _uniform_phase_violation(4)

    torch.testing.assert_close(two_phase, four_phase)


def test_relation_curve_supports_three_phases_with_feature_validity() -> None:
    probs = torch.randn(3, 5, 8, 8).softmax(dim=0)

    curve = relation_curve(
        probs,
        axis=0,
        index=2,
        roi=torch.ones(8, 8, dtype=torch.bool),
    )

    assert curve.values.shape == (2, 4, 3, 3)
    assert curve.phase_valid.shape == curve.values.shape
    assert bool(torch.isfinite(curve.values).all())


def test_student_relation_curve_skips_teacher_displacement_collection() -> None:
    labels = torch.randint(0, 2, (4, 12, 12))
    probs = F.one_hot(labels, num_classes=2).movedim(-1, 0).float()

    curve = relation_curve(
        probs,
        axis=0,
        index=1,
        roi=torch.ones(12, 12, dtype=torch.bool),
        support_max_radius=2,
        collect_displacement=False,
    )

    assert curve.support.shape[2] == 3
    assert curve.displacement_hist.shape[-1] == 0
    assert not bool(curve.displacement_valid.any())


def test_directional_diagnostics_ignore_invalid_sentinel_values() -> None:
    values = torch.tensor([[[[2.0]], [[-1.0]]]])
    valid = torch.tensor([[[[True]], [[False]]]])

    result = _matched_directional_mean(values, valid, invalid_value=-1.0)

    torch.testing.assert_close(result, torch.tensor([[2.0]]))


def test_bank_uses_nearest_valid_references_for_far_distances() -> None:
    bank = RelationBank(
        num_domains=1,
        num_phases=2,
        axes=(0,),
        capacity_per_axis=12,
        profiles_per_axis=1,
        neighbors=4,
        quantile_low=0.1,
        quantile_high=0.9,
        support_max_radius=0,
    )
    _add_far_reference_entries(bank, far_descriptor=0.0)

    target, status = bank._find_target(
        torch.zeros(7),
        axis=0,
        domain=0,
        roi_pixels=64,
        device=torch.device("cpu"),
    )

    assert status == "matched"
    assert target is not None
    assert bool(target.phase_valid[:, -1].any())
    torch.testing.assert_close(
        target.phase_distance_weights[:, -1],
        torch.ones(2, 2),
    )


def test_bank_excludes_morphology_ood_fallbacks_for_far_features() -> None:
    bank = RelationBank(
        num_domains=1,
        num_phases=2,
        axes=(0,),
        capacity_per_axis=12,
        profiles_per_axis=1,
        neighbors=4,
        quantile_low=0.1,
        quantile_high=0.9,
        support_max_radius=0,
    )
    _add_far_reference_entries(bank, far_descriptor=10.0)

    target, status = bank._find_target(
        torch.zeros(7),
        axis=0,
        domain=0,
        roi_pixels=64,
        device=torch.device("cpu"),
    )

    assert target is None
    assert status == "missing"


def test_profile_center_sampling_stratifies_both_volume_endpoints() -> None:
    centers = _sample_profile_centers(
        8,
        4,
        generator=torch.Generator().manual_seed(7),
    )

    assert centers[:2] == [0, 7]
    assert len(set(centers)) == 4
    assert all(0 < center < 7 for center in centers[2:])


def test_stratified_bank_has_full_knn_support_at_both_far_directions() -> None:
    bank = RelationBank(
        num_domains=1,
        num_phases=2,
        axes=(0,),
        capacity_per_axis=16,
        profiles_per_axis=4,
        neighbors=4,
        quantile_low=0.1,
        quantile_high=0.9,
        support_max_radius=0,
    )
    grid = torch.arange(8).view(8, 1) + torch.arange(8).view(1, 8)
    labels = grid.remainder(2).repeat(8, 1, 1).unsqueeze(0)
    probs = F.one_hot(labels, num_classes=2).movedim(-1, 1).float()
    generator = torch.Generator().manual_seed(11)
    for _ in range(4):
        bank.add(
            probs,
            condition_domain=0,
            source_domain=0,
            owned_axes=(0,),
            generator=generator,
        )
    descriptor = matching_descriptor(
        probs[0, :, 0],
        torch.ones(8, 8, dtype=torch.bool),
    )

    target, status = bank._find_target(
        descriptor,
        axis=0,
        domain=0,
        roi_pixels=64,
        device=torch.device("cpu"),
    )

    assert status == "matched"
    assert target is not None
    torch.testing.assert_close(
        target.valid_count[:, -1],
        torch.full((2, 2), bank.neighbors, dtype=torch.long),
    )
    assert bool((target.phase_distance_weights[:, -1] > 0.0).all())


def _add_far_reference_entries(
    bank: RelationBank,
    *,
    far_descriptor: float,
) -> None:
    relation = torch.tensor(((0.2, -0.2), (-0.2, 0.2)))
    for entry_index in range(bank.capacity_per_axis):
        values = torch.zeros(2, 4, 2, 2)
        phase_valid = torch.ones_like(values, dtype=torch.bool)
        if entry_index < bank.neighbors:
            phase_valid[:, -1] = False
            descriptor = torch.zeros(7)
        else:
            values[:, -1] = relation
            descriptor = torch.full((7,), far_descriptor)
        support = torch.zeros(2, 4, 1, 2, 2)
        displacement_hist = torch.zeros(2, 4, 2, 2, 2)
        bank._domains[0][0].add(
            _RelationEntry(
                descriptor=descriptor,
                values=values,
                valid=torch.ones(2, 4, dtype=torch.bool),
                support=support,
                support_valid=torch.zeros_like(support, dtype=torch.bool),
                independent=torch.full_like(values, 0.25),
                phase_valid=phase_valid,
                support_raw=torch.zeros_like(support),
                displacement_hist=displacement_hist,
                displacement_valid=torch.zeros(2, 4, 2, 2, dtype=torch.bool),
            )
        )


def _uniform_phase_violation(phases: int) -> torch.Tensor:
    distances = 1
    values = torch.full((2, distances, phases, phases), 0.2)
    support = torch.zeros(2, distances, 1, phases, 2)
    curve = RelationCurve(
        values=values,
        valid=torch.ones(2, distances, dtype=torch.bool),
        support=support,
        support_valid=torch.zeros_like(support, dtype=torch.bool),
        descriptor=torch.zeros(phases + 5),
        roi_pixels=100,
        phase_valid=torch.ones_like(values, dtype=torch.bool),
    )
    target = RelationTarget(
        lower=torch.zeros(distances, phases, phases),
        upper=torch.zeros(distances, phases, phases),
        weights=torch.ones(distances),
        valid=torch.ones(distances, dtype=torch.bool),
        support_lower=torch.zeros(distances, phases, 2),
        support_upper=torch.zeros(distances, phases, 2),
        support_weights=torch.zeros(distances, phases),
        support_valid=torch.zeros(distances, phases, 2, dtype=torch.bool),
        ood_weight=1.0,
        source="shared",
        dilation_radius=torch.zeros(distances, phases, dtype=torch.long),
    )
    return relation_penalty(
        curve,
        target,
        phase_weight=1.0,
        support_weight=0.0,
        direction_reduction="mean",
    ).loss


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
    checker = torch.tensor(((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)))
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


def _phase_statistics(profile: torch.Tensor):
    distances, phases = profile.shape
    samples = 5
    excess = torch.zeros(samples, 2, distances, phases, phases)
    independent = torch.full_like(excess, 1.0 / phases**2)
    offsets = torch.linspace(-0.02, 0.02, samples)
    for sample, offset in enumerate(offsets):
        for direction in range(2):
            signed_offset = offset if direction == 0 else -offset
            for distance in range(distances):
                for phase in range(phases):
                    value = profile[distance, phase] + signed_offset
                    excess[sample, direction, distance, phase, phase] = value
                    excess[
                        sample,
                        direction,
                        distance,
                        phase,
                        (phase + 1) % phases,
                    ] = -value
    return _summarize_phase_target(
        excess,
        independent,
        torch.ones(samples, 2, distances, phases, phases, dtype=torch.bool),
        quantile_low=0.1,
        quantile_high=0.9,
        uncertainty_floor=0.01,
    )


def _shifted_support_statistics(shifts: tuple[int, ...], *, max_radius: int):
    center_labels = torch.zeros(16, 16, dtype=torch.long)
    center_labels[5:11, 4:10] = 1
    neighbor_labels = []
    for shift in shifts:
        labels = torch.zeros_like(center_labels)
        labels[5:11, 4 + shift : 10 + shift] = 1
        neighbor_labels.append(labels)
    center = F.one_hot(center_labels, num_classes=2).movedim(-1, 0).float()
    neighbors = (
        F.one_hot(
            torch.stack(neighbor_labels),
            num_classes=2,
        )
        .movedim(-1, 1)
        .float()
    )
    return _support_statistics(
        center,
        neighbors,
        torch.ones(16, 16, dtype=torch.bool),
        max_radius=max_radius,
        min_support_pixels=8,
        min_phase_fraction=0.001,
        min_chance_gap=0.02,
    )


def _support_target(support):
    samples = 3
    return _summarize_support_target(
        support.values.unsqueeze(0).unsqueeze(1).expand(samples, 2, -1, -1, -1, -1),
        support.valid.unsqueeze(0).unsqueeze(1).expand(samples, 2, -1, -1, -1, -1),
        support.displacement_hist.unsqueeze(0)
        .unsqueeze(1)
        .expand(samples, 2, -1, -1, -1, -1),
        support.displacement_valid.unsqueeze(0)
        .unsqueeze(1)
        .expand(samples, 2, -1, -1, -1),
        quantile_low=0.1,
        quantile_high=0.9,
        uncertainty_floor=1.0 / 16.0,
    )
