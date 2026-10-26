import pytest
import torch

from src.model.diffusion import Diffusion
from src.predict.anchor import AnchorGuidance, PlaneAnchor, encode_anchors
from src.predict.generator import Generator
from src.predict.tiling.generator import TiledGenerator


def test_plane_anchor_builds_dense_condition_for_every_axis() -> None:
    size = 4
    image = torch.arange(size * size).reshape(size, size).remainder(3).long()

    for axis in (0, 1, 2):
        target = encode_anchors(
            (PlaneAnchor(image=image, axis=axis, index=2),),
            batch_size=1,
            num_phases=3,
            volume_size=size,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert target is not None
        assert target.image.shape == (1, 3, size, size, size)
        assert target.mask.shape == (1, 1, size, size, size)
        assert int(target.mask.sum()) == size * size
        assert target.mask[0, 0].select(axis, 2).all()
        generated = target.image.argmax(dim=1).select(axis + 1, 2)
        assert torch.equal(generated[0], image)


def test_cross_axis_anchor_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting intersections"):
        encode_anchors(
            (
                PlaneAnchor(
                    image=torch.zeros(4, 4, dtype=torch.long),
                    axis=0,
                    index=1,
                ),
                PlaneAnchor(
                    image=torch.ones(4, 4, dtype=torch.long),
                    axis=1,
                    index=1,
                ),
            ),
            batch_size=1,
            num_phases=3,
            volume_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_partial_anchor_is_centered_without_resizing() -> None:
    image = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    target = encode_anchors(
        (PlaneAnchor(image=image, axis=0, index=1),),
        batch_size=1,
        num_phases=3,
        volume_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert target is not None
    assert int(target.mask.sum()) == 4
    assert torch.equal(target.image.argmax(1)[0, 1, 1:3, 1:3], image)
    assert torch.equal(
        target.mask[0, 0, 1, 1:3, 1:3],
        torch.ones(2, 2, dtype=torch.bool),
    )


def test_partial_anchor_accepts_an_explicit_position() -> None:
    image = torch.ones(2, 3, dtype=torch.long)
    target = encode_anchors(
        (PlaneAnchor(image=image, axis=2, index=3, position=(2, 1)),),
        batch_size=1,
        num_phases=3,
        volume_size=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert target is not None
    assert torch.equal(target.image.argmax(1)[0, 2:4, 1:4, 3], image)
    assert int(target.mask.sum()) == 6


def test_partial_anchor_preserves_batch_and_axis_coordinates() -> None:
    images = torch.arange(12).reshape(2, 2, 3).remainder(3).long()
    target = encode_anchors(
        (PlaneAnchor(image=images, axis=1, index=4, position=(1, 0)),),
        batch_size=2,
        num_phases=3,
        volume_size=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert target is not None
    assert torch.equal(target.image.argmax(1)[:, 1:3, 4, 0:3], images)
    assert int(target.mask.sum()) == 12


def test_anchor_image_must_fit_generation_size_and_phase_count() -> None:
    with pytest.raises(ValueError, match="fit inside"):
        encode_anchors(
            (
                PlaneAnchor(
                    image=torch.zeros(5, 3, dtype=torch.long),
                    axis=0,
                    index=1,
                ),
            ),
            batch_size=1,
            num_phases=3,
            volume_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="phase labels"):
        encode_anchors(
            (
                PlaneAnchor(
                    image=torch.full((4, 4), 3, dtype=torch.long),
                    axis=0,
                    index=1,
                ),
            ),
            batch_size=1,
            num_phases=3,
            volume_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_anchor_image_rejects_fractional_phase_labels() -> None:
    with pytest.raises(ValueError, match="phase fractions must have shape"):
        encode_anchors(
            (
                PlaneAnchor(
                    image=torch.tensor([[0.2, 1.8]], dtype=torch.float32),
                    axis=0,
                    index=1,
                ),
            ),
            batch_size=1,
            num_phases=2,
            volume_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "axis,index,position",
    [
        (True, 0, None),
        (0, 0.5, None),
        (0, True, None),
        (0, 0, (False, 0)),
        (0, 0, ()),
    ],
)
def test_anchor_coordinates_require_explicit_integers(axis, index, position):
    anchor = PlaneAnchor(torch.zeros(2, 2, dtype=torch.long), axis, index, position)
    with pytest.raises(ValueError):
        encode_anchors([anchor], 1, 2, 4, torch.device("cpu"), torch.float32)


def _double(current, time, latent, **conditions):
    return 2.0 * current


def _plane_guidance(planes, strength=1.0):
    image = torch.zeros(1, 2, 4, 4, 4)
    image[:, 0] = 1.0
    image[:, 1] = -1.0
    mask = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
    mask[:, :, list(planes)] = True
    return AnchorGuidance(image, mask, strength)


def test_step_is_the_error_gradient_over_the_rms_anchor_error():
    guide = _plane_guidance((1,), strength=0.5)
    current = torch.linspace(-1, 1, 128).reshape(1, 2, 4, 4, 4)

    pred, step = guide(_double, current, torch.zeros(1), torch.zeros(1, 1))

    residual = (2.0 * current - guide.image) * guide.mask
    rms = (residual.square().sum() / guide.mask.sum()).sqrt()
    torch.testing.assert_close(pred, 2.0 * current)
    torch.testing.assert_close(step, 0.5 * 4.0 * residual / rms)
    assert not pred.requires_grad and not step.requires_grad


def test_step_size_does_not_grow_or_shrink_with_anchor_area():
    current = torch.zeros(1, 2, 4, 4, 4)
    args = (torch.zeros(1), torch.zeros(1, 1))

    _, one = _plane_guidance((1,))(_double, current, *args)
    _, two = _plane_guidance((1, 2))(_double, current, *args)

    torch.testing.assert_close(one[:, :, 1], two[:, :, 1])


def test_guidance_differentiates_inference_tensors():
    guide = _plane_guidance((1,))
    with torch.inference_mode():
        current = torch.randn(1, 2, 4, 4, 4)
        height = torch.ones(1, 1, 4, 4, 4)

        def model(state, time, latent, height):
            return torch.tanh(state * height)

        pred, step = guide(
            model, current, torch.zeros(1), torch.zeros(1, 1), height=height
        )

    assert step.abs().sum() > 0
    torch.testing.assert_close(pred, torch.tanh(current))


def test_prediction_that_ignores_the_state_gets_no_step():
    def constant(current, time, latent):
        return torch.ones_like(current)

    _, step = _plane_guidance((1,))(
        constant, torch.zeros(1, 2, 4, 4, 4), torch.zeros(1), torch.zeros(1, 1)
    )

    assert torch.equal(step, torch.zeros_like(step))


def test_diffusion_subtracts_the_step_before_every_later_transition():
    diffusion = Diffusion(2)
    states = []

    def model(current, time, latent):
        return torch.tanh(current)

    def guide(model, current, time, latent):
        states.append(current.clone())
        return model(current, time, latent), torch.full_like(current, 0.25)

    noise = torch.zeros(1, 2, 2, 2, 2)
    torch.manual_seed(5)
    result = diffusion.sample(model, noise, 3, guide=guide)
    torch.manual_seed(5)
    torch.randn(1, 3)
    plain = diffusion.sample_posterior(noise, torch.tanh(noise), 1)

    torch.testing.assert_close(states[1], plain - 0.25)
    torch.testing.assert_close(result, torch.tanh(states[1]))


class _SteerableModel(torch.nn.Module):
    num_domains = 1
    downsample_factor = 1

    def forward(self, current, time, latent, domain, vf=None):
        return torch.tanh(current)


def _anchored_match(sample, strength):
    anchor = PlaneAnchor(torch.ones(8, 8, dtype=torch.long), 0, 3)
    torch.manual_seed(11)
    volume = sample(anchors=(anchor,), anchor_strength=strength)
    return (volume[3] == 1).float().mean()


def test_guidance_pulls_the_anchor_plane_in_direct_and_tiled_sampling():
    generator = Generator(
        _SteerableModel(), Diffusion(4), torch.device("cpu"), 8, 2, 4, False
    )
    tiled = TiledGenerator(generator)

    def scaled(**options):
        return tiled.generate(shape=(8, 8, 12), overlap=2, progress=False, **options)

    for sample in (generator.generate, scaled):
        assert _anchored_match(sample, 2.0) > _anchored_match(sample, 0.0) + 0.2


def test_tiled_guidance_runs_inside_inference_mode():
    # HR extension samples its LR context under inference_mode.
    generator = Generator(
        _SteerableModel(), Diffusion(4), torch.device("cpu"), 8, 2, 4, False
    )
    tiled = TiledGenerator(generator)

    def scaled(**options):
        with torch.inference_mode():
            return tiled.generate(
                shape=(8, 8, 12), overlap=2, progress=False, **options
            )

    assert _anchored_match(scaled, 2.0) > _anchored_match(scaled, 0.0) + 0.2
