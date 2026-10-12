import json
from dataclasses import asdict, fields

import pytest
import torch

from src.train.batch import (
    ConditionPresence,
    DenoiserUpdate,
    RealBatch,
    StepPreparation,
)
from src.train.metrics import collect_metrics


@pytest.mark.parametrize("boundary_pairs", [None, 0, 2])
def test_step_metrics_serialize_unavailable_boundaries_without_changing_diagnostics(
    boundary_pairs,
):
    diagnostics = {
        "anchor/neighbor_agreement": torch.tensor(0.75),
        "anchor/neighbor_excess_jump": torch.tensor(0.25),
        "skipped/generator": 0,
    }
    if boundary_pairs is not None:
        diagnostics["anchor/boundary_pairs"] = torch.tensor(boundary_pairs)
    prepared = StepPreparation(
        transition=1,
        domain=2,
        critic_domains={},
        real=RealBatch(images={}),
        selection=None,
        target_vf=torch.tensor([[0.5, 0.5]]),
        presence=ConditionPresence(
            anchor=torch.tensor([False, False]),
            vf=torch.tensor([True, False]),
        ),
        model_conditions={},
        anchor_ramp=0.0,
    )
    update = DenoiserUpdate(
        **{field.name: torch.tensor(0.125) for field in fields(DenoiserUpdate)}
    )

    metrics = collect_metrics(
        volume_size=8,
        diagnostics=diagnostics,
        prepared=prepared,
        denoiser_update=update,
        critic_vals=[torch.tensor(0.5), 0.0, torch.tensor(0.25)],
        r1=torch.tensor(0.0),
        critic_global=torch.tensor(0.75),
        critic_local=0.0,
        critic_connectivity=0.0,
        connectivity_r1=0.0,
    )
    record = json.loads(json.dumps(asdict(metrics), allow_nan=False))
    assert record["critic"] == 0.75
    assert record["critic_axes"] == [0.5, 0.0, 0.25]
    assert record["vf_active"] is True
    assert record["vf_active_fraction"] == 0.5
    assert record["anchor_shared"] is False
    assert record["anchor_planes"] == 0
    assert record["domain"] == 2
    for key, value in (("agreement", 0.75), ("excess_jump", 0.25)):
        expected = value if boundary_pairs else None
        assert record[f"anchor_neighbor_{key}"] == expected
        assert record["diagnostics"][f"anchor/neighbor_{key}"] == expected
        assert isinstance(diagnostics[f"anchor/neighbor_{key}"], torch.Tensor)
        assert diagnostics[f"anchor/neighbor_{key}"].item() == value
