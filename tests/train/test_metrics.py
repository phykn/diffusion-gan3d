from dataclasses import asdict, fields

import torch
from torch.utils.tensorboard import SummaryWriter

from scripts.common.metrics import read_metrics
from src.train.batch import DenoiserUpdate, RealBatch, StepPreparation
from src.train.metrics import Metrics, collect_metrics, write_metrics


def test_tensorboard_records_collected_step_metrics(tmp_path):
    prepared = StepPreparation(
        transition=1,
        domain=2,
        critic_domains={},
        real=RealBatch(images={}),
        target_vf=torch.tensor([[0.5, 0.5]]),
        vf_present=torch.tensor([True, False]),
        model_conditions={},
    )
    update = DenoiserUpdate(
        **{field.name: torch.tensor(0.125) for field in fields(DenoiserUpdate)}
    )

    metrics = collect_metrics(
        diagnostics={"skipped/generator": 0},
        prepared=prepared,
        denoiser_update=update,
        critic_vals=[torch.tensor(0.5), 0.0, torch.tensor(0.25)],
        r1=torch.tensor(0.0),
    )
    with SummaryWriter(tmp_path) as writer:
        write_metrics(writer, 7, metrics)
    (record,) = read_metrics(tmp_path)
    assert record["step"] == 7
    assert record["loss/critic"] == 0.75
    assert record["loss/vf"] == 0.125
    assert [record[f"critic_plane/{axis}"] for axis in ("xy", "xz", "yz")] == [
        0.5,
        0.0,
        0.25,
    ]
    assert record["conditioning/vf_active"] == 1
    assert record["conditioning/vf_fraction"] == 0.5
    assert record["sampling/domain"] == 2


def test_tensorboard_records_every_metric_field_and_diagnostic(tmp_path):
    values = {
        item.name: index + 0.125
        for index, item in enumerate(fields(Metrics))
        if item.name not in ("critic_axes", "diagnostics")
    }
    values.update(
        transition=3,
        critic_axes=(101.125, 102.125, 103.125),
        diagnostics={"custom/value": 201.125, "gradient/generator": None},
    )
    metrics = Metrics(**values)
    with SummaryWriter(tmp_path) as writer:
        write_metrics(writer, 19, metrics)
    (record,) = read_metrics(tmp_path)
    assert record["step"] == 19
    assert record["gradient/generator"] is None
    # Three plane scalars and two timestep views replace two composite fields.
    assert len(record) - 1 == len(fields(Metrics)) + 3 + len(metrics.diagnostics)
    for name, value in asdict(metrics).items():
        if name not in ("critic_axes", "diagnostics"):
            assert value in record.values(), name
    for value in metrics.critic_axes:
        assert value in record.values()
    assert record["custom/value"] == 201.125


def test_tensorboard_reader_keeps_all_steps_across_event_files(tmp_path):
    for start, stop in ((0, 6000), (6000, 12005)):
        with SummaryWriter(tmp_path) as writer:
            for step in range(start, stop):
                writer.add_scalar("loss/generator_total", step / 8, step)
                if step % 2000 == 0:
                    writer.add_scalar("skipped/generator", 1, step)
    records = read_metrics(tmp_path)
    assert [row["step"] for row in records] == list(range(12005))
    assert [row["loss/generator_total"] for row in records] == [
        step / 8 for step in range(12005)
    ]
    assert sum(row.get("skipped/generator", 0) for row in records) == 7
