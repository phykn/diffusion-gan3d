from src.console import LABEL_WIDTH, field


def test_field_keeps_a_space_after_a_full_width_label(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    field("x" * LABEL_WIDTH, "value")
    field("short", "value")
    long, short = capsys.readouterr().out.splitlines()
    assert long == f"  {'x' * LABEL_WIDTH} value"
    assert short.index("value") == long.index("value") - 1
