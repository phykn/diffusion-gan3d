"""Aligned, colored terminal output for the training and check commands.

Color is used only on a terminal; NO_COLOR turns it off and FORCE_COLOR on.
"""

import os
import sys

from tqdm import tqdm

LABEL_WIDTH = 16
STYLES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
}


def color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty()) and os.environ.get("TERM") != "dumb"


def paint(text, *styles: str) -> str:
    text = str(text)
    if not styles or not color_enabled():
        return text
    return f"\033[{';'.join(STYLES[name] for name in styles)}m{text}\033[0m"


def emit(text: str = "") -> None:
    # tqdm.write keeps an open progress bar intact; with none it is a print.
    tqdm.write(text)
    sys.stdout.flush()


def section(title: str) -> None:
    """Start a group of fields with a blank line and a bold heading."""
    emit()
    emit(paint(title, "bold", "cyan"))


def field(label: str, value, *styles: str) -> None:
    """One aligned "label  value" line; styles color the value."""
    emit(f"  {paint(label.ljust(LABEL_WIDTH), 'dim')}{paint(value, *styles)}")


def path_field(label: str, path) -> None:
    field(label, path, "blue")


def metric(label: str, value) -> None:
    field(label, value, "bold")


def status(text: str) -> None:
    """Announce slow work that is starting."""
    emit(paint(f"▸ {text}…", "yellow"))


def success(text: str) -> None:
    emit(paint(f"✓ {text}", "bold", "green"))


def warning(text: str) -> None:
    emit(paint(f"! {text}", "bold", "yellow"))
