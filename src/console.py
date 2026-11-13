import os
import sys

from tqdm import tqdm

STYLES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
}


def color_enabled():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty()) and os.environ.get("TERM") != "dumb"


def paint(text, *styles):
    text = str(text)
    if not styles or not color_enabled():
        return text
    return f"\033[{';'.join(STYLES[name] for name in styles)}m{text}\033[0m"


def emit(text=""):
    # tqdm.write keeps an open progress bar intact; with none it is a print.
    tqdm.write(text)
    sys.stdout.flush()


def section(title):
    emit()
    emit(paint(title, "bold", "cyan"))


def field(label, value, *styles):
    emit(f"  {paint(f'{label:<15} ', 'dim')}{paint(value, *styles)}")


def path_field(label, path):
    field(label, path, "blue")


def metric(label, value):
    field(label, value, "bold")


def status(text):
    emit(paint(f"▸ {text}…", "yellow"))


def success(text):
    emit(paint(f"✓ {text}", "bold", "green"))


def warning(text):
    emit(paint(f"! {text}", "bold", "yellow"))
