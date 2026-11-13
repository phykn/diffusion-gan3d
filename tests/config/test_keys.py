"""Each schema field is read somewhere."""

import re
from dataclasses import fields, is_dataclass

from src.config.io import PROJECT_ROOT
from src.config.schema import LRConfig, SRConfig

ROOTS = ("train.py", "src", "scripts", "paper/scripts", "app/backend")


def _leaf_paths(cls, prefix=()):
    for item in fields(cls):
        path = (*prefix, item.name)
        kind = item.type
        if is_dataclass(kind):
            yield from _leaf_paths(kind, path)
        else:
            yield ".".join(path)


def _sources():
    texts = []
    for root in ROOTS:
        folder = PROJECT_ROOT / root
        files = [folder] if folder.is_file() else folder.rglob("*.py")
        texts.extend(
            path.read_text(encoding="utf-8")
            for path in files
            if path.name != "schema.py"
        )
    return "\n".join(texts)


def _unread():
    text = _sources()
    paths = set(_leaf_paths(LRConfig)) | set(_leaf_paths(SRConfig))
    unread = set()
    for path in paths:
        tail = path.split(".")[-2:]
        pattern = r"\b" + r"\.".join(map(re.escape, tail)) + r"\b"
        if len(tail) == 1:
            pattern = r"\." + pattern
        if not re.search(pattern, text):
            unread.add(path)
    return unread


def test_every_key_is_read():
    assert not _unread(), sorted(_unread())
