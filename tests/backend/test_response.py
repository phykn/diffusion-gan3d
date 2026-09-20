import asyncio
from threading import BoundedSemaphore

import anyio
import pytest
import torch

from backend.src.response import (
    CHUNK_BYTES,
    DownloadResponse,
    RawVolumeResponse,
    TemporaryFileResponse,
    raw_chunks,
)


@pytest.mark.parametrize("strided", [False, True])
def test_raw_transfer_uses_bounded_views_in_c_order(strided):
    volume = (
        torch.arange(4 * 400 * 200, dtype=torch.int32)
        .to(torch.uint8)
        .reshape(4, 400, 200)
    )
    if strided:
        volume = volume.transpose(1, 2)
    chunks = list(raw_chunks(volume))
    assert len(chunks) > 1
    assert all(
        isinstance(chunk, memoryview) and len(chunk) <= CHUNK_BYTES for chunk in chunks
    )
    assert b"".join(chunks) == volume.numpy().tobytes(order="C")
    if not strided:
        # A view shares the label storage rather than a serialized full copy.
        volume[0, 0, 0] = 17
        assert chunks[0][0] == 17


@pytest.mark.parametrize(
    "failure", ["complete", "disconnect", "cancel", "send_error", "scope_cancel"]
)
@pytest.mark.parametrize("format", ["raw", "tiff"])
def test_resources_and_download_slot_are_released_when_transfer_stops(
    tmp_path, failure, format
):
    path = tmp_path / "volume.tiff"
    if format == "tiff":
        path.write_bytes(b"x" * (CHUNK_BYTES * 3))
        inner = TemporaryFileResponse(path, stat_result=path.stat())
    else:
        inner = RawVolumeResponse(torch.zeros(3, 256, 256, dtype=torch.uint8), {})
    slots = BoundedSemaphore(1)
    assert slots.acquire(blocking=False)
    response = DownloadResponse(inner, slots.release)
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [],
        "asgi": {"spec_version": "2.4"},
    }

    async def exercise():
        body_started = anyio.Event()
        retained_views = []

        async def receive():
            await body_started.wait()
            if failure == "disconnect":
                return {"type": "http.disconnect"}
            await anyio.sleep_forever()

        async def send(message):
            if message["type"] == "http.response.body":
                assert not slots.acquire(blocking=False)
                if isinstance(message.get("body"), memoryview):
                    retained_views.append(message["body"])
                body_started.set()
                if failure == "complete":
                    return
                if failure == "cancel":
                    raise asyncio.CancelledError()
                if failure == "send_error":
                    raise OSError("client socket closed")
                if failure == "scope_cancel":
                    cancel_scope.cancel()
                await anyio.sleep_forever()

        with anyio.CancelScope() as cancel_scope:
            try:
                with anyio.fail_after(5):
                    await response(scope, receive, send)
            except BaseExceptionGroup as exc:
                assert failure == "send_error"
                assert isinstance(exc.exceptions[0], OSError)
        assert not path.exists()
        if format == "raw":
            assert inner.chunks.gi_frame is None
            assert inner.body_iterator.ag_frame is None
            for view in retained_views:
                with pytest.raises(ValueError, match="released memoryview"):
                    view.tobytes()
        assert slots.acquire(blocking=False)
        assert not slots.acquire(blocking=False)

    anyio.run(exercise)
    assert not path.exists()
