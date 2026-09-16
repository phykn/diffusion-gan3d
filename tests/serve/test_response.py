import asyncio

import anyio
import pytest
import torch

from src.serve.response import CHUNK_BYTES, TemporaryFileResponse, raw_chunks


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
    "failure", ["disconnect", "cancel", "send_error", "scope_cancel"]
)
def test_temporary_file_is_removed_when_transfer_stops(tmp_path, failure):
    path = tmp_path / "volume.tiff"
    path.write_bytes(b"x" * (CHUNK_BYTES * 3))
    response = TemporaryFileResponse(path, stat_result=path.stat())
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [],
        "asgi": {"spec_version": "2.4"},
    }

    async def exercise():
        body_started = anyio.Event()

        async def receive():
            await body_started.wait()
            if failure == "disconnect":
                return {"type": "http.disconnect"}
            await anyio.sleep_forever()

        async def send(message):
            if message["type"] == "http.response.body":
                body_started.set()
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

    anyio.run(exercise)
    assert not path.exists()
