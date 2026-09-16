"""Bounded volume serialization and ownership of temporary downloads."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import anyio
import tifffile
import torch
from fastapi.responses import FileResponse, Response, StreamingResponse

CHUNK_BYTES = 64 * 1024


async def _send_until_disconnect(stream, receive):
    async with anyio.create_task_group() as group:

        async def send_body():
            try:
                await stream()
            finally:
                group.cancel_scope.cancel()

        group.start_soon(send_body)
        while (await receive())["type"] != "http.disconnect":
            pass
        group.cancel_scope.cancel()


class DownloadResponse(Response):
    """Hold a reserved result slot until its response has released its resources."""

    def __init__(self, response: Response, release):
        super().__init__(status_code=response.status_code)
        self.raw_headers = response.raw_headers
        self.response = response
        self.release = release

    async def __call__(self, scope, receive, send):
        try:
            await self.response(scope, receive, send)
        finally:
            self.release()


class RawVolumeResponse(StreamingResponse):
    def __init__(self, volume, headers):
        self.chunks = raw_chunks(volume)
        super().__init__(
            self.chunks,
            media_type="application/octet-stream",
            headers={**headers, "Content-Length": str(volume.numel())},
        )

    async def __call__(self, scope, receive, send):
        async def send_chunk(message):
            try:
                await send(message)
            finally:
                body = message.get("body")
                if isinstance(body, memoryview):
                    # An error traceback may retain this message after the slot
                    # is returned. Detach its view from the full volume now.
                    body.release()

        try:
            await _send_until_disconnect(
                lambda: self.stream_response(send_chunk), receive
            )
        finally:
            # The worker has finished before closing its generator. Release the
            # retained volume even if the response object survives cancellation.
            try:
                with anyio.CancelScope(shield=True):
                    await self.body_iterator.aclose()
            finally:
                self.chunks.close()


def raw_chunks(volume: torch.Tensor):
    """Keep each immutable view alive until sent, including strided inputs."""
    array = volume.numpy()
    if array.flags.c_contiguous:
        view = memoryview(array).cast("B")
        for start in range(0, len(view), CHUNK_BYTES):
            yield view[start : start + CHUNK_BYTES]
    else:
        # Normal generator output is contiguous. Handle strided volumes with
        # row-sized copies, never a second full-volume allocation.
        for plane in volume:
            for row in plane:
                for start in range(0, row.numel(), CHUNK_BYTES):
                    yield memoryview(
                        row[start : start + CHUNK_BYTES].contiguous().numpy()
                    ).cast("B")


class TemporaryFileResponse(FileResponse):
    """A POST download that deletes its file on completion or disconnection."""

    async def _handle_simple(self, send, send_header_only, send_pathsend):
        # Own the handle with a synchronous finally: async close can itself be
        # cancelled, leaving Windows unable to unlink the file. Avoid pathsend
        # because the server could still be reading the path after send returns.
        with open(self.path, "rb") as file:
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            if send_header_only:
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )
                return
            while True:
                chunk = await anyio.to_thread.run_sync(file.read, self.chunk_size)
                more = len(chunk) == self.chunk_size
                await send(
                    {"type": "http.response.body", "body": chunk, "more_body": more}
                )
                if not more:
                    break

    async def __call__(self, scope, receive, send):
        # This endpoint serves complete POST results; there is no persistent
        # file to resume. Keep all transfers on the owned-handle path above.
        scope = {
            **scope,
            "headers": [
                (key, value)
                for key, value in scope["headers"]
                if key.lower() not in (b"range", b"if-range")
            ],
        }
        try:
            await _send_until_disconnect(
                lambda: super(TemporaryFileResponse, self).__call__(
                    scope, receive, send
                ),
                receive,
            )
        finally:
            Path(self.path).unlink(missing_ok=True)


def volume_response(volume: torch.Tensor, format: str, headers: dict):
    if format == "raw":
        return RawVolumeResponse(volume, headers)
    path = None
    try:
        with NamedTemporaryFile(suffix=".tiff", delete=False) as file:
            path = Path(file.name)
            tifffile.imwrite(file, volume.numpy())
        return TemporaryFileResponse(
            path,
            media_type="image/tiff",
            headers={**headers, "Accept-Ranges": "none"},
            filename="volume.tiff",
            stat_result=path.stat(),
        )
    except BaseException:
        # Transfer ownership to the response only after the complete file and
        # headers exist; failures before that point must also remove the file.
        if path is not None:
            path.unlink(missing_ok=True)
        raise
