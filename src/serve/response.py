"""Bounded volume serialization and ownership of temporary downloads."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import anyio
import tifffile
import torch
from fastapi.responses import FileResponse, StreamingResponse

CHUNK_BYTES = 64 * 1024


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
            async with anyio.create_task_group() as group:

                async def stream():
                    try:
                        await super(TemporaryFileResponse, self).__call__(
                            scope, receive, send
                        )
                    finally:
                        group.cancel_scope.cancel()

                group.start_soon(stream)
                while (await receive())["type"] != "http.disconnect":
                    pass
                group.cancel_scope.cancel()
        finally:
            Path(self.path).unlink(missing_ok=True)


def volume_response(volume: torch.Tensor, format: str, headers: dict):
    if format == "raw":
        return StreamingResponse(
            raw_chunks(volume),
            media_type="application/octet-stream",
            headers={**headers, "Content-Length": str(volume.numel())},
        )
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
