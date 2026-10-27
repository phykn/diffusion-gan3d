from fastapi.responses import JSONResponse


class RequestSizeLimit:
    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                response = JSONResponse(
                    status_code=413, content={"detail": "request body is too large"}
                )
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        # This frame stays alive while the app runs; keep one copy of the body.
        del chunks, chunk, message

        async def replay_receive():
            nonlocal body
            if body is not None:
                result, body = body, None
                return {"type": "http.request", "body": result, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
