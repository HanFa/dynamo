# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve request-scoped frontend images only after the worker's URL-cache miss."""

import asyncio
from collections import OrderedDict
from functools import partial
from typing import Any
from uuid import UUID

from dynamo.common.http import HttpStatusError

from .image_loader import ImageFetch


class FrontendImageClient:
    def __init__(self, runtime: Any, *, timeout: float = 30) -> None:
        self._runtime = runtime
        self._timeout = timeout
        self._clients: OrderedDict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()

    def callbacks(
        self, references: list[dict[str, str] | None] | None, context: Any = None
    ) -> list[ImageFetch | None] | None:
        if references is None:
            return None
        if not isinstance(references, list):
            raise ValueError("Frontend image references must be a list")
        callbacks: list[ImageFetch | None] = []
        for reference in references:
            if reference is None:
                callbacks.append(None)
                continue
            if (
                not isinstance(reference, dict)
                or not isinstance(reference.get("endpoint"), str)
                or not isinstance(reference.get("token"), str)
            ):
                raise ValueError("Invalid frontend image reference")
            # These references are produced by the Rust frontend, never copied
            # from public API input. Limit the callback to its image service.
            parts = reference["endpoint"].split(".")
            if len(parts) != 3 or not parts[0] or parts[2] != "fetch":
                raise ValueError("Invalid frontend image endpoint")
            if not parts[1].startswith("images-"):
                raise ValueError("Invalid frontend image endpoint")
            UUID(parts[1][len("images-") :])
            UUID(reference["token"])
            callbacks.append(partial(self.fetch, reference, context))
        return callbacks

    async def fetch(self, reference: dict[str, str], context: Any = None) -> str:
        try:
            return await asyncio.wait_for(
                self._fetch(reference, context), self._timeout
            )
        except asyncio.TimeoutError as error:
            raise HttpStatusError(
                408, "Frontend image fetching timed out", "frontend"
            ) from error

    async def _fetch(self, reference: dict[str, str], context: Any) -> str:
        if self._runtime is None:
            raise ValueError("Frontend image fetching requires a Dynamo runtime")
        endpoint = reference["endpoint"]
        async with self._lock:
            client = self._clients.get(endpoint)
            if client is None:
                client = await self._runtime.endpoint(endpoint).client()
                self._clients[endpoint] = client
                if len(self._clients) > 16:
                    self._clients.popitem(last=False)
            self._clients.move_to_end(endpoint)
        await client.wait_for_instances()
        stream = await client.round_robin(
            {"token": reference["token"]}, annotated=False, context=context
        )
        response = None
        async for item in stream:
            if response is not None:
                raise ValueError("Unexpected frontend image response count")
            response = item
        if not isinstance(response, dict):
            raise ValueError("Missing frontend image response")
        error = response.get("error")
        if error is not None:
            status = 408 if error.get("class") == "DeadlineExceeded" else 400
            if error.get("class") == "CapacityExhausted":
                status = 429
            raise HttpStatusError(status, "Frontend could not fetch image", "frontend")
        data_url = response.get("data_url")
        if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
            raise ValueError("Invalid frontend image response")
        return data_url
