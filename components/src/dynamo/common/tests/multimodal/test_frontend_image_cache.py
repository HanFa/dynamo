# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend-fetched payloads reuse the worker's existing URL-cache contract."""

import asyncio
import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from dynamo.common.http import HttpStatusError
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy
from dynamo.common.multimodal.image_loader import ImageLoader

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

URL = "https://example.com/image.png"
FETCH = "dynamo.common.multimodal.image_loader.fetch_bytes"


@pytest.fixture
def image_bytes():
    with BytesIO() as buffer:
        Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
        return buffer.getvalue()


@pytest.fixture
def data_url(image_bytes):
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode()


@pytest.fixture
def loader():
    return ImageLoader(
        cache_size=2,
        url_policy=UrlValidationPolicy(allow_http=True, allow_private_ips=True),
    )


async def test_frontend_image_reuses_existing_url_cache(loader, data_url, image_bytes):
    with patch(FETCH, AsyncMock(return_value=image_bytes)) as fetch:
        cached = await loader.load_image(URL)
        frontend = AsyncMock(return_value=data_url)
        result = await loader.load_image_batch(
            [{"Url": "https://EXAMPLE.COM/image.png#fragment"}],
            image_fetches=[frontend],
        )
    assert result[0] is cached
    frontend.assert_not_awaited()
    fetch.assert_awaited_once()


async def test_frontend_miss_populates_url_cache_without_downloading(loader, data_url):
    frontend = AsyncMock(return_value=data_url)
    with patch(FETCH, AsyncMock()) as fetch:
        first = await loader.load_image(URL, fetch_image=frontend)
        second = await loader.load_image(URL, fetch_image=frontend)
        legacy = await loader.load_image(URL)
    assert first is second is legacy
    assert first.getpixel((0, 0)) == (255, 0, 0)
    frontend.assert_awaited_once()
    fetch.assert_not_awaited()


async def test_concurrent_frontend_images_resolve_once(loader, data_url):
    frontend = AsyncMock(return_value=data_url)
    images = await asyncio.gather(
        *(loader.load_image(URL, fetch_image=frontend) for _ in range(8))
    )
    assert all(image is images[0] for image in images)
    frontend.assert_awaited_once()


async def test_warm_cache_does_not_wait_for_pending_or_failed_frontend(
    loader, data_url
):
    cached = await loader.load_image(URL, fetch_image=AsyncMock(return_value=data_url))
    gate = asyncio.Event()
    pending = AsyncMock(side_effect=gate.wait)
    failed = AsyncMock(side_effect=HttpStatusError(503, "unavailable", "frontend"))
    for frontend in (pending, failed):
        assert (
            await asyncio.wait_for(loader.load_image(URL, fetch_image=frontend), 1)
            is cached
        )
        frontend.assert_not_awaited()


async def test_failed_frontend_fetch_can_be_retried(loader, data_url):
    frontend = AsyncMock(side_effect=[ValueError("fetch failed"), data_url])
    with patch(FETCH, AsyncMock()) as fetch:
        with pytest.raises(ValueError, match="fetch failed"):
            await loader.load_image(URL, fetch_image=frontend)
        assert not loader._inflight
        assert loader.cache_entries == 0
        image = await loader.load_image(URL, fetch_image=frontend)
    assert image.size == (2, 2)
    fetch.assert_not_awaited()


async def test_frontend_images_follow_lru_eviction(loader, data_url):
    frontend = AsyncMock(return_value=data_url)
    first = await loader.load_image(URL, fetch_image=frontend)
    await loader.load_image(URL + "?b", fetch_image=frontend)
    assert await loader.load_image(URL, fetch_image=frontend) is first
    await loader.load_image(URL + "?c", fetch_image=frontend)
    await loader.load_image(URL + "?b", fetch_image=frontend)
    assert frontend.await_count == 4
    assert loader.cache_entries == 2


@pytest.mark.parametrize("cache_size,session_scoped", [(0, False), (2, True)])
async def test_frontend_images_honor_cache_bypass(data_url, cache_size, session_scoped):
    loader = ImageLoader(
        cache_size=cache_size,
        session_scoped_cache=session_scoped,
        url_policy=UrlValidationPolicy(allow_http=True, allow_private_ips=True),
    )
    frontend = AsyncMock(return_value=data_url)
    first = await loader.load_image(URL, fetch_image=frontend)
    second = await loader.load_image(URL, fetch_image=frontend)
    assert first is not second
    assert frontend.await_count == 2
    assert loader.cache_entries == 0


async def test_frontend_images_preserve_session_isolation(data_url):
    loader = ImageLoader(
        session_scoped_cache=True,
        url_policy=UrlValidationPolicy(allow_http=True, allow_private_ips=True),
    )
    frontend = AsyncMock(return_value=data_url)
    first = await loader.load_image(URL, fetch_image=frontend, cache_scope="a")
    second = await loader.load_image(URL, fetch_image=frontend, cache_scope="b")
    assert first is not second
    assert await loader.load_image(URL, cache_scope="a") is first
    assert await loader.load_image(URL, cache_scope="b") is second


async def test_original_url_policy_is_checked_before_cache_hit(data_url):
    loader = ImageLoader()
    url = "http://127.0.0.1/image.png"
    loader._cache_put(url, Image.new("RGB", (2, 2)))
    frontend = AsyncMock(return_value=data_url)
    with pytest.raises(UrlValidationError):
        await loader.load_image(url, fetch_image=frontend)
    frontend.assert_not_awaited()


async def test_frontend_fetches_preserve_uuid_and_inline_slots(loader, data_url):
    frontend = AsyncMock(return_value=data_url)
    images = await loader.load_image_batch(
        [{"Url": URL}, {"UuidOnly": "cached"}, {"Url": data_url}],
        image_fetches=[frontend, None, None],
        preserve_uuid_slots=True,
    )
    assert images[0].size == images[2].size == (2, 2)
    assert images[1] is None
    assert loader.cache_entries == 1
    frontend.assert_awaited_once()


@pytest.mark.parametrize("fetches", [[], [None, None], [42], [""]])
async def test_misaligned_frontend_fetches_are_rejected(loader, fetches):
    with pytest.raises(ValueError, match="align"):
        await loader.load_image_batch([{"Url": URL}], image_fetches=fetches)


async def test_frontend_fetches_require_http_image_slots(loader, data_url):
    frontend = AsyncMock(return_value=data_url)
    with pytest.raises(ValueError, match="URL"):
        await loader.load_image_batch(
            [{"UuidOnly": "cached"}], image_fetches=[frontend]
        )
    with pytest.raises(ValueError, match="HTTP"):
        await loader.load_image(data_url, fetch_image=frontend)
    frontend.assert_not_awaited()
