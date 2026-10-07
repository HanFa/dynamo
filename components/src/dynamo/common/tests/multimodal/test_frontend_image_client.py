# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from dynamo.common.http import HttpStatusError
from dynamo.common.multimodal.frontend_image_client import FrontendImageClient

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def reference():
    return {
        "endpoint": f"test-namespace.images-{uuid4().hex}.fetch",
        "token": str(uuid4()),
    }


def client_with_response(response):
    async def responses(*args, **kwargs):
        async def stream():
            yield response

        return stream()

    transport = Mock(
        wait_for_instances=AsyncMock(),
        round_robin=AsyncMock(side_effect=responses),
    )
    runtime = Mock()
    runtime.endpoint.return_value.client = AsyncMock(return_value=transport)
    return FrontendImageClient(runtime), runtime, transport


async def test_frontend_image_reference_is_lazy_and_preserves_context(reference):
    client, runtime, transport = client_with_response(
        {"data_url": "data:image/png;base64,YQ==", "error": None}
    )
    context = object()
    callbacks = client.callbacks([None, reference], context)
    assert callbacks[0] is None
    runtime.endpoint.assert_not_called()
    for _ in range(2):
        assert await callbacks[1]() == "data:image/png;base64,YQ=="
    runtime.endpoint.assert_called_once_with(reference["endpoint"])
    transport.round_robin.assert_awaited_with(
        {"token": reference["token"]}, annotated=False, context=context
    )


@pytest.mark.parametrize(
    "error_class,status",
    [("DeadlineExceeded", 408), ("CapacityExhausted", 429), ("InvalidArgument", 400)],
)
async def test_frontend_errors_keep_http_status(reference, error_class, status):
    client, _, _ = client_with_response(
        {
            "data_url": None,
            "error": {"class": error_class, "message": "private diagnostics"},
        }
    )
    with pytest.raises(HttpStatusError) as error:
        await client.callbacks([reference])[0]()
    assert error.value.status == status
    assert "private" not in str(error.value)


async def test_frontend_discovery_timeout_is_bounded(reference):
    client, _, transport = client_with_response({})
    client._timeout = 0.01
    transport.wait_for_instances.side_effect = asyncio.Event().wait
    with pytest.raises(HttpStatusError) as error:
        await client.callbacks([reference])[0]()
    assert error.value.status == 408
    transport.round_robin.assert_not_awaited()


@pytest.mark.parametrize(
    "references",
    ["url", [42], [{}], [{"endpoint": "dynamo.workers.generate", "token": "bad"}]],
)
async def test_malformed_frontend_references_are_rejected(references):
    client = FrontendImageClient(None)
    with pytest.raises(ValueError):
        client.callbacks(references)


@pytest.mark.parametrize(
    "response", [None, {}, {"data_url": "http://example.com/image"}]
)
async def test_malformed_frontend_response_never_fetches_origin(reference, response):
    client, _, _ = client_with_response(response)
    with pytest.raises(ValueError):
        await client.callbacks([reference])[0]()
