# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def parse_config(args: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


@pytest.mark.parametrize(
    "env, args, expected",
    [
        (None, [], False),
        (None, ["--frontend-image-fetch"], True),
        ("true", [], True),
        ("false", ["--frontend-image-fetch"], True),
        ("true", ["--no-frontend-image-fetch"], False),
    ],
)
def test_frontend_image_fetch_config(monkeypatch, env, args, expected):
    monkeypatch.delenv("DYN_FRONTEND_IMAGE_FETCH", raising=False)
    if env is not None:
        monkeypatch.setenv("DYN_FRONTEND_IMAGE_FETCH", env)
    before = dict(os.environ)
    assert parse_config(args).frontend_image_fetch is expected
    assert dict(os.environ) == before


@pytest.mark.parametrize("processor", ["vllm", "sglang"])
def test_frontend_image_fetch_requires_rust_processor(processor):
    with pytest.raises(ValueError, match="requires --dyn-chat-processor dynamo"):
        parse_config(["--frontend-image-fetch", "--dyn-chat-processor", processor])


def test_frontend_image_fetch_binding_accepts_resolved_config(monkeypatch):
    from dynamo._core import EngineType, EntrypointArgs

    monkeypatch.setenv("DYN_FRONTEND_IMAGE_FETCH", "true")
    for args in [[], ["--no-frontend-image-fetch"]]:
        config = parse_config(args)
        EntrypointArgs(
            EngineType.Dynamic, frontend_image_fetch=config.frontend_image_fetch
        )
    with pytest.raises(ValueError, match="requires the Rust chat preprocessor"):
        EntrypointArgs(
            EngineType.Dynamic,
            frontend_image_fetch=True,
            chat_engine_factory=lambda *_: None,
        )
