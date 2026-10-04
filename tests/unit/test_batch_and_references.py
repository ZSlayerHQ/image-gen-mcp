"""Batch output (n), several input images and moderation on edits.

The defaults must leave existing behaviour exactly as it was: the arguments that
reach the SDK, the cache keys, the stored metadata and the result dictionaries.
These tests pin that contract and the new features with deterministic fakes; no
network is used.
"""

import base64
import dataclasses
import hashlib
import io
import itertools
import json
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from PIL import Image

from image_gen_mcp import server
from image_gen_mcp.config.settings import (
    CacheSettings,
    ImageSettings,
    OpenAISettings,
    ProvidersSettings,
    Settings,
    StorageSettings,
)
from image_gen_mcp.providers.base import (
    ImageResponse,
    LLMProvider,
    ProviderConfig,
    ProviderError,
)
from image_gen_mcp.providers.gemini import GeminiProvider
from image_gen_mcp.providers.openai import OpenAIProvider
from image_gen_mcp.resources.image_resources import ImageResourceManager
from image_gen_mcp.storage.manager import ImageStorageManager
from image_gen_mcp.tools.batch import (
    BATCH_CAPABLE_MODELS,
    REFERENCE_COST_NOTE,
    PartialSaveError,
)
from image_gen_mcp.tools.image_editing import ImageEditingTool
from image_gen_mcp.tools.image_generation import ImageGenerationTool
from image_gen_mcp.utils.cache import CacheManager
from image_gen_mcp.utils.openai_client import OpenAIClientManager

PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6"
    "kgAAAABJRU5ErkJggg=="
)
PNG_BYTES = base64.b64decode(PNG_B64)
PNG_URL = f"data:image/png;base64,{PNG_B64}"


def png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# Ten distinct outputs, so a test can tell which image landed where.
OUTPUTS = [png((25 * i, 255 - 25 * i, 100)) for i in range(10)]
assert len(set(OUTPUTS)) == 10


def sdk_response(count: int, usage=None) -> SimpleNamespace:
    """An Images API response shaped like the SDK's, with ``count`` distinct images."""
    return SimpleNamespace(
        data=[SimpleNamespace(b64_json=b64(OUTPUTS[i]), url=None) for i in range(count)],
        usage=usage,
        created=1700000000,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        providers=ProvidersSettings(
            openai=OpenAISettings(api_key="test-key-not-real", max_retries=0)
        ),
        storage=StorageSettings(base_path=str(tmp_path / "storage")),
        cache=CacheSettings(enabled=True, backend="memory"),
        images=ImageSettings(default_model="gpt-image-2.5-sunburst"),
    )


@pytest_asyncio.fixture
async def storage(settings):
    manager = ImageStorageManager(settings.storage)
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.close()


@pytest_asyncio.fixture
async def cache(settings):
    manager = CacheManager(settings.cache)
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.close()


@pytest.fixture
def fixed_ids(monkeypatch, storage):
    """Deterministic image IDs (dated today, so history lists them) and task IDs."""
    today = datetime.now().strftime("%Y%m%d")
    counter = itertools.count(1)
    monkeypatch.setattr(
        storage, "generate_image_id", lambda: f"img_{today}000000_{next(counter):012d}"
    )
    tasks = itertools.count(1)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(tasks)))
    return lambda number: f"img_{today}000000_{number:012d}"


@pytest.fixture
def generation_tool(storage, cache, settings):
    return ImageGenerationTool(storage, cache, settings)


@pytest_asyncio.fixture
async def openai_provider(generation_tool):
    await generation_tool._ensure_providers_registered()
    return generation_tool.provider_registry.get_provider_for_model("gpt-image-2.5-flare")


@pytest.fixture
def sdk_generate(openai_provider):
    """The real tool and provider path, down to a fake SDK images.generate."""
    generate = AsyncMock(return_value=sdk_response(1))
    openai_provider.client = SimpleNamespace(images=SimpleNamespace(generate=generate))
    return generate


@pytest.fixture
def fake_provider(monkeypatch):
    """Replace OpenAIProvider.generate_image; record its kwargs; return N images."""
    state = {"calls": [], "count": 1}

    async def generate_image(self, **kwargs):
        state["calls"].append(kwargs)
        images = OUTPUTS[: state["count"]]
        return ImageResponse(
            image_data=images[0], metadata={}, additional_images=images[1:]
        )

    monkeypatch.setattr(OpenAIProvider, "generate_image", generate_image)
    return state


@pytest.fixture
def edit_client(settings):
    client = OpenAIClientManager(settings.providers.openai)
    client.edit_image = AsyncMock(return_value=sdk_response(1))
    return client


@pytest.fixture
def editing_tool(storage, cache, settings, edit_client):
    return ImageEditingTool(storage, cache, settings, openai_client=edit_client)


def count_estimates(monkeypatch, owner) -> list[int]:
    """Record the image count every cost estimate is asked for."""
    real = owner.estimate_cost
    counts: list[int] = []

    def estimate_cost(*args, **kwargs):
        counts.append(args[2] if owner.__class__ is OpenAIProvider else args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, "estimate_cost", estimate_cost)
    return counts


def forbid_cache_lookup(monkeypatch, cache):
    """Fail the test if the cache is consulted (the rejection must come first)."""
    for name in ("get_image_generation", "get_image_edit"):
        monkeypatch.setattr(
            cache, name, AsyncMock(side_effect=AssertionError("cache consulted"))
        )


def stored_metadata(storage, image_id: str) -> dict:
    return json.loads(storage.get_metadata_path(image_id).read_text())


async def history(storage, settings) -> dict[str, dict]:
    listing = json.loads(
        await ImageResourceManager(storage, settings.storage).get_recent_images(limit=50)
    )
    return {entry["image_id"]: entry for entry in listing["images"]}


LEGACY_HISTORY_KEYS = {
    "image_id", "created_at", "prompt", "resource_uri", "file_size_bytes",
    "dimensions", "format", "parameters", "cost_estimate",
}


# --- generation: the default path is golden -------------------------------------


GEN_ARGS = dict(prompt="a red cube", model="gpt-image-2.5-flare", quality="low", size="1024x1024")
LEGACY_GEN_PARAMS = {
    "prompt": "a red cube", "quality": "low", "size": "1024x1024", "style": "vivid",
    "moderation": "auto", "output_format": "png", "compression": 100,
    "background": "auto", "model": "gpt-image-2.5-flare",
}
# The SDK arguments and the request estimate the unpatched server produces for
# GEN_ARGS (golden probe against the pre-change tree, 2026-10-04).
GOLDEN_GEN_SDK_KWARGS = {
    "model": "gpt-image-2.5-flare", "prompt": "a red cube", "n": 1, "quality": "low",
    "moderation": "auto", "size": "1024x1024", "output_format": "png",
    "background": "auto",
}
GOLDEN_FLARE_LOW_SQUARE_COST = 0.0059


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_default", [False, True])
async def test_generate_default_is_golden(
    generation_tool, sdk_generate, storage, cache, settings, fixed_ids, explicit_default
):
    extra = {"n": 1} if explicit_default else {}
    result = await generation_tool.generate(**GEN_ARGS, **extra)

    assert [call.kwargs for call in sdk_generate.call_args_list] == [GOLDEN_GEN_SDK_KWARGS]
    image_id = fixed_ids(1)
    assert result == {
        "task_id": "00000000-0000-0000-0000-000000000001",
        "image_id": image_id,
        "image_url": f"file://{storage.get_image_path(image_id, 'png').absolute()}",
        "resource_uri": f"generated-images://{image_id}",
        "metadata": {
            "model": "gpt-image-2.5-flare", "provider": "openai", "size": "1024x1024",
            "quality": "low", "style": "vivid", "moderation": "auto",
            "output_format": "png", "background": "auto", "prompt": "a red cube",
            "created_at": None, "cost_estimate": GOLDEN_FLARE_LOW_SQUARE_COST,
            "file_size_bytes": len(OUTPUTS[0]), "dimensions": "1024x1024",
            "format": "PNG",
        },
    }
    stored = stored_metadata(storage, image_id)
    assert "batch_index" not in stored
    assert set(stored["cost_info"]) == {
        "breakdown", "currency", "estimate_accuracy", "estimated_cost_usd", "model",
        "provider",
    }
    assert set((await history(storage, settings))[image_id]) == LEGACY_HISTORY_KEYS
    # The result is cached under exactly the pre-change key.
    assert cache.cache.get(cache._make_key("image_gen", **LEGACY_GEN_PARAMS)) == result


# --- generation: batches ----------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_batch_saves_every_image_in_order(
    generation_tool, sdk_generate, openai_provider, storage, cache, settings, monkeypatch
):
    sdk_generate.return_value = sdk_response(3)
    counts = count_estimates(monkeypatch, openai_provider)
    result = await generation_tool.generate(**GEN_ARGS, n=3)

    assert sdk_generate.await_count == 1
    assert sdk_generate.call_args.kwargs["n"] == 3
    assert counts == [3]
    total = openai_provider.estimate_cost(
        "gpt-image-2.5-flare", "a red cube", 3, quality="low", size="1024x1024"
    )["estimated_cost_usd"]
    assert total > GOLDEN_FLARE_LOW_SQUARE_COST
    metadata = result["metadata"]
    assert metadata["cost_estimate"] == total
    assert metadata["cost_estimate_scope"] == "request_total"
    assert (metadata["images_requested"], metadata["images_returned"]) == (3, 3)

    ids = [result["image_id"]] + [extra["image_id"] for extra in result["additional_images"]]
    assert len(set(ids)) == 3
    primary_bytes, _ = await storage.load_image(result["image_id"])
    assert primary_bytes == OUTPUTS[0]
    for position, extra in enumerate(result["additional_images"], start=1):
        assert set(extra) == {
            "image_id", "image_url", "resource_uri", "local_path", "file_size_bytes",
        }
        assert Path(extra["local_path"]).read_bytes() == OUTPUTS[position]
        assert extra["file_size_bytes"] == len(OUTPUTS[position])
        assert extra["image_url"] == f"file://{Path(extra['local_path']).absolute()}"
        assert extra["resource_uri"] == f"generated-images://{extra['image_id']}"

    shares = []
    listed = await history(storage, settings)
    for position, image_id in enumerate(ids):
        cost = stored_metadata(storage, image_id)["cost_info"]
        assert stored_metadata(storage, image_id)["batch_index"] == position
        assert cost["scope"] == "per_image_share_of_request"
        assert cost["request_images"] == 3
        assert cost["request"]["estimated_cost_usd"] == total
        assert "note" not in cost
        shares.append(cost["estimated_cost_usd"])
        entry = listed[image_id]
        assert entry["cost_estimate"] == cost["estimated_cost_usd"]
        assert entry["cost_scope"] == "per_image_share_of_request"
        assert entry["batch_index"] == position
        assert "cost_note" not in entry
    assert sum(shares) == pytest.approx(total, abs=1e-5)

    assert cache.cache.get(cache._make_key("image_gen", **LEGACY_GEN_PARAMS, n=3)) == result
    assert cache.cache.get(cache._make_key("image_gen", **LEGACY_GEN_PARAMS)) is None


@pytest.mark.asyncio
async def test_generate_short_batch_is_reported_not_topped_up(
    generation_tool, sdk_generate, openai_provider, storage, monkeypatch
):
    sdk_generate.return_value = sdk_response(2)
    counts = count_estimates(monkeypatch, openai_provider)
    result = await generation_tool.generate(**GEN_ARGS, n=3)

    assert sdk_generate.await_count == 1
    assert counts == [2]
    assert result["metadata"]["images_requested"] == 3
    assert result["metadata"]["images_returned"] == 2
    [extra] = result["additional_images"]
    assert Path(extra["local_path"]).read_bytes() == OUTPUTS[1]
    assert stored_metadata(storage, extra["image_id"])["cost_info"]["request_images"] == 2


@pytest.mark.asyncio
async def test_generate_batch_cache_hit_returns_the_batch(generation_tool, sdk_generate):
    sdk_generate.return_value = sdk_response(2)
    first = await generation_tool.generate(**GEN_ARGS, n=2)
    again = await generation_tool.generate(**GEN_ARGS, n=2)
    assert again == first
    assert sdk_generate.await_count == 1
    await generation_tool.generate(**GEN_ARGS, n=3)
    await generation_tool.generate(**GEN_ARGS)
    assert sdk_generate.await_count == 3  # other counts are separate entries


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-image-1", "gpt-image-1.5", "dall-e-2"])
async def test_generate_rejects_batch_outside_scope(
    generation_tool, sdk_generate, cache, monkeypatch, model
):
    forbid_cache_lookup(monkeypatch, cache)
    with pytest.raises(ValueError, match="supported only for"):
        await generation_tool.generate(prompt="p", model=model, n=2)
    sdk_generate.assert_not_awaited()


class OfflineGemini(GeminiProvider):
    """GeminiProvider routing without its service-account loading."""

    def __init__(self):
        LLMProvider.__init__(self, ProviderConfig(api_key="test-key-not-real"))
        self.name = "gemini"

    def is_available(self) -> bool:
        return True

    async def generate_image(self, **kwargs):
        raise AssertionError("API called")


@pytest.mark.asyncio
async def test_generate_rejects_gemini_batch(generation_tool, cache, monkeypatch):
    await generation_tool._ensure_providers_registered()
    await generation_tool.provider_registry.register_provider(OfflineGemini())
    assert generation_tool.provider_registry.get_provider_for_model("imagen-4").name == "gemini"
    forbid_cache_lookup(monkeypatch, cache)
    with pytest.raises(ValueError, match="supported only for"):
        await generation_tool.generate(prompt="p", model="imagen-4", n=2)


@pytest.mark.asyncio
async def test_capability_limit_still_rejects_an_allowed_model(
    generation_tool, sdk_generate, editing_tool, edit_client, cache, monkeypatch
):
    capability = OpenAIProvider.SUPPORTED_MODELS["gpt-image-2.5-flare"]
    monkeypatch.setitem(
        OpenAIProvider.SUPPORTED_MODELS,
        "gpt-image-2.5-flare",
        dataclasses.replace(capability, max_images_per_request=2),
    )
    forbid_cache_lookup(monkeypatch, cache)
    with pytest.raises(ValueError, match="at most 2 image"):
        await generation_tool.generate(**GEN_ARGS, n=3)
    with pytest.raises(ValueError, match="at most 2 image"):
        await editing_tool.edit(**EDIT_CALL, n=3)
    sdk_generate.assert_not_awaited()
    edit_client.edit_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_output_for_a_batch_request_is_a_short_batch(
    generation_tool, sdk_generate, editing_tool, edit_client, storage
):
    result = await generation_tool.generate(**GEN_ARGS, n=2)
    edited = await editing_tool.edit(**EDIT_CALL, n=2)
    for item in (result, edited):
        assert (item["metadata"]["images_requested"], item["metadata"]["images_returned"]) == (2, 1)
        assert item["additional_images"] == []
        stored = stored_metadata(storage, item["image_id"])
        assert stored["batch_index"] == 0
        assert stored["cost_info"]["request_images"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["dall-e-2", "imagen-4"])
async def test_edit_rejects_non_edit_models_before_cache(
    editing_tool, edit_client, cache, monkeypatch, model
):
    forbid_cache_lookup(monkeypatch, cache)
    with pytest.raises(ValueError, match="cannot edit images"):
        await editing_tool.edit(**{**EDIT_CALL, "model": model}, n=2)
    edit_client.edit_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, 11, True, 2.0])
async def test_generate_rejects_bad_n(generation_tool, sdk_generate, bad):
    with pytest.raises(ValueError, match="n must be an integer"):
        await generation_tool.generate(**GEN_ARGS, n=bad)
    sdk_generate.assert_not_awaited()


# --- partial saves (both tools) ---------------------------------------------------


def failing_saves(storage, monkeypatch, fail_on: int, where: str) -> dict:
    """Make save number ``fail_on`` fail: before anything is written ("file"), or
    with a real metadata-write error after its image file is written ("metadata")."""
    real_save = storage.save_image
    real_metadata_path = storage.get_metadata_path
    state = {"saves": 0, "saved": [], "failing": False}

    async def save_image(**kwargs):
        state["saves"] += 1
        state["failing"] = state["saves"] == fail_on
        try:
            if state["failing"] and where == "file":
                raise OSError("disk full")
            saved = await real_save(**kwargs)
            state["saved"].append(saved)
            return saved
        finally:
            state["failing"] = False

    def get_metadata_path(image_id):
        path = real_metadata_path(image_id)
        if state["failing"] and where == "metadata":
            path.mkdir(parents=True, exist_ok=True)  # a folder where the file must go
        return path

    monkeypatch.setattr(storage, "save_image", save_image)
    monkeypatch.setattr(storage, "get_metadata_path", get_metadata_path)
    return state


def assert_partial_save(error, state, storage, cache, fail_on: int, where: str):
    message = str(error)
    assert f"saving image {fail_on} of 3 failed:" in message
    assert "Do not repeat the request to recover them: it would be billed again." in message
    if fail_on == 1:
        assert "Already saved: none." in message
    for position, (image_id, path) in enumerate(state["saved"]):
        assert f"{image_id} at {path}" in message
        assert Path(path).read_bytes() == OUTPUTS[position]
    assert len(state["saved"]) == fail_on - 1
    assert state["saves"] == fail_on  # nothing after the failure
    image_files = list(storage.images_path.rglob("*.png"))
    # A metadata failure leaves its image file behind (accepted orphan).
    assert len(image_files) == fail_on - (1 if where == "file" else 0)
    assert len(cache.cache.cache) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", [1, 2])
@pytest.mark.parametrize("where", ["file", "metadata"])
async def test_generate_partial_save_reports_what_was_saved(
    generation_tool, sdk_generate, storage, cache, monkeypatch, fail_on, where
):
    sdk_generate.return_value = sdk_response(3)
    state = failing_saves(storage, monkeypatch, fail_on, where)
    with pytest.raises(PartialSaveError) as info:
        await generation_tool.generate(**GEN_ARGS, n=3)
    assert sdk_generate.await_count == 1
    assert str(info.value).startswith("Image generation succeeded but")
    assert_partial_save(info.value, state, storage, cache, fail_on, where)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", [1, 2])
@pytest.mark.parametrize("where", ["file", "metadata"])
async def test_edit_partial_save_reports_what_was_saved(
    editing_tool, edit_client, storage, cache, monkeypatch, fail_on, where
):
    edit_client.edit_image.return_value = sdk_response(3)
    state = failing_saves(storage, monkeypatch, fail_on, where)
    with pytest.raises(PartialSaveError) as info:
        await editing_tool.edit(**EDIT_CALL, n=3)
    assert edit_client.edit_image.await_count == 1
    assert str(info.value).startswith("Image edit succeeded but")
    assert_partial_save(info.value, state, storage, cache, fail_on, where)


@pytest.mark.asyncio
async def test_single_image_save_failure_keeps_the_old_error(
    generation_tool, sdk_generate, editing_tool, storage, monkeypatch
):
    failing_saves(storage, monkeypatch, 1, "file")
    with pytest.raises(RuntimeError, match="^Image generation failed: disk full$"):
        await generation_tool.generate(**GEN_ARGS)
    failing_saves(storage, monkeypatch, 1, "file")
    with pytest.raises(RuntimeError, match="^Image editing failed: disk full$"):
        await editing_tool.edit(**EDIT_CALL)


# --- provider decoding ------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_decodes_every_image_in_order():
    provider = OpenAIProvider(ProviderConfig(api_key="test-key-not-real", max_retries=0))
    generate = AsyncMock(return_value=sdk_response(3))
    provider.client = SimpleNamespace(images=SimpleNamespace(generate=generate))
    response = await provider.generate_image(
        model="gpt-image-2.5-flare", prompt="p", quality="low", size="1024x1024", n=3
    )
    assert response.image_data == OUTPUTS[0]
    assert response.additional_images == OUTPUTS[1:3]
    assert generate.call_args.kwargs["n"] == 3


@pytest.mark.asyncio
async def test_provider_url_fallback_keeps_order(monkeypatch):
    provider = OpenAIProvider(ProviderConfig(api_key="test-key-not-real", max_retries=0))
    data = [SimpleNamespace(b64_json=None, url=f"https://example.invalid/{i}") for i in range(2)]
    provider.client = SimpleNamespace(
        images=SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(data=data)))
    )
    download = AsyncMock(side_effect=[OUTPUTS[0], OUTPUTS[1]])
    monkeypatch.setattr(provider, "_download_image", download)
    response = await provider.generate_image(model="gpt-image-2", prompt="p", n=2)
    assert [call.args[0] for call in download.call_args_list] == [
        "https://example.invalid/0", "https://example.invalid/1",
    ]
    assert (response.image_data, response.additional_images) == (OUTPUTS[0], [OUTPUTS[1]])


@pytest.mark.asyncio
async def test_provider_rejects_empty_and_unreadable_outputs():
    provider = OpenAIProvider(ProviderConfig(api_key="test-key-not-real", max_retries=0))
    provider.client = SimpleNamespace(
        images=SimpleNamespace(generate=AsyncMock(return_value=sdk_response(0)))
    )
    with pytest.raises(ProviderError, match="no images"):
        await provider.generate_image(model="gpt-image-2.5-flare", prompt="p")

    malformed = sdk_response(2)
    malformed.data[1] = SimpleNamespace(b64_json=None, url=None)
    provider.client.images.generate = AsyncMock(return_value=malformed)
    with pytest.raises(ProviderError, match="neither base64 data nor URL"):
        await provider.generate_image(model="gpt-image-2.5-flare", prompt="p", n=2)


def test_batch_scope_is_an_allowlist_not_the_capability_count():
    caps = OpenAIProvider.SUPPORTED_MODELS
    assert BATCH_CAPABLE_MODELS == {
        "gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare",
    }
    assert all(caps[model].max_images_per_request == 10 for model in BATCH_CAPABLE_MODELS)
    assert caps["gpt-image-1"].max_images_per_request == 1
    # DALL-E 2 declares 10 but is retired and outside the batch scope.
    assert caps["dall-e-2"].max_images_per_request == 10


# --- edit: the SDK call -------------------------------------------------------


LEGACY_EDIT_SDK_KWARGS = {
    "model": "gpt-image-2.5-flare",
    "image": ("image.png", PNG_BYTES, "image/png"),
    "prompt": "make it blue",
    "n": 1,
    "size": "1024x1024",
    "quality": "low",
    "output_format": "png",
    "background": "auto",
}


def sdk_edit_mock(manager: OpenAIClientManager) -> AsyncMock:
    edit = AsyncMock(return_value=sdk_response(1))
    manager._client = SimpleNamespace(images=SimpleNamespace(edit=edit))
    return edit


EDIT_CALL = dict(
    image_data=PNG_B64, prompt="make it blue", model="gpt-image-2.5-flare",
    quality="low", size="1024x1024",
)


@pytest.mark.asyncio
async def test_client_edit_defaults_unchanged(settings):
    manager = OpenAIClientManager(settings.providers.openai)
    edit = sdk_edit_mock(manager)
    await manager.edit_image(**EDIT_CALL)
    await manager.edit_image(**EDIT_CALL, additional_images=None, moderation="auto")
    await manager.edit_image(**EDIT_CALL, additional_images=[], moderation="auto")
    for call in edit.call_args_list:
        assert call.kwargs == LEGACY_EDIT_SDK_KWARGS


@pytest.mark.asyncio
async def test_client_edit_sends_ordered_list_and_moderation(settings):
    manager = OpenAIClientManager(settings.providers.openai)
    edit = sdk_edit_mock(manager)
    jpeg = b"\xff\xd8\xff\xe0" + b"0" * 16
    await manager.edit_image(
        **EDIT_CALL,
        additional_images=[base64.b64encode(jpeg).decode(), PNG_URL],
        moderation="low",
        n=2,
    )
    kwargs = edit.call_args.kwargs
    assert [part[1] for part in kwargs["image"]] == [PNG_BYTES, jpeg, PNG_BYTES]
    assert kwargs["extra_body"] == {"moderation": "low"}
    assert kwargs["n"] == 2


@pytest.mark.asyncio
async def test_real_sdk_multipart_encoding(settings):
    """The installed SDK puts the inputs in order as image[] and moderation as a field."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(200, json={"created": 0, "data": [{"b64_json": PNG_B64}]})

    manager = OpenAIClientManager(settings.providers.openai)
    manager._client = AsyncOpenAI(
        api_key="test-key-not-real",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    second = b"\x89PNG\r\n\x1a\n" + b"second-image"
    await manager.edit_image(
        **EDIT_CALL,
        additional_images=[base64.b64encode(second).decode()],
        moderation="low",
    )
    body = seen["body"]
    assert seen["path"].endswith("/images/edits")
    assert body.count(b'name="image[]"') == 2
    assert body.index(PNG_BYTES) < body.index(b"second-image")
    assert b'name="moderation"\r\n\r\nlow' in body

    seen.clear()
    await manager.edit_image(**EDIT_CALL)
    assert b'name="image"' in seen["body"] and b'name="image[]"' not in seen["body"]
    assert b'name="moderation"' not in seen["body"]


# --- edit: the tool -------------------------------------------------------------


LEGACY_EDIT_CACHE = {
    "image_data": PNG_B64, "prompt": "make it blue", "mask_data": None,
    "quality": "low", "size": "1024x1024", "output_format": "png",
    "compression": 100, "background": "auto", "model": "gpt-image-2.5-flare",
}


def edit_key(cache: CacheManager, **params) -> str:
    """The edit cache key exactly as CacheManager computes it."""
    key_params = {k: v for k, v in params.items() if k != "image_data"}
    key_params["image_hash"] = hashlib.sha256(params["image_data"].encode()).hexdigest()[:16]
    return cache._make_key("image_edit", **key_params)


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_default", [False, True])
async def test_edit_default_is_golden(storage, cache, settings, fixed_ids, explicit_default):
    manager = OpenAIClientManager(settings.providers.openai)
    edit = sdk_edit_mock(manager)
    tool = ImageEditingTool(storage, cache, settings, openai_client=manager)
    extra = (
        {"n": 1, "moderation": "auto", "additional_images": None}
        if explicit_default
        else {}
    )
    result = await tool.edit(**EDIT_CALL, **extra)

    assert [call.kwargs for call in edit.call_args_list] == [LEGACY_EDIT_SDK_KWARGS]
    image_id = fixed_ids(1)
    path = storage.get_image_path(image_id, "png")
    assert result == {
        "task_id": "00000000-0000-0000-0000-000000000001",
        "image_id": image_id,
        "image_url": f"file://{path.absolute()}",
        "resource_uri": f"generated-images://{image_id}",
        "operation": "edit",
        "metadata": {
            "model": "gpt-image-2.5-flare", "size": "1024x1024", "quality": "low",
            "output_format": "png", "background": "auto", "prompt": "make it blue",
            "has_mask": False, "created_at": None,
            "cost_estimate": GOLDEN_FLARE_LOW_SQUARE_COST,
            "file_size_bytes": len(OUTPUTS[0]), "dimensions": "1024x1024",
            "format": "PNG", "local_path": str(path),
        },
    }
    stored = stored_metadata(storage, image_id)
    assert not {"batch_index", "reference_image_count"} & set(stored)
    assert "moderation" not in stored["parameters"]
    assert not {"scope", "note", "request"} & set(stored["cost_info"])
    assert set((await history(storage, settings))[image_id]) == LEGACY_HISTORY_KEYS
    assert cache.cache.get(edit_key(cache, **LEGACY_EDIT_CACHE)) == result


@pytest.mark.asyncio
async def test_edit_cache_keys_the_raw_data_url(editing_tool, edit_client, cache):
    result = await editing_tool.edit(**{**EDIT_CALL, "image_data": PNG_URL})
    assert edit_client.edit_image.call_args.kwargs["image_data"] == PNG_B64
    assert cache.cache.get(edit_key(cache, **{**LEGACY_EDIT_CACHE, "image_data": PNG_URL})) == result
    assert cache.cache.get(edit_key(cache, **LEGACY_EDIT_CACHE)) is None


@pytest.mark.asyncio
async def test_edit_batch_with_references(
    editing_tool, edit_client, storage, cache, settings, monkeypatch
):
    edit_client.edit_image.return_value = sdk_response(3)
    counts = count_estimates(monkeypatch, edit_client)
    refs = [PNG_URL, PNG_B64]
    result = await editing_tool.edit(**EDIT_CALL, additional_images=refs, moderation="low", n=3)

    kwargs = edit_client.edit_image.call_args.kwargs
    assert (kwargs["additional_images"], kwargs["moderation"], kwargs["n"]) == (refs, "low", 3)
    assert counts == [3]
    total = edit_client.estimate_cost(
        "make it blue", 3, model="gpt-image-2.5-flare", quality="low", size="1024x1024"
    )["estimated_cost_usd"]
    metadata = result["metadata"]
    assert metadata["cost_estimate"] == total
    assert metadata["cost_estimate_scope"] == "request_total"
    assert metadata["reference_image_count"] == 2
    assert metadata["cost_note"] == f"cost_estimate {REFERENCE_COST_NOTE}"
    assert (metadata["images_requested"], metadata["images_returned"]) == (3, 3)
    assert Path(metadata["local_path"]).read_bytes() == OUTPUTS[0]
    for position, extra in enumerate(result["additional_images"], start=1):
        assert Path(extra["local_path"]).read_bytes() == OUTPUTS[position]

    ids = [result["image_id"]] + [extra["image_id"] for extra in result["additional_images"]]
    listed = await history(storage, settings)
    shares = []
    for position, image_id in enumerate(ids):
        stored = stored_metadata(storage, image_id)
        assert stored["batch_index"] == position
        assert stored["reference_image_count"] == 2
        assert stored["parameters"]["moderation"] == "low"
        cost = stored["cost_info"]
        assert cost["scope"] == "per_image_share_of_request"
        assert cost["note"] == REFERENCE_COST_NOTE
        assert cost["request"]["estimated_cost_usd"] == total
        shares.append(cost["estimated_cost_usd"])
        entry = listed[image_id]
        assert (entry["cost_scope"], entry["cost_note"], entry["batch_index"]) == (
            "per_image_share_of_request", REFERENCE_COST_NOTE, position,
        )
        assert entry["cost_estimate"] == cost["estimated_cost_usd"]
    assert sum(shares) == pytest.approx(total, abs=1e-5)
    assert cache.cache.get(edit_key(cache, **LEGACY_EDIT_CACHE)) is None


@pytest.mark.asyncio
async def test_edit_single_reference_keeps_its_cost_caveat(
    editing_tool, edit_client, storage, settings
):
    result = await editing_tool.edit(**EDIT_CALL, additional_images=[PNG_B64])

    assert "additional_images" not in result
    assert "images_requested" not in result["metadata"]
    assert "cost_estimate_scope" not in result["metadata"]
    assert result["metadata"]["reference_image_count"] == 1
    stored = stored_metadata(storage, result["image_id"])
    assert "batch_index" not in stored
    assert stored["cost_info"]["note"] == REFERENCE_COST_NOTE
    assert stored["cost_info"]["estimated_cost_usd"] == result["metadata"]["cost_estimate"]
    assert "scope" not in stored["cost_info"]
    entry = (await history(storage, settings))[result["image_id"]]
    assert entry["cost_note"] == REFERENCE_COST_NOTE
    assert not {"cost_scope", "batch_index"} & set(entry)


@pytest.mark.asyncio
async def test_edit_short_and_empty_responses(
    editing_tool, edit_client, storage, cache, monkeypatch
):
    counts = count_estimates(monkeypatch, edit_client)
    edit_client.edit_image.return_value = sdk_response(2)
    result = await editing_tool.edit(**EDIT_CALL, n=3)
    assert counts == [2]
    assert (result["metadata"]["images_requested"], result["metadata"]["images_returned"]) == (3, 2)
    assert len(result["additional_images"]) == 1

    before = set(storage.images_path.rglob("*.png"))
    entries = len(cache.cache.cache)
    edit_client.edit_image.return_value = sdk_response(0)
    with pytest.raises(RuntimeError, match="no images"):
        await editing_tool.edit(**{**EDIT_CALL, "prompt": "another"}, n=2)
    malformed = sdk_response(2)
    malformed.data[1] = SimpleNamespace(b64_json=None, url=None)
    edit_client.edit_image.return_value = malformed
    with pytest.raises(RuntimeError, match="Image editing failed"):
        await editing_tool.edit(**{**EDIT_CALL, "prompt": "a third"}, n=2)
    assert set(storage.images_path.rglob("*.png")) == before
    assert len(cache.cache.cache) == entries


@pytest.mark.asyncio
async def test_edit_cache_separates_references_counts_and_moderation(editing_tool, edit_client):
    other = base64.b64encode(b"\x89PNG\r\n\x1a\nother").decode()
    edit_client.edit_image.return_value = sdk_response(2)
    first = await editing_tool.edit(**EDIT_CALL, additional_images=[PNG_B64, other], moderation="low", n=2)
    again = await editing_tool.edit(**EDIT_CALL, additional_images=[PNG_B64, other], moderation="low", n=2)
    assert again == first
    assert edit_client.edit_image.await_count == 1
    for variant in (
        dict(additional_images=[other, PNG_B64], moderation="low", n=2),  # order
        dict(additional_images=[PNG_B64], moderation="low", n=2),  # set
        dict(additional_images=[PNG_B64, other], moderation="auto", n=2),  # moderation
        dict(additional_images=[PNG_B64, other], moderation="low", n=3),  # count
    ):
        await editing_tool.edit(**EDIT_CALL, **variant)
    assert edit_client.edit_image.await_count == 5
    await editing_tool.edit(**EDIT_CALL, additional_images=[])
    await editing_tool.edit(**EDIT_CALL)
    assert edit_client.edit_image.await_count == 6  # empty list and None share the default key


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n": 0}, "n must be an integer"),
        ({"n": 11}, "n must be an integer"),
        ({"n": True}, "n must be an integer"),
        ({"moderation": "high"}, "moderation must be"),
        ({"additional_images": [PNG_B64] * 16}, "at most 15"),
        ({"model": "gpt-image-1", "n": 2}, "supported only for"),
        ({"model": "gpt-image-1.5", "n": 2}, "supported only for"),
    ],
)
async def test_edit_rejects_before_cache_and_api(
    editing_tool, edit_client, cache, monkeypatch, kwargs, message
):
    forbid_cache_lookup(monkeypatch, cache)
    call = {**EDIT_CALL, **kwargs}
    with pytest.raises(ValueError, match=message):
        await editing_tool.edit(**call)
    edit_client.edit_image.assert_not_awaited()


# --- the MCP tool functions -----------------------------------------------------


@pytest.fixture
def server_tools(monkeypatch):
    generation = AsyncMock()
    generation.generate.return_value = {"tool": "generate"}
    editing = AsyncMock()
    editing.edit.return_value = {"tool": "edit"}
    context = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(
                image_generation_tool=generation, image_editing_tool=editing
            )
        )
    )
    monkeypatch.setattr(server, "mcp", SimpleNamespace(get_context=lambda: context))
    return generation, editing


SERVER_GENERATE_ARGS = dict(
    prompt="a cube", model="gpt-image-2.5-flare", quality="low", size="1024x1024",
    style="vivid", moderation="auto", output_format="png", compression=100,
    background="auto",
)
SERVER_EDIT_ARGS = dict(
    image_data=PNG_B64, prompt="make it blue", mask_data=None, size="1024x1024",
    quality="low", output_format="png", compression=100, background="auto",
    model="gpt-image-2.5-flare",
)


@pytest.mark.asyncio
async def test_server_generate_forwards_n(server_tools):
    generation, _ = server_tools
    await server.generate_image(**SERVER_GENERATE_ARGS)  # n omitted: arrives as its Field
    assert generation.generate.call_args.kwargs["n"] == 1
    await server.generate_image(**SERVER_GENERATE_ARGS, n=4)
    assert generation.generate.call_args.kwargs["n"] == 4


@pytest.mark.asyncio
async def test_server_edit_forwards_new_fields(server_tools):
    _, editing = server_tools
    await server.edit_image(**SERVER_EDIT_ARGS)  # new fields omitted: arrive as Fields
    kwargs = editing.edit.call_args.kwargs
    assert (kwargs["additional_images"], kwargs["moderation"], kwargs["n"]) == (None, "auto", 1)

    await server.edit_image(
        **SERVER_EDIT_ARGS, additional_images=[PNG_B64, PNG_URL], moderation="low", n=2
    )
    kwargs = editing.edit.call_args.kwargs
    assert kwargs["additional_images"] == [PNG_URL, PNG_URL]  # validated to data URLs
    assert (kwargs["moderation"], kwargs["n"]) == ("low", 2)

    await server.edit_image(**SERVER_EDIT_ARGS, additional_images=[])
    assert editing.edit.call_args.kwargs["additional_images"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad, message",
    [("not base64!", "Invalid base64 image data"), ("data:image/png;base64,@@@", "Invalid data URL")],
)
async def test_server_edit_rejects_invalid_reference(server_tools, bad, message):
    _, editing = server_tools
    with pytest.raises(ValueError, match=message):
        await server.edit_image(**SERVER_EDIT_ARGS, additional_images=[PNG_B64, bad])
    editing.edit.assert_not_awaited()
