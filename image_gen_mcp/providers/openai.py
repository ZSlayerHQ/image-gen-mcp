"""OpenAI provider implementation."""

import base64
import logging
import math
from typing import Any

import httpx
from openai import AsyncOpenAI

from .base import (
    ImageResponse,
    LLMProvider,
    ModelCapability,
    ProviderConfig,
    ProviderError,
)

logger = logging.getLogger(__name__)


# Per-image output token approximations at 1024x1024, sampled from OpenAI's
# image-generation calculator. Token cost scales roughly linearly with pixel
# count at a given quality tier. These are ROUGH and only intended to give
# callers an order-of-magnitude estimate — the actual calculator uses a more
# complex model we don't replicate here.
_GPT_IMAGE_2_TOKENS_BASE_1024: dict[str, int] = {
    "low": 170,       # ≈ $0.005 at $30/1M
    "medium": 1370,   # ≈ $0.041
    "high": 5500,     # ≈ $0.165
    "auto": 1370,     # treat auto as medium for estimation
}

# Token pricing for gpt-image-* models. Shared with OpenAIClientManager so
# pricing lives in one place.
GPT_IMAGE_TOKEN_PRICING: dict[str, dict[str, Any]] = {
    "gpt-image-1": {
        "text_input_per_1m_tokens": 5.0,
        "image_output_per_1m_tokens": 40.0,
        "tokens_per_image": 1750,
    },
    "gpt-image-1.5": {
        "text_input_per_1m_tokens": 4.0,
        "image_output_per_1m_tokens": 32.0,
        "tokens_per_image": 1750,
    },
    "gpt-image-2": {
        "text_input_per_1m_tokens": 5.0,
        "image_output_per_1m_tokens": 30.0,
        "tokens_per_image_by_quality": _GPT_IMAGE_2_TOKENS_BASE_1024,
    },
    # Both GPT Image 2.5 models bill at GPT Image 2's token rates (OpenAI model
    # pages, 2026-09-08). Output tokens come from _gpt_image_2_5_tokens_per_image.
    "gpt-image-2.5-flare": {
        "text_input_per_1m_tokens": 5.0,
        "image_output_per_1m_tokens": 30.0,
    },
    "gpt-image-2.5-sunburst": {
        "text_input_per_1m_tokens": 5.0,
        "image_output_per_1m_tokens": 30.0,
    },
}

GPT_IMAGE_2_5_MODELS: frozenset[str] = frozenset(
    {"gpt-image-2.5-flare", "gpt-image-2.5-sunburst"}
)

# Output-token grid for the GPT Image 2.5 models: patches along the image's
# long edge per quality tier. Copied from the GptImageTokenCalculator on
# OpenAI's image-generation guide (read 2026-09-27), which uses one table for
# both Flare and Sunburst. OpenAI notes the two models can consume different
# token counts at the same setting, so treat results as estimates.
_GPT_IMAGE_2_5_GRID_LONG_EDGE: dict[str, int] = {
    "low": 16,
    "medium": 24,
    "high": 48,
    "xhigh": 64,
    "max": 96,
}


def _parse_dimensions(size: str) -> tuple[int, int] | None:
    """Parse a 'WxH' size string into (W, H). None on failure or 'auto'."""
    if not isinstance(size, str):
        return None
    normalized = size.strip().lower()
    if normalized == "auto" or "x" not in normalized:
        return None
    try:
        w_str, h_str = normalized.split("x", 1)
        w, h = int(w_str), int(h_str)
    except (ValueError, TypeError):
        return None
    if w <= 0 or h <= 0:
        return None
    return w, h


def _parse_pixels(size: str) -> int | None:
    """Parse a 'WxH' size string and return W*H. None on failure or 'auto'."""
    dimensions = _parse_dimensions(size)
    if dimensions is None:
        return None
    return dimensions[0] * dimensions[1]


def _gpt_image_2_tokens_per_image(quality: str, size: str) -> int:
    """Approximate output tokens for a single gpt-image-2 image.

    Uses the quality-tier baseline at 1024x1024 with a sub-linear size
    multiplier: scales linearly *down* for smaller images but caps the
    upscale at ~1.3x. This matches the OpenAI calculator, which prices
    high-quality 4K at only ~28% more than high-quality 1024x1024 despite
    the 7.9x pixel ratio.

    Unknown quality falls back to the auto/medium tier; auto/unparseable
    size uses the 1024x1024 baseline unchanged.
    """
    base = _GPT_IMAGE_2_TOKENS_BASE_1024.get(
        (quality or "auto").lower(),
        _GPT_IMAGE_2_TOKENS_BASE_1024["auto"],
    )
    pixels = _parse_pixels(size)
    if pixels is None:
        return base
    ratio = pixels / (1024 * 1024)
    # Scale down linearly below 1024² (small sizes cost less); cap the
    # upscale at 1.3x to match the observed calculator flattening.
    multiplier = ratio if ratio < 1.0 else min(ratio, 1.3)
    return max(1, int(base * multiplier))


def _gpt_image_2_5_tokens_per_image(quality: str, size: str) -> int:
    """Output tokens for one gpt-image-2.5-flare / -sunburst image.

    Mirrors OpenAI's GptImageTokenCalculator: the long edge gets the quality
    tier's patch count, the short edge that count divided by the aspect
    ratio (rounded half-to-even), and tokens are
    ``ceil(patches * (2_000_000 + W*H) / 4_000_000)``.

    'auto' (or an unknown tier) is estimated as medium, and 'auto' or an
    unparseable size as 1024x1024, because the real figure depends on what
    the model picks.
    """
    grid = _GPT_IMAGE_2_5_GRID_LONG_EDGE.get(
        (quality or "auto").lower(), _GPT_IMAGE_2_5_GRID_LONG_EDGE["medium"]
    )
    width, height = _parse_dimensions(size) or (1024, 1024)
    long_edge, short_edge = max(width, height), min(width, height)
    scaled = grid / (long_edge / short_edge)
    floor = math.floor(scaled)
    short_grid = floor + floor % 2 if scaled - floor == 0.5 else round(scaled)
    patches = grid * short_grid
    return math.ceil(patches * (2_000_000 + width * height) / 4_000_000)


def gpt_image_tokens_per_image(model: str, quality: str, size: str) -> int:
    """Approximate output tokens per image for any gpt-image-* model."""
    if model in GPT_IMAGE_2_5_MODELS:
        return _gpt_image_2_5_tokens_per_image(quality, size)
    if model == "gpt-image-2":
        return _gpt_image_2_tokens_per_image(quality, size)
    return GPT_IMAGE_TOKEN_PRICING[model]["tokens_per_image"]


class OpenAIProvider(LLMProvider):
    """OpenAI provider for image generation using gpt-image and DALL-E models."""

    # Shared capabilities for gpt-image-1 family models
    _GPT_IMAGE_CAPABILITY = dict(
        supported_sizes=["auto", "1024x1024", "1536x1024", "1024x1536"],
        supported_qualities=["auto", "high", "medium", "low"],
        supported_formats=["png", "jpeg", "webp"],
        max_images_per_request=1,
        supports_style=False,
        supports_background=True,
        supports_compression=True,
        custom_parameters={
            "moderation": ["auto", "low"],
            "background": ["auto", "transparent", "opaque"],
        },
    )

    # Flexible WxH rules shared by gpt-image-2 and the gpt-image-2.5 models
    # (OpenAI Image API reference, size parameter).
    _FLEXIBLE_SIZE_CONSTRAINTS = {
        "multiple_of": 16,
        "max_edge": 3840,
        "max_aspect_ratio": 3.0,
        "min_pixels": 655_360,
        "max_pixels": 8_294_400,
    }

    # gpt-image-2 adds flexible sizing up to 4K
    _GPT_IMAGE_2_CAPABILITY = dict(
        supported_sizes=[
            "auto",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "3840x2160",
        ],
        supported_qualities=["auto", "high", "medium", "low"],
        supported_formats=["png", "jpeg", "webp"],
        max_images_per_request=1,
        supports_style=False,
        supports_background=True,
        supports_compression=True,
        supports_custom_sizes=True,
        size_constraints=dict(_FLEXIBLE_SIZE_CONSTRAINTS),
        custom_parameters={
            "moderation": ["auto", "low"],
            # gpt-image-2 does not natively support transparent backgrounds
            # (OpenAI docs), but callers may pass 'transparent' — it is
            # transparently downgraded to 'auto' on every outbound path via
            # OpenAIProvider._resolve_background so the public interface
            # stays uniform across models.
            "background": ["auto", "transparent", "opaque"],
        },
    )

    # gpt-image-2.5-flare / -sunburst (released 2026-09-08): gpt-image-2's
    # flexible sizing, native transparent backgrounds, and two extra quality
    # tiers above 'high'. Outputs above 3,686,400 pixels (2560x1440) are
    # experimental per OpenAI.
    _GPT_IMAGE_2_5_CAPABILITY = dict(
        supported_sizes=[
            "auto",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "3840x2160",
        ],
        supported_qualities=["auto", "max", "xhigh", "high", "medium", "low"],
        supported_formats=["png", "jpeg", "webp"],
        max_images_per_request=1,
        supports_style=False,
        supports_background=True,
        supports_compression=True,
        supports_custom_sizes=True,
        size_constraints=dict(_FLEXIBLE_SIZE_CONSTRAINTS),
        custom_parameters={
            "moderation": ["auto", "low"],
            "background": ["auto", "transparent", "opaque"],
        },
    )

    # Supported models and their capabilities
    SUPPORTED_MODELS = {
        "gpt-image-2.5-sunburst": ModelCapability(
            model_id="gpt-image-2.5-sunburst",
            **_GPT_IMAGE_2_5_CAPABILITY,
        ),
        "gpt-image-2.5-flare": ModelCapability(
            model_id="gpt-image-2.5-flare",
            **_GPT_IMAGE_2_5_CAPABILITY,
        ),
        "gpt-image-1": ModelCapability(
            model_id="gpt-image-1",
            **_GPT_IMAGE_CAPABILITY,
        ),
        "gpt-image-1.5": ModelCapability(
            model_id="gpt-image-1.5",
            **_GPT_IMAGE_CAPABILITY,
        ),
        "gpt-image-2": ModelCapability(
            model_id="gpt-image-2",
            **_GPT_IMAGE_2_CAPABILITY,
        ),
        "dall-e-3": ModelCapability(
            model_id="dall-e-3",
            supported_sizes=["1024x1024", "1792x1024", "1024x1792"],
            supported_qualities=["auto", "high"],
            supported_formats=["png"],
            max_images_per_request=1,
            supports_style=True,
            supports_background=False,
            supports_compression=False,
            custom_parameters={
                "style": ["vivid", "natural"],
            },
        ),
        "dall-e-2": ModelCapability(
            model_id="dall-e-2",
            supported_sizes=["256x256", "512x512", "1024x1024"],
            supported_qualities=["auto"],
            supported_formats=["png"],
            max_images_per_request=10,
            supports_style=False,
            supports_background=False,
            supports_compression=False,
            custom_parameters={},
        ),
    }

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            organization=config.organization,
            base_url=config.base_url or "https://api.openai.com/v1",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def get_supported_models(self) -> set[str]:
        """Return set of supported OpenAI model IDs."""
        return set(self.SUPPORTED_MODELS.keys())

    @staticmethod
    def _validate_custom_size(size: str, constraints: dict[str, Any]) -> bool:
        """Check a WxH string against a model's custom-size constraints."""
        try:
            w_str, h_str = size.lower().split("x")
            w, h = int(w_str), int(h_str)
        except (ValueError, AttributeError):
            return False
        if w <= 0 or h <= 0:
            return False
        multiple_of = constraints["multiple_of"]
        if w % multiple_of or h % multiple_of:
            return False
        if max(w, h) > constraints["max_edge"]:
            return False
        if max(w, h) / min(w, h) > constraints["max_aspect_ratio"]:
            return False
        pixels = w * h
        return constraints["min_pixels"] <= pixels <= constraints["max_pixels"]

    @classmethod
    def _resolve_size(
        cls, size: str, capabilities: ModelCapability, model: str
    ) -> str:
        """Return a size string the API will accept, falling back to auto.

        Normalizes whitespace and case up-front so cache keys, metadata, and
        outbound requests all see the same canonical value regardless of
        caller input (e.g. ``"  1600X896 "`` → ``"1600x896"``).
        """
        normalized = size.strip().lower() if isinstance(size, str) else size
        if normalized in capabilities.supported_sizes:
            return normalized
        if (
            capabilities.supports_custom_sizes
            and capabilities.size_constraints
            and cls._validate_custom_size(normalized, capabilities.size_constraints)
        ):
            return normalized
        fallback = (
            "auto" if "auto" in capabilities.supported_sizes
            else capabilities.supported_sizes[0]
        )
        logger.warning(
            "Size %r not supported by %s, using %s", size, model, fallback
        )
        return fallback

    def get_model_capabilities(self, model_id: str) -> ModelCapability | None:
        """Get capabilities for a specific OpenAI model."""
        return self.SUPPORTED_MODELS.get(model_id)

    @classmethod
    def _resolve_background(cls, background: str, model: str) -> str:
        """Downgrade unsupported background values per model.

        gpt-image-2 transparency is only a preview per OpenAI docs, so
        callers that request it get 'auto' instead with a warning. The
        gpt-image-2.5 models and gpt-image-1 / 1.5 support transparent
        natively and pass it through.
        """
        if (
            model == "gpt-image-2"
            and isinstance(background, str)
            and background.strip().lower() == "transparent"
        ):
            logger.warning(
                "background='transparent' is not supported by gpt-image-2; "
                "using 'auto' instead"
            )
            return "auto"
        return background

    # Quality tiers that only the gpt-image-2.5 models offer, and the
    # nearest tier to use on models that lack them.
    _QUALITY_STEP_DOWN: dict[str, str] = {"max": "high", "xhigh": "high"}

    @classmethod
    def _resolve_quality(
        cls, quality: Any, capabilities: ModelCapability, model: str
    ) -> Any:
        """Return a quality value the model accepts.

        'xhigh' / 'max' on a model without them step down to 'high' where
        the model has it; any other unsupported value falls back to 'auto'
        (or the model's first quality). Enum inputs are unwrapped.
        """
        value = quality.value if hasattr(quality, "value") else quality
        normalized = value.strip().lower() if isinstance(value, str) else value
        if normalized in capabilities.supported_qualities:
            return normalized
        step_down = cls._QUALITY_STEP_DOWN.get(normalized)
        if step_down in capabilities.supported_qualities:
            fallback = step_down
        elif "auto" in capabilities.supported_qualities:
            fallback = "auto"
        else:
            fallback = capabilities.supported_qualities[0]
        logger.warning(
            "Quality %r not supported by %s, using %s", quality, model, fallback
        )
        return fallback

    def validate_model_params(
        self, model: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize params, resolving size and quality so metadata and
        cache reflect what will actually be sent to the API (values the
        model can't take are downgraded to a supported fallback here, not
        silently rewritten later)."""
        # Normalize size BEFORE delegating to super() — for models without
        # supports_custom_sizes, the base validator does an exact-match check
        # against supported_sizes and would downgrade whitespace/case variants
        # (e.g. "  1024X1024  ") to the default.
        capabilities = self.get_model_capabilities(model)
        normalized_size: str | None = None
        params = dict(params)
        if capabilities and "quality" in params:
            # Resolve quality here too: the base validator would send an
            # unsupported 'max' / 'xhigh' to 'auto' rather than to 'high'.
            params["quality"] = self._resolve_quality(
                params["quality"], capabilities, model
            )
        if capabilities and "size" in params:
            raw = params["size"]
            # Unwrap enums (ImageSize etc.) — str(enum) gives "Class.MEMBER"
            # for non-StrEnum classes, not the underlying value.
            normalized_size = raw.value if hasattr(raw, "value") else raw
            if isinstance(normalized_size, str):
                normalized_size = normalized_size.strip().lower()
            params["size"] = normalized_size

        params = super().validate_model_params(model, params)

        if capabilities and normalized_size is not None:
            params["size"] = self._resolve_size(
                normalized_size, capabilities, model
            )
        return params

    async def generate_image(
        self,
        model: str,
        prompt: str,
        quality: str = "auto",
        size: str = "auto",
        style: str = "vivid",
        moderation: str = "auto",
        output_format: str = "png",
        compression: int = 100,
        background: str = "auto",
        n: int = 1,
        **kwargs,
    ) -> ImageResponse:
        """Generate image using OpenAI's Images API."""

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Model '{model}' is not supported by OpenAI provider",
                provider_name=self.name,
                error_code="UNSUPPORTED_MODEL",
            )

        capabilities = self.SUPPORTED_MODELS[model]

        # Build request parameters
        request_params = {
            "model": model,
            "prompt": prompt,
            "n": min(n, capabilities.max_images_per_request),
        }

        # Map quality parameter based on model
        if model == "dall-e-3":
            request_params["quality"] = (
                "hd" if quality == "high" else "standard"
            )
        elif not model.startswith("dall-e"):
            # gpt-image models take the quality value directly, once it is
            # one the model supports (xhigh/max exist only on gpt-image-2.5)
            request_params["quality"] = self._resolve_quality(
                quality, capabilities, model
            )
            request_params["moderation"] = moderation

        # Add size parameter
        request_params["size"] = self._resolve_size(size, capabilities, model)

        # Add style parameter if supported
        if capabilities.supports_style:
            request_params["style"] = style

        # Add gpt-image specific parameters (not supported by DALL-E)
        if model.startswith("gpt-image-"):
            request_params["output_format"] = output_format
            request_params["background"] = self._resolve_background(
                background, model
            )
            if output_format in ["jpeg", "webp"] and compression < 100:
                request_params["output_compression"] = compression

        try:
            self._logger.info(f"Generating image with OpenAI model {model}")
            self._logger.debug(f"Request parameters: {request_params}")

            response = await self.client.images.generate(**request_params)

            # Process response
            if hasattr(response.data[0], "b64_json") and response.data[0].b64_json:
                # Base64 response (gpt-image-1)
                image_bytes = base64.b64decode(response.data[0].b64_json)
            elif hasattr(response.data[0], "url") and response.data[0].url:
                # URL response (DALL-E models)
                image_bytes = await self._download_image(response.data[0].url)
            else:
                raise ProviderError(
                    "OpenAI response contains neither base64 data nor URL",
                    provider_name=self.name,
                    error_code="INVALID_RESPONSE",
                )

            # Build metadata
            metadata = {
                "model": model,
                "prompt": prompt,
                "size": request_params["size"],
                "quality": request_params.get("quality", "auto"),
                "style": request_params.get("style", "vivid"),
                "output_format": request_params.get("output_format", "png"),
                "provider": self.name,
                "created_at": getattr(response, "created", None),
            }

            # Add usage information if available
            if hasattr(response, "usage") and response.usage:
                metadata["usage"] = {
                    "total_tokens": response.usage.total_tokens,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }

            return ImageResponse(
                image_data=image_bytes,
                metadata=metadata,
                provider_response=(
                    response.model_dump() if hasattr(response, "model_dump") else None
                ),
            )

        except Exception as e:
            self._logger.error(f"Error generating image with OpenAI: {e}")
            raise ProviderError(
                f"OpenAI image generation failed: {str(e)}",
                provider_name=self.name,
                error_code="GENERATION_FAILED",
            )

    async def edit_image(
        self,
        model: str,
        image_data: str | bytes,
        prompt: str,
        mask_data: str | bytes | None = None,
        quality: str = "auto",
        size: str = "1536x1024",
        output_format: str = "png",
        compression: int = 100,
        background: str = "auto",
        n: int = 1,
        **kwargs,
    ) -> ImageResponse:
        """Edit image using OpenAI's Images API."""

        # Validate model
        if model not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Model '{model}' is not supported by OpenAI provider",
                provider_name=self.name,
                error_code="UNSUPPORTED_MODEL",
            )

        capabilities = self.SUPPORTED_MODELS[model]

        from ..utils import prepare_image_upload

        # Decode inputs and build SDK upload tuples.
        _, _, image_file = prepare_image_upload(image_data)

        mask_file = None
        if mask_data:
            _, _, mask_file = prepare_image_upload(mask_data)

        # Build request parameters
        request_params = {
            "model": model,
            "image": image_file,
            "prompt": prompt,
            "n": min(n, capabilities.max_images_per_request),
        }

        if mask_file:
            request_params["mask"] = mask_file

        # Add size parameter
        request_params["size"] = self._resolve_size(size, capabilities, model)

        # Add gpt-image family specific parameters
        if model.startswith("gpt-image-"):
            request_params["quality"] = self._resolve_quality(
                quality, capabilities, model
            )
            request_params["output_format"] = output_format
            request_params["background"] = self._resolve_background(
                background, model
            )
            if output_format in ["jpeg", "webp"] and compression < 100:
                request_params["output_compression"] = compression

        try:
            self._logger.info(f"Editing image with OpenAI model {model}")
            self._logger.debug(f"Request parameters: {list(request_params.keys())}")

            response = await self.client.images.edit(**request_params)

            # Process response (similar to generate_image)
            if hasattr(response.data[0], "b64_json") and response.data[0].b64_json:
                image_bytes = base64.b64decode(response.data[0].b64_json)
            elif hasattr(response.data[0], "url") and response.data[0].url:
                image_bytes = await self._download_image(response.data[0].url)
            else:
                raise ProviderError(
                    "OpenAI response contains neither base64 data nor URL",
                    provider_name=self.name,
                    error_code="INVALID_RESPONSE",
                )

            metadata = {
                "model": model,
                "prompt": prompt,
                "size": request_params["size"],
                "output_format": request_params.get("output_format", "png"),
                "provider": self.name,
                "operation": "edit",
                "created_at": getattr(response, "created", None),
            }

            return ImageResponse(
                image_data=image_bytes,
                metadata=metadata,
                provider_response=(
                    response.model_dump() if hasattr(response, "model_dump") else None
                ),
            )

        except Exception as e:
            self._logger.error(f"Error editing image with OpenAI: {e}")
            raise ProviderError(
                f"OpenAI image editing failed: {str(e)}",
                provider_name=self.name,
                error_code="EDITING_FAILED",
            )

    async def check_health(self) -> dict[str, Any]:
        """Ping OpenAI API using the free GET /v1/models endpoint."""
        try:
            models = await self.client.models.list()
            supported = set(self.SUPPORTED_MODELS.keys())
            image_models = [
                m.id for m in models.data if m.id in supported
            ]
            if not image_models:
                return {
                    "status": "unhealthy",
                    "error": (
                        "No supported image models available "
                        "for current credentials"
                    ),
                    "models_available": [],
                }
            return {
                "status": "healthy",
                "models_available": image_models,
            }
        except Exception as e:
            self._logger.warning(f"OpenAI health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def _download_image(self, image_url: str) -> bytes:
        """Download image from URL (for DALL-E models that return URLs)."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                return response.content
        except Exception as e:
            raise ProviderError(
                f"Failed to download image from URL: {str(e)}",
                provider_name=self.name,
                error_code="DOWNLOAD_FAILED",
            )

    def estimate_cost(
        self,
        model: str,
        prompt: str,
        image_count: int = 1,
        quality: str = "auto",
        size: str = "1024x1024",
    ) -> dict[str, Any]:
        """Estimate cost for OpenAI image generation.

        For gpt-image-2 and the gpt-image-2.5 models, output tokens vary
        substantially by quality and size, so the estimate is quality- and
        size-aware (see gpt_image_tokens_per_image). Results are marked with
        ``estimate_accuracy: "rough"`` — callers should not rely on these
        as authoritative billing figures.
        """

        fixed_pricing = {
            "dall-e-3": {"cost_per_image": 0.04},
            "dall-e-2": {"cost_per_image": 0.02},
        }

        if model in GPT_IMAGE_TOKEN_PRICING:
            model_pricing = GPT_IMAGE_TOKEN_PRICING[model]
            text_tokens = len(prompt.split()) * 1.3  # Rough approximation
            text_cost = (text_tokens / 1_000_000) * model_pricing[
                "text_input_per_1m_tokens"
            ]

            tokens_per_image = gpt_image_tokens_per_image(model, quality, size)

            image_tokens_total = tokens_per_image * image_count
            image_cost = (
                image_tokens_total / 1_000_000
            ) * model_pricing["image_output_per_1m_tokens"]
            total_cost = text_cost + image_cost

            return {
                "provider": self.name,
                "model": model,
                "estimated_cost_usd": round(total_cost, 4),
                "currency": "USD",
                "estimate_accuracy": "rough",
                "breakdown": {
                    "text_input_cost": round(text_cost, 4),
                    "image_output_cost": round(image_cost, 4),
                    "text_tokens": int(text_tokens),
                    "image_tokens": image_tokens_total,
                    "tokens_per_image": tokens_per_image,
                    "quality": quality,
                    "size": size,
                    "total_images": image_count,
                },
            }
        elif model in fixed_pricing:
            per_image = fixed_pricing[model]["cost_per_image"]
            total_cost = per_image * image_count

            return {
                "provider": self.name,
                "model": model,
                "estimated_cost_usd": round(total_cost, 4),
                "currency": "USD",
                "breakdown": {
                    "per_image": per_image,
                    "total_images": image_count,
                    "base_cost": total_cost,
                },
            }
        else:
            return super().estimate_cost(model, prompt, image_count)
