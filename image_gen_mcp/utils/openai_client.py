"""OpenAI API client manager with retry logic and error handling."""

import logging
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.images_response import ImagesResponse

from ..config.settings import OpenAISettings
from ..providers.openai import (
    GPT_IMAGE_TOKEN_PRICING,
    OpenAIProvider,
    gpt_image_tokens_per_image,
)

logger = logging.getLogger(__name__)



class OpenAIClientManager:
    """Manages OpenAI API client with retry logic and error handling."""

    def __init__(self, settings: OpenAISettings):
        self.settings = settings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        return AsyncOpenAI(
            api_key=self.settings.api_key,
            organization=self.settings.organization,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout,
            max_retries=self.settings.max_retries,
        )

    async def generate_image(
        self,
        prompt: str,
        model: str = "gpt-image-2.5-sunburst",
        quality: str = "auto",
        size: str = "1536x1024",
        style: str = "vivid",
        moderation: str = "auto",
        output_format: str = "png",
        compression: int = 100,
        background: str = "auto",
        n: int = 1,
    ) -> ImagesResponse:
        """Generate an image using OpenAI's Images API."""

        # xhigh/max exist only on the gpt-image-2.5 models; step them down
        # for older models the same way OpenAIProvider does.
        capabilities = OpenAIProvider.SUPPORTED_MODELS.get(model)
        if capabilities is not None:
            quality = OpenAIProvider._resolve_quality(quality, capabilities, model)

        request_params = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "quality": quality,
            "size": size,
            "moderation": moderation,
        }

        # Add gpt-image-1 family specific parameters
        if model.startswith("gpt-image-"):
            request_params.update(
                {
                    "style": style,
                    "output_format": output_format,
                    # Downgrade model-incompatible background values
                    # (e.g. transparent on gpt-image-2) via the same
                    # resolver used by OpenAIProvider.
                    "background": OpenAIProvider._resolve_background(
                        background, model
                    ),
                }
            )

            # Add compression for JPEG/WebP
            if output_format in ["jpeg", "webp"] and compression < 100:
                request_params["output_compression"] = compression

        try:
            logger.info(f"Generating image with prompt: {prompt[:100]}...")
            logger.debug(f"Request parameters: {list(request_params.keys())}")
            response = await self.client.images.generate(**request_params)

            logger.info(f"Successfully generated {len(response.data)} image(s)")
            return response

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise

    async def edit_image(
        self,
        image_data: str | bytes,
        prompt: str,
        mask_data: str | bytes | None = None,
        model: str = "gpt-image-2.5-sunburst",
        quality: str = "auto",
        size: str = "1536x1024",
        output_format: str = "png",
        compression: int = 100,
        background: str = "auto",
        n: int = 1,
        additional_images: list[str | bytes] | None = None,
        moderation: str = "auto",
    ) -> ImagesResponse:
        """Edit an image using OpenAI's Images API.

        ``additional_images`` are sent after ``image_data``, in order, as
        further input images (the API takes up to 16 in total).
        """

        # Coerce enum inputs (ImageQuality, ImageSize, ImageFormat, etc.) to
        # their string values so the OpenAI SDK receives valid literals
        # ("high", "png", ...) instead of enum reprs ("ImageQuality.HIGH").
        # The generation path does this inside OpenAIProvider; the edit path
        # goes through this client, so coerce here.
        quality = getattr(quality, "value", quality)
        size = getattr(size, "value", size)
        output_format = getattr(output_format, "value", output_format)
        background = getattr(background, "value", background)
        model = getattr(model, "value", model)

        from . import prepare_image_upload

        # Decode inputs and build SDK upload tuples.
        _, _, image_file = prepare_image_upload(image_data)
        extra_files = [
            prepare_image_upload(extra)[2] for extra in (additional_images or [])
        ]

        mask_file = None
        if mask_data:
            _, _, mask_file = prepare_image_upload(mask_data)

        # Normalize size against the model's capabilities so invalid custom
        # sizes (e.g. "9999x9999") fall back locally instead of failing at
        # the remote API.
        capabilities = OpenAIProvider.SUPPORTED_MODELS.get(model)
        if capabilities is not None:
            size = OpenAIProvider._resolve_size(size, capabilities, model)
            quality = OpenAIProvider._resolve_quality(quality, capabilities, model)

        request_params = {
            "model": model,
            # Several inputs go as a list, the primary first; one stays a tuple.
            "image": [image_file, *extra_files] if extra_files else image_file,
            "prompt": prompt,
            "n": n,
            "size": size,
        }

        if mask_file:
            request_params["mask"] = mask_file

        # The SDK's images.edit has no moderation argument, so a non-default
        # level travels in the request body; "auto" (the API default) sends none.
        moderation = getattr(moderation, "value", moderation)
        if moderation and moderation != "auto":
            request_params["extra_body"] = {"moderation": moderation}

        # Add gpt-image-1 family specific parameters
        if model.startswith("gpt-image-"):
            request_params["quality"] = quality
            request_params["output_format"] = output_format
            # Same background downgrade as generate_image / OpenAIProvider
            # so transparent on gpt-image-2 never hits the real API.
            request_params["background"] = OpenAIProvider._resolve_background(
                background, model
            )
            if output_format in ["jpeg", "webp"] and compression < 100:
                request_params["output_compression"] = compression

        try:
            logger.info(f"Editing image with prompt: {prompt[:100]}...")
            logger.debug(f"Request parameters: {list(request_params.keys())}")
            logger.debug("API client configured for image editing")
            response = await self.client.images.edit(**request_params)

            logger.info(
                f"Successfully edited image, generated {len(response.data)} result(s)"
            )
            return response

        except Exception as e:
            logger.error(f"Error editing image: {e}")
            raise

    def estimate_cost(
        self,
        prompt: str,
        image_count: int = 1,
        model: str = "gpt-image-2.5-sunburst",
        quality: str = "auto",
        size: str = "1024x1024",
    ) -> dict[str, Any]:
        """Estimate the cost of image generation.

        For gpt-image-2 and the gpt-image-2.5 models the per-image token
        count depends on quality and size (see gpt_image_tokens_per_image).
        Results are rough — use OpenAIProvider.estimate_cost on the active
        provider instance for the authoritative per-model breakdown.
        """

        pricing = GPT_IMAGE_TOKEN_PRICING.get(model)
        if pricing is None:
            # Non-gpt-image models (e.g. dall-e-*) use per-image pricing that
            # this helper doesn't know about. Return a zero estimate rather
            # than silently reporting gpt-image-2 rates, and log so callers
            # notice. For accurate per-model cost data, use
            # OpenAIProvider.estimate_cost.
            logger.warning(
                "estimate_cost called with unsupported model %r; "
                "returning zero cost. Use OpenAIProvider.estimate_cost for "
                "non-gpt-image models.",
                model,
            )
            return {
                "estimated_cost_usd": 0.0,
                "text_input_tokens": 0,
                "image_output_tokens": 0,
                "breakdown": {
                    "text_input_cost": 0.0,
                    "image_output_cost": 0.0,
                },
            }
        quality = getattr(quality, "value", quality)
        size = getattr(size, "value", size)
        tokens_per_image = gpt_image_tokens_per_image(model, quality, size)

        # Rough token estimation (actual tokenization may vary)
        text_tokens = len(prompt.split()) * 1.3
        text_input_cost = (
            text_tokens / 1_000_000
        ) * pricing["text_input_per_1m_tokens"]
        image_output_cost = (
            image_count * tokens_per_image / 1_000_000
        ) * pricing["image_output_per_1m_tokens"]
        total_cost = text_input_cost + image_output_cost

        return {
            "estimated_cost_usd": round(total_cost, 4),
            "text_input_tokens": int(text_tokens),
            "image_output_tokens": int(tokens_per_image * image_count),
            "breakdown": {
                "text_input_cost": round(text_input_cost, 4),
                "image_output_cost": round(image_output_cost, 4),
            },
        }

    async def download_image(self, image_url: str) -> bytes:
        """Download image from URL (for dall-e models that return URLs)."""
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            return response.content
