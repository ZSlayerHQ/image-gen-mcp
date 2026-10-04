"""Image editing tool implementation."""

import base64
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from ..config.settings import Settings
from ..providers.openai import OpenAIProvider
from ..storage.manager import ImageStorageManager
from ..utils.cache import CacheManager
from ..utils.path_utils import build_image_url_path
from .batch import (
    BATCH_CAPABLE_MODELS,
    REFERENCE_COST_NOTE,
    PartialSaveError,
    partial_save_message,
    per_image_cost_info,
)

logger = logging.getLogger(__name__)


class ImageEditingTool:
    """Tool for editing images using multiple LLM providers."""

    def __init__(
        self,
        storage_manager: ImageStorageManager,
        cache_manager: CacheManager,
        settings: Settings,
        openai_client=None,
    ):
        """
        Args:
            storage_manager: ImageStorageManager instance.
            cache_manager: CacheManager instance.
            settings: Settings instance (must have .providers, .images, etc.).
            openai_client: Optional OpenAI client manager.
        """
        self.settings = settings
        self._validate_openai_settings()
        self.storage_manager = storage_manager
        self.cache_manager = cache_manager
        self.openai_client = openai_client

    def _validate_openai_settings(self):
        """Ensure OpenAI provider settings are present (required for image editing)."""
        self.validate_openai_provider_settings(self.settings)

    @staticmethod
    def validate_openai_provider_settings(settings):
        """Validate that the OpenAI provider settings are present and correct."""
        if not hasattr(settings, 'providers') or settings.providers is None:
            raise ValueError(
                "Settings must have a 'providers' attribute for image editing "
                "functionality."
            )
        if not hasattr(settings.providers, 'openai') or not settings.providers.openai:
            raise ValueError(
                "OpenAI provider settings are required for image editing "
                "functionality. Please configure the 'providers.openai' section "
                "in your settings with api_key, api_base, and other required "
                "parameters."
            )

    @staticmethod
    def edit_capable_models() -> list[str]:
        """OpenAI models this tool can edit with (the gpt-image-* family)."""
        return [
            model_id
            for model_id in OpenAIProvider.SUPPORTED_MODELS
            if model_id.startswith("gpt-image-")
        ]

    def _get_transport_type(self) -> str:
        """Detect the current transport type from environment or default to stdio."""
        import sys

        # Check if we're running with HTTP transport based on command line args
        if hasattr(sys, "argv"):
            for i, arg in enumerate(sys.argv):
                if arg == "--transport" and i + 1 < len(sys.argv):
                    return sys.argv[i + 1]
        # Default to stdio for Claude Desktop integration
        return "stdio"

    def _build_image_url(self, image_id: str, file_format: str = "png") -> str:
        """
        Build image URL using base_host setting, server host, or file path
        based on transport.
        """
        transport_type = self._get_transport_type()

        if self.settings.images.base_host:
            # Use configured host base (e.g., nginx/CDN URL) with full path
            url_path = build_image_url_path(image_id, file_format)
            return f"{self.settings.images.base_host.rstrip('/')}/{url_path}"
        elif transport_type in ["streamable-http", "sse"]:
            # Use MCP server host with HTTP endpoint for HTTP transports
            return f"http://{self.settings.server.host}:{self.settings.server.port}/images/{image_id}"
        else:
            # For stdio transport, return file path that Claude Desktop can access
            from pathlib import Path

            from ..utils.path_utils import build_image_storage_path

            image_path = build_image_storage_path(
                Path(self.settings.storage.base_path), image_id, file_format
            )
            return f"file://{image_path.absolute()}"

    async def edit(
        self,
        image_data: str,
        prompt: str,
        mask_data: str | None = None,
        size: str = "1536x1024",
        quality: str = "auto",
        output_format: str = "png",
        compression: int = 100,
        background: str = "auto",
        model: str | None = None,
        additional_images: list[str] | None = None,
        moderation: str = "auto",
        n: int = 1,
    ) -> dict[str, Any]:
        """Edit an existing image with text instructions.

        ``model`` picks the OpenAI GPT Image model for this edit; when it is
        omitted the configured default model is used. ``additional_images``
        (up to 15) are further input images sent after ``image_data``, in
        order. ``n`` (1-10) asks for several edited images in one request.
        """

        # Resolve and check the model before anything else: only the OpenAI
        # gpt-image-* models can edit.
        model = model or self.settings.images.default_model
        edit_models = self.edit_capable_models()
        if model not in edit_models:
            raise ValueError(
                f"Model {model!r} cannot edit images. "
                f"Use one of: {', '.join(edit_models)}"
            )

        # Apply defaults from settings
        quality = quality or self.settings.images.default_quality
        size = size or self.settings.images.default_size
        output_format = output_format or self.settings.images.default_output_format
        compression = (
            compression
            if compression is not None
            else self.settings.images.default_compression
        )

        # Generate task ID for tracking
        task_id = str(uuid.uuid4())

        # Normalize size and quality against the model's capabilities so the
        # cache key and persisted metadata reflect what will actually be sent
        # to the API (invalid custom sizes fall back to the supported default;
        # xhigh/max step down to high on models without them).
        capabilities = OpenAIProvider.SUPPORTED_MODELS[model]
        size = OpenAIProvider._resolve_size(size, capabilities, model)
        quality = OpenAIProvider._resolve_quality(quality, capabilities, model)

        # Validate the batch and reference options before the cache or the API.
        moderation = getattr(moderation, "value", moderation) or "auto"
        if moderation not in ("auto", "low"):
            raise ValueError("moderation must be 'auto' or 'low'")
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 10:
            raise ValueError("n must be an integer from 1 to 10")
        if n > 1 and model not in BATCH_CAPABLE_MODELS:
            raise ValueError(
                "Several images per request (n > 1) are supported only for "
                f"{', '.join(sorted(BATCH_CAPABLE_MODELS))}, not '{model}'."
            )
        if n > capabilities.max_images_per_request:
            raise ValueError(
                f"Model '{model}' returns at most "
                f"{capabilities.max_images_per_request} image(s) per request, "
                f"so n={n} is not supported for it."
            )
        extras = list(additional_images or [])
        if len(extras) > 15:
            raise ValueError(
                "additional_images takes at most 15 images (16 with image_data)"
            )

        # Check cache first (using hash of image data)
        cache_params = {
            "image_data": image_data,
            "prompt": prompt,
            "mask_data": mask_data,
            "quality": quality,
            "size": size,
            "output_format": output_format,
            "compression": compression,
            "background": background,
            "model": model,
        }
        # New options enter the key only when used, so existing keys are
        # unchanged. References are keyed by ordered short hashes, not raw data.
        if extras:
            cache_params["additional_image_hashes"] = [
                hashlib.sha256(
                    extra.encode() if isinstance(extra, str) else extra
                ).hexdigest()[:16]
                for extra in extras
            ]
        if moderation != "auto":
            cache_params["moderation"] = moderation
        if n != 1:
            cache_params["n"] = n

        cached_result = await self.cache_manager.get_image_edit(**cache_params)
        if cached_result:
            logger.info(f"Returning cached edit result for prompt: {prompt[:50]}...")
            return cached_result

        try:
            # Validate image data format
            if image_data.startswith("data:"):
                # Extract base64 data from data URL
                image_data = image_data.split(",", 1)[1]

            # Edit image using OpenAI API
            logger.info(f"Editing image for task {task_id}")
            response = await self.openai_client.edit_image(
                image_data=image_data,
                prompt=prompt,
                mask_data=mask_data,
                model=model,
                quality=quality,
                size=size,
                output_format=output_format,
                compression=compression,
                background=background,
                n=n,
                additional_images=extras or None,
                moderation=moderation,
            )

            # Decode every returned image, in order; the first is the primary
            outputs = list(response.data or [])
            if not outputs:
                raise RuntimeError("OpenAI response contains no images")
            decoded = [base64.b64decode(item.b64_json) for item in outputs]
            image_bytes = decoded[0]
            batch = n > 1 or len(decoded) > 1

            # Estimate cost (quality/size-aware for gpt-image-2 and 2.5),
            # counting every image returned
            cost_info = self.openai_client.estimate_cost(
                prompt,
                len(decoded),
                model=model,
                quality=quality,
                size=size,
            )

            # Add actual usage if available
            if hasattr(response, "usage") and response.usage:
                cost_info.update(
                    {
                        "actual_usage": {
                            "total_tokens": response.usage.total_tokens,
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        }
                    }
                )

            # Prepare metadata
            metadata = {
                "task_id": task_id,
                "operation": "edit",
                "prompt": prompt,
                "has_mask": mask_data is not None,
                "parameters": {
                    "model": model,
                    "quality": quality,
                    "size": size,
                    "output_format": output_format,
                    "compression": compression,
                    "background": background,
                },
                "cost_info": cost_info,
                "api_response": {
                    "created": getattr(response, "created", None),
                    "size": getattr(response, "size", size),
                    "quality": getattr(response, "quality", quality),
                    "output_format": getattr(response, "output_format", output_format),
                    "background": getattr(response, "background", background),
                },
            }
            if extras:
                metadata["reference_image_count"] = len(extras)
            if moderation != "auto":
                metadata["parameters"]["moderation"] = moderation

            if batch:
                # Save every image of a batch in order, each storing its share
                # of the request's cost. A failed save never triggers another
                # call: the request was already billed.
                stored_cost = per_image_cost_info(
                    cost_info, len(decoded), REFERENCE_COST_NOTE if extras else None
                )
                saved: list[tuple[str, Path]] = []
                for index, data in enumerate(decoded):
                    try:
                        saved.append(
                            await self.storage_manager.save_image(
                                image_data=data,
                                metadata={
                                    **metadata,
                                    "cost_info": stored_cost,
                                    "batch_index": index,
                                },
                                file_format=output_format,
                            )
                        )
                    except Exception as e:
                        raise PartialSaveError(
                            partial_save_message(
                                "Image edit", index, len(decoded), saved, e
                            )
                        ) from e
                image_id, image_path = saved[0]
                additional_results = [
                    {
                        "image_id": extra_id,
                        "image_url": self._build_image_url(extra_id, output_format),
                        "resource_uri": f"generated-images://{extra_id}",
                        "local_path": str(extra_path),
                        "file_size_bytes": len(data),
                    }
                    for (extra_id, extra_path), data in zip(saved[1:], decoded[1:])
                ]
            else:
                if extras:
                    # The stored estimate keeps its caveat with the image.
                    metadata["cost_info"] = {**cost_info, "note": REFERENCE_COST_NOTE}
                # Save to local storage
                image_id, image_path = await self.storage_manager.save_image(
                    image_data=image_bytes,
                    metadata=metadata,
                    file_format=output_format,
                )
                additional_results = []

            # Build image URL instead of base64 data
            image_url = self._build_image_url(image_id, output_format)

            # Prepare result
            result = {
                "task_id": task_id,
                "image_id": image_id,
                "image_url": image_url,  # URL instead of base64 data
                "resource_uri": f"generated-images://{image_id}",
                "operation": "edit",
                "metadata": {
                    "model": model,
                    "size": size,
                    "quality": quality,
                    "output_format": output_format,
                    "background": background,
                    "prompt": prompt,
                    "has_mask": mask_data is not None,
                    "created_at": metadata.get("created_at"),
                    "cost_estimate": cost_info.get("estimated_cost_usd"),
                    "file_size_bytes": len(image_bytes),
                    "dimensions": size,
                    "format": output_format.upper(),
                    "local_path": str(image_path),
                },
            }
            if extras:
                result["metadata"]["reference_image_count"] = len(extras)
                result["metadata"]["cost_note"] = f"cost_estimate {REFERENCE_COST_NOTE}"
            if batch:
                result["metadata"]["images_requested"] = n
                result["metadata"]["images_returned"] = len(decoded)
                result["metadata"]["cost_estimate_scope"] = "request_total"
                result["additional_images"] = additional_results

            # Cache the result with URL
            await self.cache_manager.set_image_edit(result, **cache_params)

            logger.info(f"Successfully edited image {image_id} for task {task_id}")
            return result

        except PartialSaveError:
            logger.error(f"Partial save for edit task {task_id}")
            raise
        except Exception as e:
            logger.error(f"Error editing image for task {task_id}: {e}")
            raise RuntimeError(f"Image editing failed: {str(e)}") from e
