"""Helpers for requests that return several images (n > 1)."""

from pathlib import Path
from typing import Any

# Models that may be asked for several images in one request. The capability
# count alone is not enough: retired DALL-E entries also declare more than one.
BATCH_CAPABLE_MODELS = frozenset(
    {"gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare"}
)

REFERENCE_COST_NOTE = (
    "excludes the input tokens of the input images (the edited image and the references)"
)


class PartialSaveError(RuntimeError):
    """A billed request returned images, but not all of them could be saved."""


def partial_save_message(
    operation: str,
    failed_index: int,
    total: int,
    saved: list[tuple[str, Path]],
    error: Exception,
) -> str:
    """Describe a partial save so the caller can use what was saved.

    ``failed_index`` is zero-based. ``saved`` lists the (image_id, path) pairs
    already written, in output order.
    """
    listed = "; ".join(f"{image_id} at {path}" for image_id, path in saved) or "none"
    return (
        f"{operation} succeeded but saving image {failed_index + 1} of {total} "
        f"failed: {error}. Already saved: {listed}. Do not repeat the request to "
        "recover them: it would be billed again."
    )


def per_image_cost_info(
    cost_info: dict[str, Any], images: int, note: str | None = None
) -> dict[str, Any]:
    """Stored cost for one image of a batch: an equal share of the request.

    The history lists each stored image's ``estimated_cost_usd``; storing the
    whole request total on every image would count it once per image. The
    request's own estimate is kept whole under ``request``.
    """
    total = cost_info.get("estimated_cost_usd")
    share = (
        round(total / images, 6)
        if isinstance(total, (int, float)) and images
        else total
    )
    stored = {
        "scope": "per_image_share_of_request",
        "estimated_cost_usd": share,
        "request_images": images,
        "request": cost_info,
    }
    if note:
        stored["note"] = note
    return stored
