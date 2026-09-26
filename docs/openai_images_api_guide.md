# OpenAI Images API Guide

> **Note**: This server now supports multiple AI providers. For comprehensive documentation covering OpenAI, Gemini, and other providers, see the [Multi-Provider API Guide](multi_provider_api_guide.md).

This guide covers the OpenAI Images API integration used by the Image Gen MCP Server. Facts about the GPT Image 2.5 models were checked against OpenAI's documentation on 27 September 2026 (see [Sources](#sources)).

## Quick Reference

The Image Gen MCP Server supports the following OpenAI models:
- **gpt-image-2.5-sunburst** (default): GPT Image 2.5 base model, released 8 September 2026. Highest image quality and editing precision; the best choice here for exact lettering. Adds the `xhigh` and `max` quality tiers.
- **gpt-image-2.5-flare**: GPT Image 2.5 small model, released the same day. Fastest option for everyday, high-volume work, with image quality comparable to `gpt-image-2`. Same quality tiers, sizes and prices as Sunburst.
- **gpt-image-2**: Flexible sizing up to 4K (3840x2160), qualities up to `high`
- **gpt-image-1.5**: Fixed preset sizes
- **gpt-image-1**: Original gpt-image model
- **dall-e-3**, **dall-e-2**: OpenAI's changelog lists both as removed from the API on 12 May 2026. They are still registered in this server, but requests are not expected to work; use a GPT Image model instead.

### GPT Image 2.5 at a glance

| Setting | gpt-image-2.5-sunburst / gpt-image-2.5-flare |
|---------|-----------------------------------------------|
| Model ids | `gpt-image-2.5-sunburst`, `gpt-image-2.5-flare` (snapshots `…-2026-09-08`) |
| Endpoints | Image API generations and edits; Responses API image generation tool. Not Batch |
| Quality | `auto` (default), `low`, `medium`, `high`, `xhigh`, `max` |
| Size | `auto`, `1024x1024`, `1536x1024`, `1024x1536`, or any `WxH`: both edges multiples of 16, longest edge ≤ 3840, aspect ratio ≤ 3:1, 655,360–8,294,400 pixels. Common 2K/4K sizes: `2048x2048`, `2048x1152`, `3840x2160`, `2160x3840`. Outputs above 3,686,400 pixels (2560x1440) are experimental |
| Background | `auto`, `opaque`, `transparent` (native; use PNG or WebP with transparent) |
| Output format | `png` (default), `jpeg`, `webp`; `output_compression` 0–100 for JPEG/WebP |
| Moderation | `auto` (default) or `low` |
| Edits | Up to 16 input images; optional mask (applied to the first image, same format and size as the image, alpha channel, under 50 MB); mask editing is prompt-guided, not pixel-exact |
| Price | Text input $5, image input $8, image output $30 per 1M tokens (cached input $1.25 / $2 applies only to the Responses API image tool). Same rates as `gpt-image-2` |

## Create Image

**Endpoint**: `POST https://api.openai.com/v1/images/generations`

Creates an image given a prompt. [Learn more](https://developers.openai.com/api/docs/guides/image-generation).

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Text description of the desired image(s). Max 32,000 chars for `gpt-image-*` (this server's tools accept up to 4,000) |
| `model` | string | No | Model to use: `gpt-image-2.5-sunburst` (default), `gpt-image-2.5-flare`, `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1` (`dall-e-3` / `dall-e-2` were removed from the API on 12 May 2026) |
| `n` | integer | No | Number of images. This server always requests `n=1` |
| `size` | string | No | `gpt-image-1` / `gpt-image-1.5`: `1024x1024`, `1536x1024`, `1024x1536`, or `auto`. `gpt-image-2` and the `gpt-image-2.5` models: those presets plus any `WxH` within the limits in the table above (e.g. `2048x1152`, `3840x2160`) |
| `quality` | string | No | `gpt-image-*`: `auto`, `high`, `medium`, `low`. The `gpt-image-2.5` models also take `xhigh` and `max`; this server steps those down to `high` for models without them |
| `output_format` | string | No | Format for `gpt-image-*`: `png`, `jpeg`, or `webp` |
| `background` | string | No | `transparent`, `opaque`, or `auto`. Native on `gpt-image-2.5-*`, `gpt-image-1.5` and `gpt-image-1`. For `gpt-image-2`, OpenAI now lists transparency as a preview; this server still downgrades `transparent` to `auto` for that model |
| `moderation` | string | No | Moderation level for `gpt-image-*`: `auto` or `low` |

### Example Request

```bash
curl https://api.openai.com/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $PROVIDERS__OPENAI__API_KEY" \
  -d '{
    "model": "gpt-image-2.5-sunburst",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024",
    "quality": "high"
  }'
```

### Python Example

```python
import base64
from openai import OpenAI
client = OpenAI()

img = client.images.generate(
    model="gpt-image-2.5-sunburst",
    prompt="A cute baby sea otter",
    n=1,
    size="1024x1024",
    quality="high",
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

## Create Image Edit

**Endpoint**: `POST https://api.openai.com/v1/images/edits`

Creates an edited or extended image given source images and a prompt. Supports the `gpt-image-*` models; `gpt-image-2.5-sunburst` is OpenAI's pick where editing precision matters most.

### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file/array | Yes | Image(s) to edit. For `gpt-image-*`: up to 16 images, PNG/WEBP/JPG < 50MB |
| `prompt` | string | Yes | Edit instructions. Max 32,000 chars for `gpt-image-*` |
| `model` | string | No | `gpt-image-2.5-sunburst` (this server's default), `gpt-image-2.5-flare`, `gpt-image-2`, `gpt-image-1.5`, or `gpt-image-1` |
| `mask` | file | No | PNG mask with an alpha channel marking the areas to edit; applied to the first image |
| `quality`, `size`, `background`, `output_format` | string | No | Same values as generation |
| `n` | integer | No | Number of images to generate (1-10) |

This server's `edit_image` tool sends one image plus an optional mask.

### Example Request

```bash
curl -X POST "https://api.openai.com/v1/images/edits" \
  -H "Authorization: Bearer $PROVIDERS__OPENAI__API_KEY" \
  -F "model=gpt-image-2.5-sunburst" \
  -F "image[]=@body-lotion.png" \
  -F "image[]=@bath-bomb.png" \
  -F 'prompt=Create a lovely gift basket with these items'
```

## Using with MCP Server

### Configuration

```bash
# Environment variables
PROVIDERS__OPENAI__API_KEY=sk-your-openai-api-key-here
PROVIDERS__OPENAI__BASE_URL=https://api.openai.com/v1
PROVIDERS__OPENAI__ORGANIZATION=org-your-org-id
PROVIDERS__OPENAI__ENABLED=true
IMAGES__DEFAULT_MODEL=gpt-image-2.5-sunburst  # also the default edit model
```

### Generate Image via MCP

```python
# Generate with OpenAI model
result = await session.call_tool("generate_image", {
    "prompt": "A beautiful sunset over mountains",
    "model": "gpt-image-2.5-flare",
    "quality": "high",
    "size": "1536x1024"
})
```

### Edit Image via MCP

```python
# Edit image using OpenAI models ("model" is optional; defaults to
# IMAGES__DEFAULT_MODEL)
result = await session.call_tool("edit_image", {
    "image_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "prompt": "Add a rainbow to the sky",
    "model": "gpt-image-2.5-sunburst"
})
```

## Getting Exact Text in Images

The GPT Image 2.5 models render lettering well enough for shop signs, neon, posters, labels and decals, but only if the prompt pins the text down. OpenAI still lists precise text placement as a limitation, so check every result.

1. **Quote the exact string.** Put the required wording in double quotes and say how many times it appears: `The sign reads exactly "HARROWFIELD ARCADE", once.` Use one quoted string for each separate piece of text.
2. **State the font or lettering style.** For example "chunky hand-painted red capitals", "cream serif capitals on dark green" or "pink neon script tubing".
3. **State the placement.** Say where the text sits (on the fascia board, centred, across the top of the poster), how big it is relative to the surface, and any line breaks (`on two lines: "HARROWFIELD" above "ARCADE"`).
4. **State the language and spelling.** For example "British English spelling". Spell invented or unusual names letter by letter when a result gets them wrong (`H-A-R-R-O-W-F-I-E-L-D`).
5. **Rule out extra text.** End with "No other text, no watermarks, no logos."
6. **Choose the quality deliberately.** Draft at `low`. For small or dense lettering, compare `medium` and `high`. Use `xhigh` or `max` only when a lower setting still gets the text wrong, because a higher setting costs more, takes longer and doesn't guarantee a better result. Use `gpt-image-2.5-sunburst` for finals where the lettering must be exact, and `gpt-image-2.5-flare` for drafts and volume.
7. **Check and fix.** Read every letter of the output. If it's wrong, keep the prompt strict and change one thing at a time, or fix only the text with `edit_image`: `Change only the sign text to exactly "CLOSED". Keep the font, colours and layout.`

Example:

```text
Photo of a small British seaside amusement arcade shopfront.
The fascia sign reads exactly "HARROWFIELD ARCADE", once, in chunky red
hand-painted capitals on a white board, centred. British English.
No other text, no watermarks, no logos.
```

## Model Comparison

| Model | Generation | Editing | Max Prompt | Sizes | Quality Options |
|-------|------------|---------|------------|-------|----------------|
| gpt-image-2.5-sunburst | ✅ | ✅ | 32,000 chars | Presets + any WxH (mult-of-16, max edge 3840, aspect ≤3:1) | auto, low, medium, high, xhigh, max |
| gpt-image-2.5-flare | ✅ | ✅ | 32,000 chars | Presets + any WxH (mult-of-16, max edge 3840, aspect ≤3:1) | auto, low, medium, high, xhigh, max |
| gpt-image-2 | ✅ | ✅ | 32,000 chars | Presets + any WxH (mult-of-16, max edge 3840, aspect ≤3:1) | auto, high, medium, low |
| gpt-image-1.5 | ✅ | ✅ | 32,000 chars | 1024x1024, 1536x1024, 1024x1536 | auto, high, medium, low |
| gpt-image-1 | ✅ | ✅ | 32,000 chars | 1024x1024, 1536x1024, 1024x1536 | auto, high, medium, low |
| dall-e-3 (removed from API) | ✅ | ❌ | 4,000 chars | 1024x1024, 1792x1024, 1024x1792 | hd, standard |
| dall-e-2 (removed from API) | ✅ | ❌ | 1,000 chars | 256x256, 512x512, 1024x1024 | standard |

## Costs

Both GPT Image 2.5 models bill $30 per 1M image output tokens, like `gpt-image-2`, but equal token rates don't mean equal cost per image: token use depends on model, quality and size. The server estimates output tokens with the formula from OpenAI's image-generation calculator, which uses one table for both 2.5 models, so treat the figures as estimates and use the response's `usage` for real consumption.

| Size | low | medium | high | xhigh | max |
|------|-----|--------|------|-------|-----|
| 1024x1024 | 196 (~$0.006) | 439 (~$0.013) | 1,756 (~$0.053) | 3,122 (~$0.094) | 7,024 (~$0.211) |
| 1536x1024 | 158 (~$0.005) | 343 (~$0.010) | 1,372 (~$0.041) | 2,459 (~$0.074) | 5,488 (~$0.165) |
| 3840x2160 | 371 (~$0.011) | 865 (~$0.026) | 3,336 (~$0.100) | 5,930 (~$0.178) | 13,342 (~$0.400) |

Output tokens per image; text and image input tokens are extra. A larger non-square size can use fewer output tokens than a square one at the same quality.

## Response Format

```json
{
  "created": 1713833628,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ],
  "usage": {
    "total_tokens": 100,
    "input_tokens": 50,
    "output_tokens": 50
  }
}
```

## Error Handling

Common error codes:
- `400`: Bad request - invalid parameters
- `401`: Unauthorized - invalid API key
- `429`: Rate limit exceeded
- `500`: Internal server error

GPT Image models may require API Organization Verification in the OpenAI developer console.

## Best Practices

1. **Model Selection**:
   - Use `gpt-image-2.5-sunburst` (the default) when quality, edit precision or exact lettering matter most
   - Use `gpt-image-2.5-flare` for drafts, variations and high-volume work, and switch to it for a workload only once it meets the same quality bar
   - Use `gpt-image-2` for existing workflows already validated on it

2. **Parameter Optimization**:
   - Tune quality one step at a time: start low, raise it only when the output falls short
   - Use appropriate sizes for your use case; outputs above 3,686,400 pixels (2560x1440) are experimental
   - Set moderation levels according to content policy needs

3. **Error Handling**: Implement retry logic with exponential backoff
4. **Rate Limiting**: Respect OpenAI's rate limits (5 images per minute at tier 1 for the 2.5 models)
5. **Cost Management**: Monitor token usage and costs

## Migration Notes

If you're migrating from the single-provider version:
- Add `model` parameter to specify which OpenAI model to use
- Use `list_available_models` tool to discover available models
- Consider trying different models for different use cases

## Sources

- [GPT-Image-2.5 Sunburst model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst)
- [GPT-Image-2.5 Flare model page](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare)
- [API changelog](https://developers.openai.com/api/docs/changelog) (8 September 2026 release; 12 May 2026 DALL·E removal)
- [Image generation guide](https://developers.openai.com/api/docs/guides/image-generation) (sizes, qualities, formats, moderation, masks, cost calculator)
- [Image prompting guide](https://developers.openai.com/api/docs/guides/image-prompting) (model choice, quality tuning, exact text)
- [Create image reference](https://developers.openai.com/api/reference/resources/images/methods/generate) and [Create image edit reference](https://developers.openai.com/api/reference/resources/images/methods/edit)

For complete documentation covering all supported providers, see the [Multi-Provider API Guide](multi_provider_api_guide.md).
