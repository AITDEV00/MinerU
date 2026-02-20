# Split VLM Flow (parse_method=vlm)

When `parse_method=vlm` is used with the hybrid backend, MinerU uses a **split layout+extraction** flow:

- **Layout**: MinerU2.5-2509-1.2B (via `MINERU_VL_SERVER`) — produces the required `middle.json` structure
- **Extraction**: Qwen3-VL-30B-A3B-Instruct (via `MINERU_VL_SERVER_EXTRACTION`) — handles text, table, and equation extraction

## Changes Made

1. **`mineru/backend/hybrid/hybrid_analyze.py`**
   - Added `parse_method=vlm` to `ocr_classify`
   - Added `_is_split_vlm_ocr()` — enables split flow when parse_method=vlm, language in supported set, and `MINERU_VL_SERVER_EXTRACTION` is set
   - Added `_batch_split_layout_extract()` and `_aio_batch_split_layout_extract()` — layout via MinerU2.5, extraction via Qwen3-VL
   - Added `_get_layout_server_url()` and `_get_extraction_server_url()` for server resolution
   - Removed `_resolve_vlm_server_for_language` (language-based full replacement)

2. **CLI/API**
   - Added `vlm` to `parse_method` options in `fast_api.py` and `client.py`

## Deployment

### 1. Build API Image with Patched MinerU

Use `Dockerfile.api-arabic` (build from ait-projects root):

```bash
cd /home/ffkhan/ait-projects
docker build -f sglang-mlops-images/minerU/Dockerfile.api-arabic \
  -t adeo-mineru-api:arabic-vlm .
# Push to Harbor
docker tag adeo-mineru-api:arabic-vlm zdc-ai-harbor.ecouncil.ae/aiteam/adeo-mineru-api:arabic-vlm
docker push zdc-ai-harbor.ecouncil.ae/aiteam/adeo-mineru-api:arabic-vlm
```

### 2. Environment Variables

Add to `icarus-mineru.env` (or ConfigMap):

```bash
# Layout: MinerU2.5 (same as MINERU_VL_SERVER)
# Extraction: Qwen3-VL-30B when parse_method=vlm
MINERU_VL_SERVER_EXTRACTION=https://inference.adeoaiengine.ecouncil.ae/models/91c6dca7-b1c4-4e0b-8fec-11e8b82448f2/proxy/
# Optional: separate API key for extraction server
# MINERU_VL_API_KEY_EXTRACTION=sk-...
# Optional: extraction batch size (default 8). Set 0 to disable chunking.
# MINERU_VL_EXTRACTION_BATCH_SIZE=8
```

### 3. Usage

When calling the MinerU API with `parse_method=vlm` and any supported `lang_list` (ch, ch_lite, ch_server, en, korean, japan, chinese_cht, ta, te, ka, th, el, latin, arabic, east_slavic, cyrillic, devanagari):

1. Layout detection uses MinerU2.5 (preserves correct `middle.json` structure)
2. Content extraction (text, tables, equations) uses Qwen3-VL-30B

For `parse_method=auto` or `parse_method=ocr`: layout and extraction both use MinerU2.5 (unchanged behavior).

## Flow

```
PDF (parse_method=vlm, lang supported) → hybrid-http-client
  → _is_split_vlm_ocr: True (MINERU_VL_SERVER_EXTRACTION set)
  → layout_client (MinerU2.5): batch_layout_detect
  → extraction_client (Qwen3-VL): batch_predict on block images (chunked via MINERU_VL_EXTRACTION_BATCH_SIZE)
  → middle_json (layout structure from MinerU2.5, content from Qwen3-VL)
```

## Batch Extraction (MINERU_VL_EXTRACTION_BATCH_SIZE)

Extraction requests to Qwen3-VL are chunked into batches (similar to `vlm_processor` in ait-icarus-backend):

- **Default**: 8 blocks per batch
- **Set to 0**: No chunking — all blocks sent with max_concurrency limit (original behavior)
- **Benefits**: Limits concurrent requests per round, reduces 504 "Upstream service unavailable" on overloaded gateways
