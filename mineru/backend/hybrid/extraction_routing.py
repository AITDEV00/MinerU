# Copyright (c) Opendatalab. All rights reserved.
"""
Extraction routing for split VLM path: prepare stage that routes blocks to
PDF text extraction, OCR, or Qwen3-VL based on content type and language.
"""

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from mineru.utils.span_pre_proc import calculate_char_in_span, chars_to_content
from mineru.utils.language import detect_lang
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_text_tool import get_page

# Languages that use OCR (faster); others use Qwen3-VL
# Matches PaddleOCR-supported langs and API lang_list options
OCR_LANGUAGES = frozenset([
    "en", "ch", "ch_lite", "ch_server", "latin", "korean", "japan", "chinese_cht",
    "th", "el",
    "de", "fr", "es", "it", "pt", "nl", "pl",  # Latin-script variants
])

# Block types that always need Qwen3-VL (no PDF text, no OCR)
VLM_ONLY_TYPES = frozenset([
    "table", "equation", "interline_equation", "equation_interline",
    "image", "chart", "code", "code_body",
])

# Min chars to consider PDF text "meaningful"
MIN_PDF_TEXT_LEN = 3


def _block_type_str(block) -> str:
    """Human-readable layout block type for logging and routing (empty if unset)."""
    bt = getattr(block, "type", None) or (block.type if hasattr(block, "type") else "")
    if bt is None or bt == "":
        return ""
    return bt.value if hasattr(bt, "value") else str(bt)


def log_smart_routing_type_breakdown(result: "RoutingResult", blocks_list: list) -> None:
    """Log how many blocks of each layout type go to PDF text vs OCR vs Qwen3-VL (initial route)."""
    pdf_c: Counter = Counter()
    for page_idx, block_idx, _ in result.pdf_text_blocks:
        if page_idx < len(blocks_list) and block_idx < len(blocks_list[page_idx]):
            t = _block_type_str(blocks_list[page_idx][block_idx]) or "unknown"
            pdf_c[t] += 1
    ocr_c: Counter = Counter()
    for page_idx, block_idx, _, _ in result.ocr_blocks:
        if page_idx < len(blocks_list) and block_idx < len(blocks_list[page_idx]):
            t = _block_type_str(blocks_list[page_idx][block_idx]) or "unknown"
            ocr_c[t] += 1
    vlm_c: Counter = Counter()
    for page_idx, block_idx, _ in result.vlm_indices:
        if page_idx < len(blocks_list) and block_idx < len(blocks_list[page_idx]):
            t = _block_type_str(blocks_list[page_idx][block_idx]) or "unknown"
            vlm_c[t] += 1
    logger.info(
        "Smart routing by block type — "
        f"PDF text path: {dict(pdf_c)} | "
        f"OCR path: {dict(ocr_c)} | "
        f"Qwen3-VL path (initial): {dict(vlm_c)}"
    )


def log_vlm_batch_breakdown(
    blocks_list: list,
    vlm_combined: list[tuple[int, int, int]],
    routing_vlm: list[tuple[int, int, int]],
    ocr_unavailable_fallback: list[tuple[int, int, int]],
    ocr_empty_fallback: list[tuple[int, int, int]],
) -> None:
    """
    Log why each block in the final Qwen3-VL batch was sent there:
    vlm_direct (type/language/pdf rules), ocr_unavailable, ocr_empty.
    """
    if not vlm_combined:
        return
    direct_set = set(map(tuple, routing_vlm))
    ocr_unavail_set = set(map(tuple, ocr_unavailable_fallback))
    ocr_empty_set = set(map(tuple, ocr_empty_fallback))
    by_reason: Counter = Counter()
    by_type_reason: dict[str, Counter] = defaultdict(Counter)
    for p, b, f in vlm_combined:
        key = (p, b, f)
        if key in direct_set:
            reason = "vlm_direct"
        elif key in ocr_unavail_set:
            reason = "ocr_unavailable_fallback"
        elif key in ocr_empty_set:
            reason = "ocr_empty_fallback"
        else:
            reason = "unknown"
        by_reason[reason] += 1
        if p < len(blocks_list) and b < len(blocks_list[p]):
            bt = _block_type_str(blocks_list[p][b]) or "unknown"
            by_type_reason[bt][reason] += 1
    detail = {bt: dict(by_type_reason[bt]) for bt in sorted(by_type_reason)}
    logger.info(
        f"Qwen3-VL batch reasons: {dict(by_reason)} | per block type: {detail}"
    )


@dataclass
class RoutingResult:
    """Result of prepare-stage routing per block."""
    pdf_text_blocks: list[tuple[int, int, str]] = field(default_factory=list)  # (page_idx, block_idx, text)
    ocr_blocks: list[tuple[int, int, Any, int]] = field(default_factory=list)   # (page_idx, block_idx, block_image, flat_idx)
    vlm_indices: list[tuple[int, int, int]] = field(default_factory=list)      # (page_idx, block_idx, flat_index)


def _layout_bbox_to_pdf_coords(bbox: list[float], page_width: float, page_height: float) -> tuple[float, float, float, float]:
    """Convert layout bbox (normalized 0-1, y down) to PDF coords (y up)."""
    x0 = bbox[0] * page_width
    y1_img = bbox[1] * page_height  # top in image
    x1 = bbox[2] * page_width
    y0_img = bbox[3] * page_height  # bottom in image
    # PDF: y=0 at bottom, y increases up. Image: y=0 at top.
    # pdf_y = page_height - img_y
    pdf_y0 = page_height - y0_img
    pdf_y1 = page_height - y1_img
    return (x0, pdf_y0, x1, pdf_y1)


def _extract_text_for_block(page_dict: dict, layout_bbox_pdf: tuple, median_height: float) -> str:
    """Extract text from PDF page chars that overlap with layout block bbox."""
    page_all_chars = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    page_all_chars.append(char)

    if not page_all_chars:
        return ""

    # Build a synthetic span for our layout block
    span = {
        "bbox": layout_bbox_pdf,
        "chars": [],
        "height": layout_bbox_pdf[3] - layout_bbox_pdf[1],
        "width": layout_bbox_pdf[2] - layout_bbox_pdf[0],
    }

    # Collect chars that fall inside our block (same logic as fill_char_in_spans)
    for idx, char in enumerate(page_all_chars):
        char_bbox = char.get("bbox")
        if char_bbox is None:
            continue
        if hasattr(char_bbox, "bbox"):
            char_bbox = char_bbox.bbox
        if isinstance(char_bbox, (list, tuple)) and len(char_bbox) >= 4:
            pass
        else:
            continue
        if calculate_char_in_span(char_bbox, layout_bbox_pdf, char.get("char", "")):
            c = dict(char) if isinstance(char, dict) else {"char": char.get("char", ""), "bbox": char_bbox}
            c.setdefault("char_idx", char.get("char_idx", idx))
            span["chars"].append(c)

    chars_to_content(span)
    return span.get("content", "").strip()


def run_prepare_stage(
    pdf_bytes: bytes,
    pdf_doc: Any,
    images: list,
    blocks_list: list,
    not_extract_list: list,
    language: str,
    prepared: list,
) -> RoutingResult:
    """
    Run prepare stage: route blocks to PDF text, OCR, or VLM.
    prepared: list of (block_images, prompts, params, indices) per page from batch_prepare_for_extract.
    """
    result = RoutingResult()
    pdf_extractable = classify(pdf_bytes) == "txt"

    flat_idx = 0
    for page_idx, (block_images, prompts, params, indices) in enumerate(prepared):
        if page_idx >= len(blocks_list):
            break
        blocks = blocks_list[page_idx]
        img = images[page_idx] if page_idx < len(images) else None

        page_dict = None
        page_width = 595
        page_height = 842
        if pdf_extractable and pdf_doc is not None:
            try:
                page = pdf_doc[page_idx]
                page_dict = get_page(page)
                page_width = page_dict.get("width", 595)
                page_height = page_dict.get("height", 842)
            except Exception as e:
                logger.debug(f"PDF text extraction failed for page {page_idx}: {e}")

        for i, block_idx in enumerate(indices):
            if block_idx >= len(blocks):
                flat_idx += 1
                continue
            block = blocks[block_idx]
            block_type = _block_type_str(block)

            # Skip blocks in not_extract_list
            if not_extract_list and block_type in not_extract_list:
                flat_idx += 1
                continue

            # Table, equation, image always go to VLM
            if block_type in VLM_ONLY_TYPES:
                result.vlm_indices.append((page_idx, block_idx, flat_idx))
                flat_idx += 1
                continue

            block_image = block_images[i] if i < len(block_images) else None

            # Try PDF text extraction
            pdf_text = ""
            if page_dict is not None:
                try:
                    bbox = getattr(block, "bbox", None) or []
                    if len(bbox) >= 4:
                        pdf_bbox = _layout_bbox_to_pdf_coords(bbox, page_width, page_height)
                        median_h = (pdf_bbox[3] - pdf_bbox[1]) or 12.0
                        pdf_text = _extract_text_for_block(page_dict, pdf_bbox, median_h)
                except Exception as e:
                    logger.debug(f"PDF extract for block ({page_idx},{block_idx}): {e}")

            # Use stripped length so whitespace-only PDF matches do not skip OCR/VLM
            if pdf_text and len(pdf_text.strip()) >= MIN_PDF_TEXT_LEN:
                detected = detect_lang(pdf_text)
                if detected in ("en", "zh", "ja", "ko", "th", "el", "fr", "de", "es", "it"):
                    result.pdf_text_blocks.append((page_idx, block_idx, pdf_text))
                    flat_idx += 1
                    continue
                elif detected in ("ar",):
                    result.vlm_indices.append((page_idx, block_idx, flat_idx))
                    flat_idx += 1
                    continue
                else:
                    result.pdf_text_blocks.append((page_idx, block_idx, pdf_text))
                    flat_idx += 1
                    continue

            # No PDF text - route by document language
            if language in OCR_LANGUAGES:
                if block_image is not None:
                    result.ocr_blocks.append((page_idx, block_idx, block_image, flat_idx))
                else:
                    result.vlm_indices.append((page_idx, block_idx, flat_idx))
            else:
                result.vlm_indices.append((page_idx, block_idx, flat_idx))

            flat_idx += 1

    log_smart_routing_type_breakdown(result, blocks_list)
    return result


def is_smart_routing_enabled() -> bool:
    """Check if MINERU_VL_SMART_ROUTING_ENABLE is set. Default: true (enabled)."""
    return os.getenv("MINERU_VL_SMART_ROUTING_ENABLE", "1").lower() in ("1", "true", "yes")
