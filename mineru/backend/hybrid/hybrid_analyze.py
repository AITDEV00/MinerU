#  Copyright (c) Opendatalab. All rights reserved.
import asyncio
import os
import time
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.structs import BlockType
from tqdm import tqdm

from mineru.backend.hybrid.hybrid_model_output_to_middle_json import result_to_middle_json
from mineru.backend.pipeline.model_init import HybridModelSingleton
from mineru.backend.vlm.vlm_analyze import ModelSingleton
from mineru.data.data_reader_writer import DataWriter
from mineru.utils.config_reader import get_device
from mineru.utils.enum_class import ImageType, NotExtractType
from mineru.utils.model_utils import crop_img, get_vram, clean_memory
from mineru.utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, sorted_boxes, merge_det_boxes, \
    update_det_boxes, OcrConfidence
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import load_images_from_pdf

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 让mps可以fallback
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 16

not_extract_list = [item.value for item in NotExtractType]

def ocr_classify(pdf_bytes, parse_method: str = 'auto',) -> bool:
    # 确定OCR设置
    _ocr_enable = False
    if parse_method == 'auto':
        if classify(pdf_bytes) == 'ocr':
            _ocr_enable = True
    elif parse_method == 'ocr':
        _ocr_enable = True
    elif parse_method == 'vlm':
        _ocr_enable = True  # VLM path handles extraction; treat as OCR-enabled for middle_json
    return _ocr_enable

def ocr_det(
    hybrid_pipeline_model,
    np_images,
    results,
    mfd_res,
    _ocr_enable,
    batch_radio: int = 1,
):
    ocr_res_list = []
    if not hybrid_pipeline_model.enable_ocr_det_batch:
        # 非批处理模式 - 逐页处理
        for np_image, page_mfd_res, page_results in tqdm(
            zip(np_images, mfd_res, results),
            total=len(np_images),
            desc="OCR-det"
        ):
            ocr_res_list.append([])
            img_height, img_width = np_image.shape[:2]
            for res in page_results:
                if res['type'] not in not_extract_list:
                    continue
                x0 = max(0, int(res['bbox'][0] * img_width))
                y0 = max(0, int(res['bbox'][1] * img_height))
                x1 = min(img_width, int(res['bbox'][2] * img_width))
                y1 = min(img_height, int(res['bbox'][3] * img_height))
                if x1 <= x0 or y1 <= y0:
                    continue
                res['poly'] = [x0, y0, x1, y0, x1, y1, x0, y1]
                new_image, useful_list = crop_img(
                    res, np_image, crop_paste_x=50, crop_paste_y=50
                )
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    page_mfd_res, useful_list
                )
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                ocr_res = hybrid_pipeline_model.ocr_model.ocr(
                    bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False
                )[0]
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(
                        ocr_res, useful_list, _ocr_enable, bgr_image, hybrid_pipeline_model.lang
                    )

                    ocr_res_list[-1].extend(ocr_result_list)
    else:
        # 批处理模式 - 按语言和分辨率分组
        # 收集所有需要OCR检测的裁剪图像
        all_cropped_images_info = []

        for np_image, page_mfd_res, page_results in zip(
                np_images, mfd_res, results
        ):
            ocr_res_list.append([])
            img_height, img_width = np_image.shape[:2]
            for res in page_results:
                if res['type'] not in not_extract_list:
                    continue
                x0 = max(0, int(res['bbox'][0] * img_width))
                y0 = max(0, int(res['bbox'][1] * img_height))
                x1 = min(img_width, int(res['bbox'][2] * img_width))
                y1 = min(img_height, int(res['bbox'][3] * img_height))
                if x1 <= x0 or y1 <= y0:
                    continue
                res['poly'] = [x0, y0, x1, y0, x1, y1, x0, y1]
                new_image, useful_list = crop_img(
                    res, np_image, crop_paste_x=50, crop_paste_y=50
                )
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    page_mfd_res, useful_list
                )
                bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                all_cropped_images_info.append((
                    bgr_image, useful_list, adjusted_mfdetrec_res, ocr_res_list[-1]
                ))

        # 按分辨率分组并同时完成padding
        RESOLUTION_GROUP_STRIDE = 64  # 32

        resolution_groups = defaultdict(list)
        for crop_info in all_cropped_images_info:
            cropped_img = crop_info[0]
            h, w = cropped_img.shape[:2]
            # 直接计算目标尺寸并用作分组键
            target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            group_key = (target_h, target_w)
            resolution_groups[group_key].append(crop_info)

        # 对每个分辨率组进行批处理
        for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det"):
            # 对所有图像进行padding到统一尺寸
            batch_images = []
            for crop_info in group_crops:
                img = crop_info[0]
                h, w = img.shape[:2]
                # 创建目标尺寸的白色背景
                padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                padded_img[:h, :w] = img
                batch_images.append(padded_img)

            # 批处理检测
            det_batch_size = min(len(batch_images), batch_radio*OCR_DET_BASE_BATCH_SIZE)
            batch_results = hybrid_pipeline_model.ocr_model.text_detector.batch_predict(batch_images, det_batch_size)

            # 处理批处理结果
            for crop_info, (dt_boxes, _) in zip(group_crops, batch_results):
                bgr_image, useful_list, adjusted_mfdetrec_res, ocr_page_res_list = crop_info

                if dt_boxes is not None and len(dt_boxes) > 0:
                    # 处理检测框
                    dt_boxes_sorted = sorted_boxes(dt_boxes)
                    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                    # 根据公式位置更新检测框
                    dt_boxes_final = (update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                                      if dt_boxes_merged and adjusted_mfdetrec_res
                                      else dt_boxes_merged)

                    if dt_boxes_final:
                        ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, _ocr_enable, bgr_image, hybrid_pipeline_model.lang
                        )
                        ocr_page_res_list.extend(ocr_result_list)
    return ocr_res_list

def mask_image_regions(np_images, results):
    # 根据vlm返回的结果，在每一页中将image、table、equation块mask成白色背景图像
    for np_image, vlm_page_results in zip(np_images, results):
        img_height, img_width = np_image.shape[:2]
        # 收集需要mask的区域
        mask_regions = []
        for block in vlm_page_results:
            if block['type'] in [BlockType.IMAGE, BlockType.TABLE, BlockType.EQUATION]:
                bbox = block['bbox']
                # 批量转换归一化坐标到像素坐标,并进行边界检查
                x0 = max(0, int(bbox[0] * img_width))
                y0 = max(0, int(bbox[1] * img_height))
                x1 = min(img_width, int(bbox[2] * img_width))
                y1 = min(img_height, int(bbox[3] * img_height))
                # 只添加有效区域
                if x1 > x0 and y1 > y0:
                    mask_regions.append((y0, y1, x0, x1))
        # 批量应用mask
        for y0, y1, x0, x1 in mask_regions:
            np_image[y0:y1, x0:x1, :] = 255
    return np_images

def normalize_poly_to_bbox(item, page_width, page_height):
    """将poly坐标归一化为bbox"""
    poly = item['poly']
    x0 = min(max(poly[0] / page_width, 0), 1)
    y0 = min(max(poly[1] / page_height, 0), 1)
    x1 = min(max(poly[4] / page_width, 0), 1)
    y1 = min(max(poly[5] / page_height, 0), 1)
    item['bbox'] = [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]
    item.pop('poly', None)


def _process_ocr_and_formulas(
    images_pil_list,
    results,
    language,
    inline_formula_enable,
    _ocr_enable,
    batch_radio: int = 1,
):
    """处理OCR和公式识别"""

    # 遍历results,对文本块截图交由OCR识别
    # 根据_ocr_enable决定ocr只开det还是det+rec
    # 根据inline_formula_enable决定是使用mfd和ocr结合的方式,还是纯ocr方式

    # 将PIL图片转换为numpy数组
    np_images = [np.asarray(pil_image).copy() for pil_image in images_pil_list]

    # 获取混合模型实例
    hybrid_model_singleton = HybridModelSingleton()
    hybrid_pipeline_model = hybrid_model_singleton.get_model(
        lang=language,
        formula_enable=inline_formula_enable,
    )

    if inline_formula_enable:
        # 在进行`行内`公式检测和识别前，先将图像中的图片、表格、`行间`公式区域mask掉
        np_images = mask_image_regions(np_images, results)
        # 公式检测
        images_mfd_res = hybrid_pipeline_model.mfd_model.batch_predict(np_images, batch_size=1, conf=0.5)
        # 公式识别
        inline_formula_list = hybrid_pipeline_model.mfr_model.batch_predict(
            images_mfd_res,
            np_images,
            batch_size=batch_radio*MFR_BASE_BATCH_SIZE,
            interline_enable=True,
        )
    else:
        inline_formula_list = [[] for _ in range(len(images_pil_list))]

    mfd_res = []
    for page_inline_formula_list in inline_formula_list:
        page_mfd_res = []
        for formula in page_inline_formula_list:
            formula['category_id'] = 13
            page_mfd_res.append({
                "bbox": [int(formula['poly'][0]), int(formula['poly'][1]),
                         int(formula['poly'][4]), int(formula['poly'][5])],
            })
        mfd_res.append(page_mfd_res)

    # vlm没有执行ocr，需要ocr_det
    ocr_res_list = ocr_det(
        hybrid_pipeline_model,
        np_images,
        results,
        mfd_res,
        _ocr_enable,
        batch_radio=batch_radio,
    )

    # 如果需要ocr则做ocr_rec
    if _ocr_enable:
        need_ocr_list = []
        img_crop_list = []
        for page_ocr_res_list in ocr_res_list:
            for ocr_res in page_ocr_res_list:
                if 'np_img' in ocr_res:
                    need_ocr_list.append(ocr_res)
                    img_crop_list.append(ocr_res.pop('np_img'))
        if len(img_crop_list) > 0:
            # Process OCR
            ocr_result_list = hybrid_pipeline_model.ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

            # Verify we have matching counts
            assert len(ocr_result_list) == len(need_ocr_list), f'ocr_result_list: {len(ocr_result_list)}, need_ocr_list: {len(need_ocr_list)}'

            # Process OCR results for this language
            for index, need_ocr_res in enumerate(need_ocr_list):
                ocr_text, ocr_score = ocr_result_list[index]
                need_ocr_res['text'] = ocr_text
                need_ocr_res['score'] = float(f"{ocr_score:.3f}")
                if ocr_score < OcrConfidence.min_confidence:
                    need_ocr_res['category_id'] = 16
                else:
                    layout_res_bbox = [need_ocr_res['poly'][0], need_ocr_res['poly'][1],
                                       need_ocr_res['poly'][4], need_ocr_res['poly'][5]]
                    layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                    layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                    if (
                            ocr_text in [
                                '（204号', '（20', '（2', '（2号', '（20号', '号','（204',
                                '(cid:)', '(ci:)', '(cd:1)', 'cd:)', 'c)', '(cd:)', 'c', 'id:)',
                                ':)', '√:)', '√i:)', '−i:)', '−:' , 'i:)',
                            ]
                            and ocr_score < 0.8
                            and layout_res_width < layout_res_height
                    ):
                        need_ocr_res['category_id'] = 16

    return inline_formula_list, ocr_res_list, hybrid_pipeline_model


def _normalize_bbox(
    inline_formula_list,
    ocr_res_list,
    images_pil_list,
):
    """归一化坐标并生成最终结果"""
    for page_inline_formula_list, page_ocr_res_list, page_pil_image in zip(
            inline_formula_list, ocr_res_list, images_pil_list
    ):
        if page_inline_formula_list or page_ocr_res_list:
            page_width, page_height = page_pil_image.size
            # 处理公式列表
            for formula in page_inline_formula_list:
                normalize_poly_to_bbox(formula, page_width, page_height)
            # 处理OCR结果列表
            for ocr_res in page_ocr_res_list:
                normalize_poly_to_bbox(ocr_res, page_width, page_height)


def get_batch_ratio(device):
    """
    根据显存大小或环境变量获取 batch ratio
    """
    # 1. 优先尝试从环境变量获取
    """
    c/s架构分离部署时，建议通过设置环境变量 MINERU_HYBRID_BATCH_RATIO 来指定 batch ratio
    建议的设置值如如下，以下配置值已考虑一定的冗余，单卡多终端部署时为了保证稳定性，可以额外保留一个client端的显存作为整体冗余
    单个client端显存大小 | MINERU_HYBRID_BATCH_RATIO
    ------------------|------------------------
    <= 6   GB         | 8
    <= 4.5 GB         | 4
    <= 3   GB         | 2
    <= 2.5 GB         | 1
    例如：
    export MINERU_HYBRID_BATCH_RATIO=4
    """
    env_val = os.getenv("MINERU_HYBRID_BATCH_RATIO")
    if env_val:
        try:
            batch_ratio = int(env_val)
            logger.info(f"hybrid batch ratio (from env): {batch_ratio}")
            return batch_ratio
        except ValueError as e:
            logger.warning(f"Invalid MINERU_HYBRID_BATCH_RATIO value: {env_val}, switching to auto mode. Error: {e}")

    # 2. 根据显存自动推断
    """
    根据总显存大小粗略估计 batch ratio，需要排除掉vllm等推理框架占用的显存开销
    """
    gpu_memory = get_vram(device)
    if gpu_memory >= 32:
        batch_ratio = 16
    elif gpu_memory >= 16:
        batch_ratio = 8
    elif gpu_memory >= 12:
        batch_ratio = 4
    elif gpu_memory >= 8:
        batch_ratio = 2
    else:
        batch_ratio = 1

    logger.info(f"hybrid batch ratio (auto, vram={gpu_memory}GB): {batch_ratio}")
    return batch_ratio


# MinerU2.5 extraction: ch, en only (non-split path)
VLM_LANGUAGES = ["ch", "en"]

# Qwen3-VL-30B extraction supports all lang_list options (split path when parse_method=vlm)
QWEN3_VL_LANGUAGES = frozenset([
    "ch", "ch_lite", "ch_server", "en", "korean", "japan", "chinese_cht",
    "ta", "te", "ka", "th", "el", "latin", "arabic", "east_slavic", "cyrillic", "devanagari",
])


def _build_split_not_extract_list() -> list[str]:
    """Build not_extract_list for split layout+extraction based on table_enable and formula_enable.
    When table_enable=False, skip table extraction (render as image). When formula_enable=False, skip equation extraction."""
    skip = []
    _table_enable = os.getenv("MINERU_VLM_TABLE_ENABLE", "true").lower() in ("true", "1", "yes")
    _formula_enable = os.getenv("MINERU_VLM_FORMULA_ENABLE", "true").lower() in ("true", "1", "yes")
    if not _table_enable:
        skip.append("table")
    if not _formula_enable:
        skip.extend(["equation", "interline_equation"])
    if skip:
        logger.info(f"Extraction skip list (table_enable={_table_enable}, formula_enable={_formula_enable}): {skip}")
    return skip


def _is_split_vlm_ocr(parse_method: str, language: str, inline_formula_enable: bool) -> bool:
    """True when parse_method=vlm, language supported by Qwen3-VL, and MINERU_VL_SERVER_EXTRACTION is set.
    formula_enable no longer gates the split flow; when False, equation extraction is skipped and equations render as images."""
    force_pipeline = os.getenv("MINERU_HYBRID_FORCE_PIPELINE_ENABLE", "0").lower() in ("1", "true", "yes")
    extraction_url = os.getenv("MINERU_VL_SERVER_EXTRACTION")
    return (
        parse_method == "vlm"
        and language in QWEN3_VL_LANGUAGES
        and not force_pipeline
        and bool(extraction_url)
    )


def _should_enable_vlm_ocr(ocr_enable: bool, language: str, inline_formula_enable: bool, parse_method: str = "auto") -> bool:
    """判断是否启用VLM OCR (VLM extraction instead of pipeline OCR)"""
    force_enable = os.getenv("MINERU_FORCE_VLM_OCR_ENABLE", "0").lower() in ("1", "true", "yes")
    if force_enable:
        return True

    force_pipeline = os.getenv("MINERU_HYBRID_FORCE_PIPELINE_ENABLE", "0").lower() in ("1", "true", "yes")
    # When parse_method=vlm and extraction server is set, _is_split_vlm_ocr handles it
    if _is_split_vlm_ocr(parse_method, language, inline_formula_enable):
        return True
    return (
        ocr_enable
        and language in VLM_LANGUAGES
        and inline_formula_enable
        and not force_pipeline
    )


def _get_layout_server_url(server_url: str | None) -> tuple[str | None, dict | None]:
    """Get MinerU2.5 layout server URL and optional headers."""
    url = server_url or os.getenv("MINERU_VL_SERVER")
    if not url:
        return None, None
    url = url.rstrip("/") + "/" if not url.endswith("/") else url
    api_key = os.getenv("MINERU_VL_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    return url, headers


def _get_extraction_server_url() -> tuple[str | None, dict | None, str | None]:
    """Get Qwen3-VL extraction server URL, headers, and optional model name."""
    url = os.getenv("MINERU_VL_SERVER_EXTRACTION")
    if not url:
        return None, None, None
    url = url.rstrip("/") + "/" if not url.endswith("/") else url
    api_key = os.getenv("MINERU_VL_API_KEY_EXTRACTION") or os.getenv("MINERU_VL_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    model_name = os.getenv("MINERU_VL_MODEL_NAME_EXTRACTION") or "Qwen/Qwen3-VL-30B-A3B-Instruct"
    return url, headers, model_name


def _batch_split_layout_extract(
    layout_client: MinerUClient,
    extraction_client: MinerUClient,
    images: list,
    not_extract_list: list | None = None,
) -> list:
    """Layout via MinerU2.5, extraction via Qwen3-VL."""
    blocks_list = layout_client.batch_layout_detect(images)
    logger.info("Extract Preparation: preparing block crops for extraction")
    prepared = layout_client.helper.batch_prepare_for_extract(
        layout_client.executor, images, blocks_list, not_extract_list
    )
    all_images, all_prompts, all_params, all_indices = [], [], [], []
    for page_idx, (block_images, prompts, params, indices) in enumerate(prepared):
        all_images.extend(block_images)
        all_prompts.extend(prompts)
        all_params.extend(params)
        all_indices.extend((page_idx, idx) for idx in indices)
    n_blocks = len(all_images)
    logger.info(f"Extraction: sending {n_blocks} blocks to Qwen3-VL")
    outputs = extraction_client.client.batch_predict(all_images, all_prompts, all_params, None)
    logger.info(f"Extraction: completed {len(outputs)} blocks")
    for (page_idx, block_idx), output in zip(all_indices, outputs):
        blocks_list[page_idx][block_idx].content = output
    return layout_client.helper.batch_post_process(layout_client.executor, blocks_list)


async def _aio_batch_split_layout_extract(
    layout_client: MinerUClient,
    extraction_client: MinerUClient,
    images: list,
    not_extract_list: list | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> list:
    """Async: Layout via MinerU2.5, extraction via Qwen3-VL."""
    semaphore = semaphore or asyncio.Semaphore(layout_client.max_concurrency)
    blocks_list = await layout_client.aio_batch_layout_detect(images, semaphore=semaphore)
    logger.info("Extract Preparation: preparing block crops for extraction")
    prepared = await asyncio.gather(*[
        layout_client.helper.aio_prepare_for_extract(
            layout_client.executor, images[i], blocks_list[i], not_extract_list
        )
        for i in range(len(images))
    ])
    all_images, all_prompts, all_params, all_indices = [], [], [], []
    for page_idx, (block_images, prompts, params, indices) in enumerate(prepared):
        all_images.extend(block_images)
        all_prompts.extend(prompts)
        all_params.extend(params)
        all_indices.extend((page_idx, idx) for idx in indices)
    n_blocks = len(all_images)
    logger.info(f"Extraction: sending {n_blocks} blocks to Qwen3-VL")
    outputs = await extraction_client.client.aio_batch_predict(
        all_images, all_prompts, all_params, None, semaphore=semaphore,
        use_tqdm=True, tqdm_desc="Extraction",
    )
    logger.info(f"Extraction: completed {len(outputs)} blocks")
    for (page_idx, block_idx), output in zip(all_indices, outputs):
        blocks_list[page_idx][block_idx].content = output
    post_processed = await asyncio.gather(*[
        layout_client.helper.aio_post_process(layout_client.executor, blocks)
        for blocks in blocks_list
    ])
    return list(post_processed)


def doc_analyze(
        pdf_bytes,
        image_writer: DataWriter | None,
        predictor: MinerUClient | None = None,
        backend="transformers",
        parse_method: str = 'auto',
        language: str = 'ch',
        inline_formula_enable: bool = True,
        model_path: str | None = None,
        server_url: str | None = None,
        **kwargs,
):
    _ocr_enable = ocr_classify(pdf_bytes, parse_method=parse_method)
    _split_vlm_ocr = _is_split_vlm_ocr(parse_method, language, inline_formula_enable)
    _vlm_ocr_enable = _should_enable_vlm_ocr(_ocr_enable, language, inline_formula_enable, parse_method)

    load_images_start = time.time()
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    load_images_time = round(time.time() - load_images_start, 2)
    logger.debug(f"load images cost: {load_images_time}, speed: {round(len(images_pil_list)/load_images_time, 3)} images/s")

    device = get_device()

    infer_start = time.time()
    if _split_vlm_ocr:
        # Layout via MinerU2.5, extraction via Qwen3-VL
        layout_url, layout_headers = _get_layout_server_url(server_url)
        extraction_url, extraction_headers, extraction_model_name = _get_extraction_server_url()
        if not layout_url or not extraction_url:
            raise ValueError("parse_method=vlm requires MINERU_VL_SERVER and MINERU_VL_SERVER_EXTRACTION")
        layout_kwargs = {**kwargs}
        extraction_kwargs = {**kwargs}
        _max_conc = int(os.getenv("MINERU_VL_MAX_CONCURRENCY", "16"))
        layout_kwargs.setdefault("max_concurrency", _max_conc)
        extraction_kwargs.setdefault("max_concurrency", _max_conc)
        if layout_headers:
            layout_kwargs["server_headers"] = layout_headers
        layout_kwargs.setdefault("http_timeout", int(os.getenv("MINERU_VL_HTTP_TIMEOUT", "600")))
        layout_kwargs.setdefault("max_retries", int(os.getenv("MINERU_VL_MAX_RETRIES", "5")))
        layout_kwargs.setdefault("retry_backoff_factor", float(os.getenv("MINERU_VL_RETRY_BACKOFF", "1.0")))
        if extraction_headers:
            extraction_kwargs["server_headers"] = extraction_headers
        if extraction_model_name:
            extraction_kwargs["model_name"] = extraction_model_name
        # VLM client tuning from env (helps with 504 upstream unavailable on overloaded inference gateway)
        _http_timeout = int(os.getenv("MINERU_VL_HTTP_TIMEOUT", "600"))
        _max_retries = int(os.getenv("MINERU_VL_MAX_RETRIES", "5"))
        _retry_backoff = float(os.getenv("MINERU_VL_RETRY_BACKOFF", "1.0"))
        extraction_kwargs.setdefault("http_timeout", _http_timeout)
        extraction_kwargs.setdefault("max_retries", _max_retries)
        extraction_kwargs.setdefault("retry_backoff_factor", _retry_backoff)
        layout_predictor = ModelSingleton().get_model(backend, model_path, layout_url, **layout_kwargs)
        # HttpVlmClient overwrites headers with MINERU_VL_API_KEY; temporarily use extraction key
        _old_key = os.environ.get("MINERU_VL_API_KEY")
        _ext_key = os.getenv("MINERU_VL_API_KEY_EXTRACTION")
        if _ext_key:
            os.environ["MINERU_VL_API_KEY"] = _ext_key
        try:
            extraction_predictor = ModelSingleton().get_model(backend, model_path, extraction_url, **extraction_kwargs)
        finally:
            if _old_key is not None:
                os.environ["MINERU_VL_API_KEY"] = _old_key
            elif _ext_key and "MINERU_VL_API_KEY" in os.environ:
                del os.environ["MINERU_VL_API_KEY"]
        split_not_extract = _build_split_not_extract_list()
        results = _batch_split_layout_extract(
            layout_predictor, extraction_predictor, images_pil_list, not_extract_list=split_not_extract
        )
        hybrid_pipeline_model = None
        inline_formula_list = [[] for _ in images_pil_list]
        ocr_res_list = [[] for _ in images_pil_list]
    elif _vlm_ocr_enable:
        if predictor is None:
            layout_url, layout_headers = _get_layout_server_url(server_url)
            model_kwargs = {**kwargs}
            if layout_headers:
                model_kwargs["server_headers"] = layout_headers
            model_kwargs.setdefault("http_timeout", int(os.getenv("MINERU_VL_HTTP_TIMEOUT", "600")))
            model_kwargs.setdefault("max_retries", int(os.getenv("MINERU_VL_MAX_RETRIES", "5")))
            model_kwargs.setdefault("retry_backoff_factor", float(os.getenv("MINERU_VL_RETRY_BACKOFF", "1.0")))
            predictor = ModelSingleton().get_model(backend, model_path, layout_url, **model_kwargs)
        results = predictor.batch_two_step_extract(images=images_pil_list)
        hybrid_pipeline_model = None
        inline_formula_list = [[] for _ in images_pil_list]
        ocr_res_list = [[] for _ in images_pil_list]
    else:
        if predictor is None:
            layout_url, layout_headers = _get_layout_server_url(server_url)
            model_kwargs = {**kwargs}
            if layout_headers:
                model_kwargs["server_headers"] = layout_headers
            model_kwargs.setdefault("http_timeout", int(os.getenv("MINERU_VL_HTTP_TIMEOUT", "600")))
            model_kwargs.setdefault("max_retries", int(os.getenv("MINERU_VL_MAX_RETRIES", "5")))
            model_kwargs.setdefault("retry_backoff_factor", float(os.getenv("MINERU_VL_RETRY_BACKOFF", "1.0")))
            predictor = ModelSingleton().get_model(backend, model_path, layout_url, **model_kwargs)
        batch_ratio = get_batch_ratio(device)
        results = predictor.batch_two_step_extract(
            images=images_pil_list,
            not_extract_list=not_extract_list
        )
        inline_formula_list, ocr_res_list, hybrid_pipeline_model = _process_ocr_and_formulas(
            images_pil_list,
            results,
            language,
            inline_formula_enable,
            _ocr_enable,
            batch_radio=batch_ratio,
        )
        _normalize_bbox(inline_formula_list, ocr_res_list, images_pil_list)
    infer_time = round(time.time() - infer_start, 2)
    logger.debug(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")

    # 生成中间JSON
    discarded_blocks_enable = kwargs.get("discarded_blocks_enable", True)
    middle_json = result_to_middle_json(
        results,
        inline_formula_list,
        ocr_res_list,
        images_list,
        pdf_doc,
        image_writer,
        _ocr_enable,
        _vlm_ocr_enable,
        hybrid_pipeline_model,
        discarded_blocks_enable=discarded_blocks_enable,
    )

    clean_memory(device)
    return middle_json, results, _vlm_ocr_enable


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    parse_method: str = 'auto',
    language: str = 'ch',
    inline_formula_enable: bool = True,
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):
    _ocr_enable = ocr_classify(pdf_bytes, parse_method=parse_method)
    _split_vlm_ocr = _is_split_vlm_ocr(parse_method, language, inline_formula_enable)
    _vlm_ocr_enable = _should_enable_vlm_ocr(_ocr_enable, language, inline_formula_enable, parse_method)

    load_images_start = time.time()
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    load_images_time = round(time.time() - load_images_start, 2)
    logger.debug(f"load images cost: {load_images_time}, speed: {round(len(images_pil_list)/load_images_time, 3)} images/s")

    device = get_device()

    infer_start = time.time()
    if _split_vlm_ocr:
        layout_url, layout_headers = _get_layout_server_url(server_url)
        extraction_url, extraction_headers, extraction_model_name = _get_extraction_server_url()
        if not layout_url or not extraction_url:
            raise ValueError("parse_method=vlm requires MINERU_VL_SERVER and MINERU_VL_SERVER_EXTRACTION")
        layout_kwargs = {**kwargs}
        extraction_kwargs = {**kwargs}
        _max_conc = int(os.getenv("MINERU_VL_MAX_CONCURRENCY", "16"))
        layout_kwargs.setdefault("max_concurrency", _max_conc)
        extraction_kwargs.setdefault("max_concurrency", _max_conc)
        if layout_headers:
            layout_kwargs["server_headers"] = layout_headers
        layout_kwargs.setdefault("http_timeout", int(os.getenv("MINERU_VL_HTTP_TIMEOUT", "600")))
        layout_kwargs.setdefault("max_retries", int(os.getenv("MINERU_VL_MAX_RETRIES", "5")))
        layout_kwargs.setdefault("retry_backoff_factor", float(os.getenv("MINERU_VL_RETRY_BACKOFF", "1.0")))
        if extraction_headers:
            extraction_kwargs["server_headers"] = extraction_headers
        if extraction_model_name:
            extraction_kwargs["model_name"] = extraction_model_name
        # VLM client tuning from env (helps with 504 upstream unavailable on overloaded inference gateway)
        _http_timeout = int(os.getenv("MINERU_VL_HTTP_TIMEOUT", "600"))
        _max_retries = int(os.getenv("MINERU_VL_MAX_RETRIES", "5"))
        _retry_backoff = float(os.getenv("MINERU_VL_RETRY_BACKOFF", "1.0"))
        extraction_kwargs.setdefault("http_timeout", _http_timeout)
        extraction_kwargs.setdefault("max_retries", _max_retries)
        extraction_kwargs.setdefault("retry_backoff_factor", _retry_backoff)
        layout_predictor = ModelSingleton().get_model(backend, model_path, layout_url, **layout_kwargs)
        # HttpVlmClient overwrites headers with MINERU_VL_API_KEY; temporarily use extraction key
        _old_key = os.environ.get("MINERU_VL_API_KEY")
        _ext_key = os.getenv("MINERU_VL_API_KEY_EXTRACTION")
        if _ext_key:
            os.environ["MINERU_VL_API_KEY"] = _ext_key
        try:
            extraction_predictor = ModelSingleton().get_model(backend, model_path, extraction_url, **extraction_kwargs)
        finally:
            if _old_key is not None:
                os.environ["MINERU_VL_API_KEY"] = _old_key
            elif _ext_key and "MINERU_VL_API_KEY" in os.environ:
                del os.environ["MINERU_VL_API_KEY"]
        split_not_extract = _build_split_not_extract_list()
        results = await _aio_batch_split_layout_extract(
            layout_predictor, extraction_predictor, images_pil_list, not_extract_list=split_not_extract
        )
        hybrid_pipeline_model = None
        inline_formula_list = [[] for _ in images_pil_list]
        ocr_res_list = [[] for _ in images_pil_list]
    elif _vlm_ocr_enable:
        if predictor is None:
            layout_url, layout_headers = _get_layout_server_url(server_url)
            model_kwargs = {**kwargs}
            if layout_headers:
                model_kwargs["server_headers"] = layout_headers
            model_kwargs.setdefault("http_timeout", int(os.getenv("MINERU_VL_HTTP_TIMEOUT", "600")))
            model_kwargs.setdefault("max_retries", int(os.getenv("MINERU_VL_MAX_RETRIES", "5")))
            model_kwargs.setdefault("retry_backoff_factor", float(os.getenv("MINERU_VL_RETRY_BACKOFF", "1.0")))
            predictor = ModelSingleton().get_model(backend, model_path, layout_url, **model_kwargs)
        results = await predictor.aio_batch_two_step_extract(images=images_pil_list)
        hybrid_pipeline_model = None
        inline_formula_list = [[] for _ in images_pil_list]
        ocr_res_list = [[] for _ in images_pil_list]
    else:
        if predictor is None:
            layout_url, layout_headers = _get_layout_server_url(server_url)
            model_kwargs = {**kwargs}
            if layout_headers:
                model_kwargs["server_headers"] = layout_headers
            model_kwargs.setdefault("http_timeout", int(os.getenv("MINERU_VL_HTTP_TIMEOUT", "600")))
            model_kwargs.setdefault("max_retries", int(os.getenv("MINERU_VL_MAX_RETRIES", "5")))
            model_kwargs.setdefault("retry_backoff_factor", float(os.getenv("MINERU_VL_RETRY_BACKOFF", "1.0")))
            predictor = ModelSingleton().get_model(backend, model_path, layout_url, **model_kwargs)
        batch_ratio = get_batch_ratio(device)
        results = await predictor.aio_batch_two_step_extract(
            images=images_pil_list,
            not_extract_list=not_extract_list
        )
        inline_formula_list, ocr_res_list, hybrid_pipeline_model = _process_ocr_and_formulas(
            images_pil_list,
            results,
            language,
            inline_formula_enable,
            _ocr_enable,
            batch_radio=batch_ratio,
        )
        _normalize_bbox(inline_formula_list, ocr_res_list, images_pil_list)
    infer_time = round(time.time() - infer_start, 2)
    logger.debug(f"infer finished, cost: {infer_time}, speed: {round(len(results)/infer_time, 3)} page/s")

    # 生成中间JSON
    discarded_blocks_enable = kwargs.get("discarded_blocks_enable", True)
    middle_json = result_to_middle_json(
        results,
        inline_formula_list,
        ocr_res_list,
        images_list,
        pdf_doc,
        image_writer,
        _ocr_enable,
        _vlm_ocr_enable,
        hybrid_pipeline_model,
        discarded_blocks_enable=discarded_blocks_enable,
    )

    clean_memory(device)
    return middle_json, results, _vlm_ocr_enable

