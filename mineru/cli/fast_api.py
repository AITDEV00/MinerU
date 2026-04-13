import sys
import uuid
import os
import re
import tempfile
import asyncio
import uvicorn
import click
import zipfile
from pathlib import Path
import glob
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from typing import Any, List, Optional
from loguru import logger

log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()  # 移除默认handler
logger.add(sys.stderr, level=log_level)  # 添加新handler

from base64 import b64encode

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.version import __version__

# 并发控制器
_request_semaphore: Optional[asyncio.Semaphore] = None

# 并发控制依赖函数
async def limit_concurrency():
    if _request_semaphore is not None:
        if _request_semaphore.locked():
            raise HTTPException(
                status_code=503,
                detail=f"Server is at maximum capacity: {os.getenv('MINERU_API_MAX_CONCURRENT_REQUESTS', 'unset')}. Please try again later."
            )
        async with _request_semaphore:
            yield
    else:
        yield

def create_app():
    # By default, the OpenAPI documentation endpoints (openapi_url, docs_url, redoc_url) are enabled.
    # To disable the FastAPI docs and schema endpoints, set the environment variable MINERU_API_ENABLE_FASTAPI_DOCS=0.
    enable_docs = str(os.getenv("MINERU_API_ENABLE_FASTAPI_DOCS", "1")).lower() in ("1", "true", "yes")
    app = FastAPI(
        openapi_url="/openapi.json" if enable_docs else None,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
    )

    # 初始化并发控制器：从环境变量MINERU_API_MAX_CONCURRENT_REQUESTS读取
    global _request_semaphore
    try:
        max_concurrent_requests = int(os.getenv("MINERU_API_MAX_CONCURRENT_REQUESTS", "0"))
    except ValueError:
        max_concurrent_requests = 0

    if max_concurrent_requests > 0:
        _request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        logger.info(f"Request concurrency limited to {max_concurrent_requests}")

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    return app

app = create_app()


def _fix_openapi_file_uploads_for_swagger_ui(schema: dict[str, Any]) -> None:
    """
    FastAPI >= 0.129.1 emits contentMediaType: application/octet-stream for UploadFile;
    Swagger UI 5.x only renders file pickers for format: binary (see fastapi#14975).
    Normalize so /docs shows Choose File again.
    """
    def fix_node(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("contentMediaType") == "application/octet-stream":
            node.pop("contentMediaType", None)
            node.setdefault("type", "string")
            node["format"] = "binary"
        for v in node.values():
            if isinstance(v, dict):
                fix_node(v)
            elif isinstance(v, list):
                for item in v:
                    fix_node(item)

    fix_node(schema.get("components", {}))
    fix_node(schema.get("paths", {}))


def _install_custom_openapi(application) -> None:
    """Replace app.openapi so generated schema is Swagger-file-picker compatible."""

    def custom_openapi():
        if application.openapi_schema:
            return application.openapi_schema
        from fastapi.openapi.utils import get_openapi

        openapi_schema = get_openapi(
            title=application.title,
            version=getattr(application, "version", "0.1.0"),
            openapi_version=getattr(application, "openapi_version", "3.1.0"),
            description=getattr(application, "description", "") or "",
            routes=application.routes,
        )
        _fix_openapi_file_uploads_for_swagger_ui(openapi_schema)
        application.openapi_schema = openapi_schema
        return openapi_schema

    application.openapi = custom_openapi  # type: ignore[method-assign]


def sanitize_filename(filename: str) -> str:
    """
    格式化压缩文件的文件名
    移除路径遍历字符, 保留 Unicode 字母、数字、._-
    禁止隐藏文件
    """
    sanitized = re.sub(r'[/\\\.]{2,}|[/\\]', '', filename)
    sanitized = re.sub(r'[^\w.-]', '_', sanitized, flags=re.UNICODE)
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized[1:]
    return sanitized or 'unnamed'


def _api_internal_pdf_work_name() -> str:
    """32-char hex; avoids Linux NAME_MAX / path length errors from long Unicode upload names."""
    return uuid.uuid4().hex

def cleanup_file(file_path: str) -> None:
    """清理临时 zip 文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")

def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


@app.get(path="/health-check")
async def health_check():
    """Health check endpoint for K8s readiness/liveness probes"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "mineru-api",
            "version": __version__
        }
    )


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


@app.post(path="/file_parse", dependencies=[Depends(limit_concurrency)])
async def parse_pdf(
        files: List[UploadFile] = File(..., description="Upload pdf or image files for parsing"),
        output_dir: str = Form("./output", description="Output local directory"),
        lang_list: List[str] = Form(
            ["ch"],
            description="""(Adapted only for pipeline and hybrid backend)Input the languages in the pdf to improve OCR accuracy.Options:
- ch: Chinese, English, Chinese Traditional.
- ch_lite: Chinese, English, Chinese Traditional, Japanese.
- ch_server: Chinese, English, Chinese Traditional, Japanese.
- en: English.
- korean: Korean, English.
- japan: Chinese, English, Chinese Traditional, Japanese.
- chinese_cht: Chinese, English, Chinese Traditional, Japanese.
- ta: Tamil, English.
- te: Telugu, English.
- ka: Kannada.
- th: Thai, English.
- el: Greek, English.
- latin: French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua.
- arabic: Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English.
- east_slavic: Russian, Belarusian, Ukrainian, English.
- cyrillic: Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English.
- devanagari: Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English.
"""
        ),
        backend: str = Form(
            "hybrid-auto-engine",
            description="""The backend for parsing:
- pipeline: More general, supports multiple languages, hallucination-free.
- vlm-auto-engine: High accuracy via local computing power, supports Chinese and English documents only.
- vlm-http-client: High accuracy via remote computing power(client suitable for openai-compatible servers), supports Chinese and English documents only.
- hybrid-auto-engine: Next-generation high accuracy solution via local computing power, supports multiple languages.
- hybrid-http-client: High accuracy via remote computing power but requires a little local computing power(client suitable for openai-compatible servers), supports multiple languages."""
        ),
        parse_method: str = Form(
            "auto",
            description="""(Adapted only for pipeline and hybrid backend)The method for parsing PDF:
- auto: Automatically determine the method based on the file type
- txt: Use text extraction method
- ocr: Use OCR method for image-based PDFs
- vlm: Layout via MinerU2.5, extraction via Qwen3-VL-30B (requires MINERU_VL_SERVER_EXTRACTION)
"""
        ),
        formula_enable: bool = Form(True, description="Enable formula parsing."),
        table_enable: bool = Form(True, description="Enable table parsing."),
        discarded_blocks_enable: bool = Form(
            True,
            description="When True, header/footer/page_number/aside/page_footnote go to discarded_blocks (excluded from main content). When False, include them as regular text blocks."
        ),
        server_url: Optional[str] = Form(
            None,
            description="(Adapted only for <vlm/hybrid>-http-client backend)openai compatible server url, e.g., http://127.0.0.1:30000"
        ),
        return_md: bool = Form(True, description="Return markdown content in response"),
        return_middle_json: bool = Form(False, description="Return middle JSON in response"),
        return_model_output: bool = Form(False, description="Return model output JSON in response"),
        return_content_list: bool = Form(False, description="Return content list JSON in response"),
        return_images: bool = Form(False, description="Return extracted images in response"),
        response_format_zip: bool = Form(False, description="Return results as a ZIP file instead of JSON"),
        draw_layout_bbox: bool = Form(
            False,
            description="Write {name}_layout.pdf next to outputs (type-colored layout boxes, same as CLI f_draw_layout_bbox).",
        ),
        draw_extraction_route_pdf: bool = Form(
            False,
            description="Write {name}_extraction_routes.pdf: bbox colors = pdf_text vs ocr vs qwen3_vl (hybrid + parse_method=vlm + smart routing).",
        ),
        start_page_id: int = Form(0, description="The starting page for PDF parsing, beginning from 0"),
        end_page_id: int = Form(99999, description="The ending page for PDF parsing, beginning from 0"),
):

    # 获取命令行配置参数
    config = getattr(app.state, "config", {})

    try:
        # 创建唯一的输出目录
        unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)

        # 处理上传的PDF文件（内部用短 ASCII work id 建目录，避免长阿拉伯语等文件名触发 Errno 36）
        pdf_file_names: list[str] = []
        pdf_bytes_list: list[bytes] = []
        response_keys: list[str] = []

        for idx, file in enumerate(files):
            content = await file.read()
            file_path = Path(file.filename or "upload")
            work_name = _api_internal_pdf_work_name()
            ext = file_path.suffix or ".bin"
            temp_path = Path(unique_dir) / f"{work_name}{ext}"
            with open(temp_path, "wb") as f:
                f.write(content)

            # 如果是图像文件或PDF，使用read_fn处理
            file_suffix = guess_suffix_by_path(temp_path)
            if file_suffix in pdf_suffixes + image_suffixes:
                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(work_name)
                    stem = file_path.stem or "document"
                    response_keys.append(stem if len(files) == 1 else f"{stem}__{idx}")
                    os.remove(temp_path)  # 删除临时文件
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"}
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_suffix}"}
                )


        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)

        # 调用异步处理函数
        await aio_do_parse(
            output_dir=unique_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            discarded_blocks_enable=discarded_blocks_enable,
            server_url=server_url,
            f_draw_layout_bbox=draw_layout_bbox,
            f_draw_span_bbox=False,
            f_draw_extraction_route_pdf=draw_extraction_route_pdf,
            f_dump_md=return_md,
            f_dump_middle_json=return_middle_json,
            f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=return_content_list,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **config
        )

        # 根据 response_format_zip 决定返回类型
        if response_format_zip:
            zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="mineru_results_")
            os.close(zip_fd)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for resp_key, pdf_name in zip(response_keys, pdf_file_names):
                    safe_pdf_name = sanitize_filename(resp_key)

                    if backend.startswith("pipeline"):
                        parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                    elif backend.startswith("vlm"):
                        parse_dir = os.path.join(unique_dir, pdf_name, "vlm")
                    elif backend.startswith("hybrid"):
                        parse_dir = os.path.join(unique_dir, pdf_name, f"hybrid_{parse_method}")
                    else:
                        parse_dir = os.path.join(unique_dir, pdf_name, f"hybrid_{parse_method}")

                    if not os.path.exists(parse_dir):
                        continue

                    # 写入文本类结果
                    if return_md:
                        path = os.path.join(parse_dir, f"{pdf_name}.md")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}.md"))

                    if return_middle_json:
                        path = os.path.join(parse_dir, f"{pdf_name}_middle.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_middle.json"))

                    if return_model_output:
                        path = os.path.join(parse_dir, f"{pdf_name}_model.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, os.path.basename(path)))

                    if return_content_list:
                        path = os.path.join(parse_dir, f"{pdf_name}_content_list.json")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_content_list.json"))

                    if draw_layout_bbox:
                        path = os.path.join(parse_dir, f"{pdf_name}_layout.pdf")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{pdf_name}_layout.pdf"))

                    if draw_extraction_route_pdf:
                        path = os.path.join(parse_dir, f"{pdf_name}_extraction_routes.pdf")
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.join(safe_pdf_name, f"{pdf_name}_extraction_routes.pdf"))

                    # 写入图片
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        image_paths = glob.glob(os.path.join(glob.escape(images_dir), "*.jpg"))
                        for image_path in image_paths:
                            zf.write(image_path, arcname=os.path.join(safe_pdf_name, "images", os.path.basename(image_path)))

            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename="results.zip",
                background=BackgroundTask(cleanup_file, zip_path)
            )
        else:
            # 构建 JSON 结果（键为上传时的逻辑名；磁盘目录仍用短 work id）
            result_dict = {}
            for resp_key, pdf_name in zip(response_keys, pdf_file_names):
                result_dict[resp_key] = {}
                data = result_dict[resp_key]

                if backend.startswith("pipeline"):
                    parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
                elif backend.startswith("vlm"):
                    parse_dir = os.path.join(unique_dir, pdf_name, "vlm")
                elif backend.startswith("hybrid"):
                    parse_dir = os.path.join(unique_dir, pdf_name, f"hybrid_{parse_method}")
                else:
                    parse_dir = os.path.join(unique_dir, pdf_name, f"hybrid_{parse_method}")

                if os.path.exists(parse_dir):
                    if return_md:
                        data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
                    if return_middle_json:
                        data["middle_json"] = get_infer_result("_middle.json", pdf_name, parse_dir)
                    if return_model_output:
                        data["model_output"] = get_infer_result("_model.json", pdf_name, parse_dir)
                    if return_content_list:
                        data["content_list"] = get_infer_result("_content_list.json", pdf_name, parse_dir)
                    if return_images:
                        images_dir = os.path.join(parse_dir, "images")
                        safe_pattern = os.path.join(glob.escape(images_dir), "*.jpg")
                        image_paths = glob.glob(safe_pattern)
                        data["images"] = {
                            os.path.basename(
                                image_path
                            ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                            for image_path in image_paths
                        }

            return JSONResponse(
                status_code=200,
                content={
                    "backend": backend,
                    "version": __version__,
                    "results": result_dict
                }
            )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file: {str(e)}"}
        )


_install_custom_openapi(app)


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
@click.option('--port', default=8080, type=int, help='Server port (default: 8080)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
def main(ctx, host, port, reload, **kwargs):

    kwargs.update(arg_parse(ctx))

    # 将配置参数存储到应用状态中
    app.state.config = kwargs

    # 将 CLI 的并发参数同步到环境变量，确保 uvicorn 重载子进程可见
    try:
        mcr = int(kwargs.get("mineru_api_max_concurrent_requests", 0) or 0)
    except ValueError:
        mcr = 0
    os.environ["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(mcr)

    """启动MinerU FastAPI服务器的命令行入口"""
    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")

    uvicorn.run(
        "mineru.cli.fast_api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()
