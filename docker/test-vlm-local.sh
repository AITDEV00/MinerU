#!/usr/bin/env bash
# Test MinerU VLM parsing locally
# Usage: ./test-vlm-local.sh [PDF_PATH] [START_PAGE] [END_PAGE]
#        ./test-vlm-local.sh --concurrent N [PDF_PATH] [START_PAGE] [END_PAGE]
# Example: ./test-vlm-local.sh "assets/DGE GovAI - Use Case Narrative v3.0.pdf" 0 4
# Example: ./test-vlm-local.sh --concurrent 3  (run 3 requests in parallel)
# START_PAGE/END_PAGE: 0-based, optional; limit pages for large PDFs

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINERU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_URL="${MINERU_API_URL:-http://localhost:8085}"

# --concurrent N: run N parallel requests
CONCURRENT_JOBS=0
if [[ "$1" == "--concurrent" || "$1" == "-j" ]]; then
  CONCURRENT_JOBS="${2:-2}"
  shift 2
  # Spawn N parallel runs and wait
  echo "Launching $CONCURRENT_JOBS concurrent requests..."
  BASE_ARGS=("$@")
  for i in $(seq 1 "$CONCURRENT_JOBS"); do
    MINERU_TEST_ID=$i "$0" "${BASE_ARGS[@]}" > "/tmp/mineru-concurrent-$i.log" 2>&1 &
  done
  wait
  echo ""
  echo "=== Summary (${CONCURRENT_JOBS} concurrent requests) ==="
  for i in $(seq 1 "$CONCURRENT_JOBS"); do
    echo "--- Job $i ---"
    grep -E "HTTP Status|Elapsed:|Saved to:" "/tmp/mineru-concurrent-$i.log" 2>/dev/null || tail -5 "/tmp/mineru-concurrent-$i.log"
  done
  exit 0
fi

OUTPUT_DIR="/tmp/mineru-test-$(date +%s)${MINERU_TEST_ID:+-$MINERU_TEST_ID}"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# PDF: arg1 or default
if [[ -n "$1" && "$1" != [0-9]* ]]; then
  PDF_ARG="$1"
  [[ "$PDF_ARG" != /* ]] && PDF_PATH="$MINERU_ROOT/$PDF_ARG" || PDF_PATH="$PDF_ARG"
  shift
else
  PDF_PATH="$MINERU_ROOT/assets/DGE EC Retreat_Day 1_VShared_V2.0.pdf"
fi

if [[ ! -f "$PDF_PATH" ]]; then
  echo "ERROR: PDF not found: $PDF_PATH"
  exit 1
fi

START_PAGE="${1:-0}"
END_PAGE="${2:-99999}"

echo "=========================================="
echo "MinerU VLM Local Test${MINERU_TEST_ID:+ #$MINERU_TEST_ID}"
echo "=========================================="
echo "PDF: $PDF_PATH"
echo "Pages: $START_PAGE–$END_PAGE (0-based)"
echo "API: $API_URL"
echo "Output: $OUTPUT_DIR"
echo "parse_method: vlm | table_enable: true | formula_enable: true"
echo "=========================================="

echo ""
echo "Starting parse (may take 2-5 min for multi-page PDF)..."
echo ""

# Build form data for curl - API expects multipart/form-data
START_TIME=$(date +%s)
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$OUTPUT_DIR/response.zip" \
  -X POST "$API_URL/file_parse" \
  -F "files=@$PDF_PATH" \
  -F "output_dir=/tmp/mineru-out" \
  -F "lang_list=en" \
  -F "backend=hybrid-http-client" \
  -F "parse_method=vlm" \
  -F "formula_enable=true" \
  -F "table_enable=true" \
  -F "discarded_blocks_enable=false" \
  -F "return_md=true" \
  -F "return_images=true" \
  -F "response_format_zip=true" \
  -F "start_page_id=$START_PAGE" \
  -F "end_page_id=$END_PAGE" \
  --max-time 900 \
  -H "Accept: application/zip" 2>/dev/null) || HTTP_CODE="000"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "HTTP Status: $HTTP_CODE"
echo "Elapsed: ${ELAPSED}s"

if [[ "$HTTP_CODE" == "200" ]]; then
  SIZE=$(stat -c%s "$OUTPUT_DIR/response.zip" 2>/dev/null || stat -f%z "$OUTPUT_DIR/response.zip" 2>/dev/null)
  echo "Response size: $SIZE bytes (ZIP)"
  echo "Saved to: $OUTPUT_DIR/response.zip"
else
  echo "Response (first 500 chars):"
  head -c 500 "$OUTPUT_DIR/response.zip" 2>/dev/null || true
  echo ""
fi

echo "=========================================="
