#!/usr/bin/env bash
# Compare MinerU VLM extraction with different concurrency and batch settings.
# Restarts the container with each config, runs the test, and reports timing.
#
# Usage: ./test-vlm-batch-compare.sh
# Requires: MinerU API running via docker compose (will restart with different env)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINERU_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PDF_PATH="$MINERU_ROOT/assets/Tourism Monthly Report - Q1 2025.pdf"
API_URL="${MINERU_API_URL:-http://localhost:8085}"

if [[ ! -f "$PDF_PATH" ]]; then
  echo "ERROR: PDF not found: $PDF_PATH"
  exit 1
fi

echo "=========================================="
echo "MinerU VLM Batch/Concurrency Comparison"
echo "=========================================="
echo "PDF: $PDF_PATH"
echo ""

# Test configs: (CONCURRENCY, BATCH_SIZE, description)
# BATCH_SIZE=0 means no chunking (original behavior)
configs=(
  "12:8:concurrency=12 batch=8 (default)"
  "12:0:concurrency=12 batch=0 (no chunking)"
  "8:8:concurrency=8 batch=8"
  "16:8:concurrency=16 batch=8"
)

for cfg in "${configs[@]}"; do
  IFS=':' read -r CONC BATCH DESC <<< "$cfg"
  echo "------------------------------------------"
  echo "Test: $DESC"
  echo "------------------------------------------"

  # Restart container with this config
  cd "$SCRIPT_DIR"
  export MINERU_SPLIT_TAG="${MINERU_SPLIT_TAG:-$(date +%d-%m-%Y)}"
  export MINERU_VL_MAX_CONCURRENCY=$CONC
  export MINERU_VL_EXTRACTION_BATCH_SIZE=$BATCH
  docker compose -f docker-compose.api-split-vlm.yml up -d --force-recreate 2>/dev/null || true

  # Wait for API to be ready
  for i in {1..30}; do
    if curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs" 2>/dev/null | grep -q 200; then
      break
    fi
    sleep 2
  done

  # Run test
  START=$(date +%s.%N)
  HTTP=$(curl -s -w "%{http_code}" -o /tmp/mineru-test-response.zip \
    -X POST "$API_URL/file_parse" \
    -F "files=@$PDF_PATH" \
    -F "output_dir=/tmp/mineru-out" \
    -F "lang_list=en" \
    -F "backend=hybrid-http-client" \
    -F "parse_method=vlm" \
    -F "formula_enable=true" \
    -F "table_enable=true" \
    -F "return_md=true" \
    -F "response_format_zip=true" \
    --max-time 600 2>/dev/null)
  END=$(date +%s.%N)
  ELAPSED=$(echo "$END - $START" | bc)

  echo "  HTTP: $HTTP | Elapsed: ${ELAPSED}s"
  echo ""
done

echo "=========================================="
echo "Done. Check docker logs for extraction details:"
echo "  docker logs mineru-api-split-vlm"
echo "=========================================="
