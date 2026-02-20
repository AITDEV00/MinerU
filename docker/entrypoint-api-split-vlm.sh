#!/bin/bash
set -e
# Use compose env; default to huggingface so OCR models download when smart routing needs them
export MINERU_MODEL_SOURCE="${MINERU_MODEL_SOURCE:-huggingface}"
export MINERU_DEFAULT_BACKEND="${MINERU_DEFAULT_BACKEND:-hybrid-http-client}"
[[ "${MINERU_SSL_VERIFY}" != "true" ]] && export PYTHONHTTPSVERIFY=0 CURL_CA_BUNDLE="" REQUESTS_CA_BUNDLE="" SSL_CERT_FILE="" SSL_CERT_VERIFICATION=0
[[ -n "${MINERU_VL_API_KEY}" ]] && export OPENAI_API_KEY="${MINERU_VL_API_KEY}"
[[ -n "${MINERU_VL_SERVER}" ]] && export OPENAI_BASE_URL="${MINERU_VL_SERVER}"

# Download MinerU models on first start (into mounted volume; skipped if already present)
if ! mineru-models-download -s huggingface -m all 2>/dev/null; then
  echo "Warning: mineru-models-download failed or models already present, continuing..."
fi

# VLM client concurrency: 24 for 2g.35GB (safe), 32 max.
MAX_CONCURRENCY="${MINERU_VL_MAX_CONCURRENCY:-24}"
exec mineru-api --host 0.0.0.0 --port 8000 --max-concurrency "$MAX_CONCURRENCY"
