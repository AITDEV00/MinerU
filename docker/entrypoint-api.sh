#!/bin/bash
set -e
export MINERU_MODEL_SOURCE=local
export MINERU_DEFAULT_BACKEND="${MINERU_DEFAULT_BACKEND:-hybrid-http-client}"
[[ "${MINERU_SSL_VERIFY}" != "true" ]] && export PYTHONHTTPSVERIFY=0 CURL_CA_BUNDLE="" REQUESTS_CA_BUNDLE="" SSL_CERT_FILE="" SSL_CERT_VERIFICATION=0
[[ -n "${MINERU_VL_API_KEY}" ]] && export OPENAI_API_KEY="${MINERU_VL_API_KEY}"
[[ -n "${MINERU_VL_SERVER}" ]] && export OPENAI_BASE_URL="${MINERU_VL_SERVER}"

# VLM client concurrency: 24 for 2g.35GB (safe), 32 max. Override via MINERU_VL_MAX_CONCURRENCY.
MAX_CONCURRENCY="${MINERU_VL_MAX_CONCURRENCY:-24}"
exec mineru-api --host 0.0.0.0 --port 8000 --max-concurrency "$MAX_CONCURRENCY"
