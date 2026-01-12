#!/bin/bash
# Patch mineru_vl_utils to disable SSL verification and fix URL path stripping

set -e

HTTPX_CLIENT_FILE=$(python3 -c "import mineru_vl_utils.vlm_client.http_client as m; import os; print(os.path.dirname(m.__file__))")/http_client.py

echo "Patching $HTTPX_CLIENT_FILE..."

# Check if already patched  
if grep -q "HTTPTransport(verify=False" "$HTTPX_CLIENT_FILE" && grep -q "PATCHED: Keep full URL" "$HTTPX_CLIENT_FILE"; then
    echo "✓ Already patched (SSL + URL fix), skipping"
    exit 0
fi

# Backup original file
cp "$HTTPX_CLIENT_FILE" "${HTTPX_CLIENT_FILE}.bak"

# Add verify=False to httpx.Client calls
sed -i 's/return httpx\.Client(/return httpx.Client(\n            verify=False,/' "$HTTPX_CLIENT_FILE"
sed -i 's/return httpx\.AsyncClient(/return httpx.AsyncClient(\n            verify=False,/' "$HTTPX_CLIENT_FILE"

# Add verify=False to HTTPTransport calls (inside RetryTransport)
sed -i 's/httpx\.HTTPTransport(/httpx.HTTPTransport(verify=False, /' "$HTTPX_CLIENT_FILE"
sed -i 's/httpx\.AsyncHTTPTransport(/httpx.AsyncHTTPTransport(verify=False, /' "$HTTPX_CLIENT_FILE"

# Fix URL path stripping: Replace the problematic if/else with simpler logic
# Find the lines that handle server_url and replace them
sed -i '/if server_url\.endswith("\/"):/,/server_url = self\._get_base_url(server_url)/ {
    /if server_url\.endswith("\/"):/c\        # PATCHED: Keep full URL path, just strip trailing slash\n        if server_url.endswith("/"):
    /else:/,/server_url = self\._get_base_url(server_url)/ d
}' "$HTTPX_CLIENT_FILE"

# Add environment variable support for max_concurrency and http_timeout
# Replace: max_concurrency: int = 100,
# With: max_concurrency: int = int(os.getenv("MINERU_VLM_MAX_CONCURRENCY", "100")),
sed -i 's/max_concurrency: int = 100,/max_concurrency: int = int(os.getenv("MINERU_VLM_MAX_CONCURRENCY", "100")),/' "$HTTPX_CLIENT_FILE"

# Replace: http_timeout: int = 600,
# With: http_timeout: int = int(os.getenv("MINERU_VLM_HTTP_TIMEOUT", "600")),
sed -i 's/http_timeout: int = 600,/http_timeout: int = int(os.getenv("MINERU_VLM_HTTP_TIMEOUT", "600")),/' "$HTTPX_CLIENT_FILE"

echo "✓ Patched httpx clients to disable SSL verification"
echo "✓ Fixed URL path preservation (removed _get_base_url() stripping)"
echo "✓ Added environment variable support for max_concurrency and http_timeout"

# Verify the patch
if grep -q "verify=False" "$HTTPX_CLIENT_FILE" && grep -q "HTTPTransport(verify=False" "$HTTPX_CLIENT_FILE" && grep -q "PATCHED: Keep full URL" "$HTTPX_CLIENT_FILE"; then
    echo "✓ All patches applied successfully"
else
    echo "✗ Patch failed to apply"
    cp "${HTTPX_CLIENT_FILE}.bak" "$HTTPX_CLIENT_FILE"
    exit 1
fi
