"""
SSL Verification Patch for MinerU VLM Client

This patch disables SSL certificate verification for MinerU's HTTP VLM client,
allowing it to connect to servers with self-signed certificates.

Usage: Import this module before using MinerU's API.
"""
import os
import httpx

# Only patch if SSL verification is disabled
if os.environ.get('MINERU_SSL_VERIFY', 'true').lower() in ('false', '0', 'no', 'off'):
    print("[SSL Patch] Patching MinerU VLM client to disable SSL verification...")
    
    try:
        from mineru_vl_utils.vlm_client import http_client
        from mineru_vl_utils.vlm_client.http_client import RetryTransport, Retry, HTTPMethod
        
        # Store original method
        _original_new_client = http_client.HttpVlmClient._new_client
        
        def _patched_new_client(self) -> httpx.Client:
            """Patched version that disables SSL verification."""
            return httpx.Client(
                headers=self.server_headers,
                verify=False,  # <-- THIS IS THE KEY CHANGE
                timeout=httpx.Timeout(
                    connect=self.connect_timeout,
                    read=self.http_timeout,
                    write=self.http_timeout,
                    pool=None,
                ),
                transport=RetryTransport(
                    retry=Retry(
                        total=self.max_retries,
                        backoff_factor=self.retry_backoff_factor,
                        allowed_methods=list(HTTPMethod),
                    ),
                    transport=httpx.HTTPTransport(
                        verify=False,  # Also disable for transport
                        limits=httpx.Limits(
                            max_connections=self.max_connections,
                            max_keepalive_connections=self.max_keepalive_connections,
                            keepalive_expiry=self.keepalive_expiry,
                        ),
                    ),
                ),
            )
        
        # Apply patch
        http_client.HttpVlmClient._new_client = _patched_new_client
        
        # Also patch async client
        _original_new_aio_client = http_client.HttpVlmClient._new_aio_client
        
        async def _patched_new_aio_client(self) -> httpx.AsyncClient:
            """Patched version that disables SSL verification for async client."""
            return httpx.AsyncClient(
                headers=self.server_headers,
                verify=False,  # <-- THIS IS THE KEY CHANGE
                timeout=httpx.Timeout(
                    connect=self.connect_timeout,
                    read=self.http_timeout,
                    write=self.http_timeout,
                    pool=None,
                ),
                transport=RetryTransport(
                    retry=Retry(
                        total=self.max_retries,
                        backoff_factor=self.retry_backoff_factor,
                        allowed_methods=list(HTTPMethod),
                    ),
                    transport=httpx.AsyncHTTPTransport(
                        verify=False,  # Also disable for transport
                        limits=httpx.Limits(
                            max_connections=self.max_connections,
                            max_keepalive_connections=self.max_keepalive_connections,
                            keepalive_expiry=self.keepalive_expiry,
                        ),
                    ),
                ),
            )
        
        http_client.HttpVlmClient._new_aio_client = _patched_new_aio_client
        
        print("[SSL Patch] Successfully patched MinerU VLM client!")
        
    except Exception as e:
        print(f"[SSL Patch] Warning: Could not patch MinerU VLM client: {e}")
else:
    print("[SSL Patch] SSL verification enabled, no patching needed.")
