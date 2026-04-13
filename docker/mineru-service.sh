#!/usr/bin/env bash
# MinerU service control: up, down, status
# Usage: ./mineru-service.sh [up|down|status]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-status}" in
  up)
    export MINERU_SPLIT_TAG="${MINERU_SPLIT_TAG:-$(date +%d-%m-%Y)}"
    cd "$SCRIPT_DIR" && docker compose -f docker-compose.api-split-vlm.yml up -d
    echo "MinerU API: http://localhost:8085/docs"
    ;;
  down)
    cd "$SCRIPT_DIR" && docker compose -f docker-compose.api-split-vlm.yml down
    echo ""
    echo "Checking port 8085..."
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8085/docs 2>/dev/null | grep -q 200; then
      echo "WARNING: Port 8085 still responds! Orphaned processes may exist."
      echo "Run: sudo kill \$(pgrep -f 'mineru-api.*8000' || true) \$(pgrep -f 'docker-proxy.*8085' || true)"
    else
      echo "Port 8085 is free."
    fi
    ;;
  status)
    echo "=== MinerU service status ==="
    echo ""
    echo "Docker container:"
    docker ps -a --filter "name=mineru" --format "  {{.Names}}: {{.Status}} {{.Ports}}" 2>/dev/null || echo "  (none)"
    echo ""
    echo "Port 8085:"
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8085/docs 2>/dev/null | grep -q 200; then
      echo "  HTTP 200 - API is UP"
    else
      echo "  Not responding - API is DOWN"
    fi
    echo ""
    echo "Processes (may show orphaned if docker lost track):"
    ps aux 2>/dev/null | grep -E "mineru-api|docker-proxy.*8085" | grep -v grep || echo "  (none)"
    ;;
  *)
    echo "Usage: $0 [up|down|status]"
    exit 1
    ;;
esac
