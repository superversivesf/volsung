#!/bin/bash
# Start the Volsung TTS Service
#
# Usage:
#   ./scripts/start-tts.sh              # Start with default settings
#   ./scripts/start-tts.sh --port 8002  # Start on custom port
#   PORT=8002 ./scripts/start-tts.sh    # Use environment variable
#
# Environment Variables:
#   TTS_SERVICE_HOST    - Server bind address (default: 0.0.0.0)
#   TTS_SERVICE_PORT    - Server port (default: 8001)
#   TTS_IDLE_TIMEOUT    - Seconds before unloading models (default: 300)
#   TTS_DEVICE          - Device override (default: auto-detected)
#   TTS_DTYPE           - Data type (default: bfloat16 for CUDA, float32 otherwise)
#   LOG_LEVEL           - Logging level (default: info)

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
PORT="${TTS_SERVICE_PORT:-8001}"
HOST="${TTS_SERVICE_HOST:-0.0.0.0}"
LOG_LEVEL="${LOG_LEVEL:-info}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT         Server port (default: 8001)"
            echo "  --host HOST         Server bind address (default: 0.0.0.0)"
            echo "  --log-level LEVEL   Logging level (default: info)"
            echo "  --help              Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  TTS_SERVICE_PORT    Server port"
            echo "  TTS_SERVICE_HOST    Server bind address"
            echo "  TTS_IDLE_TIMEOUT    Model idle timeout in seconds"
            echo "  TTS_DEVICE          Device override (cuda, cpu, mps)"
            echo "  TTS_DTYPE           Data type (bfloat16, float16, float32)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Export settings
export TTS_SERVICE_HOST="$HOST"
export TTS_SERVICE_PORT="$PORT"

echo "========================================"
echo "  Volsung TTS Service"
echo "========================================"
echo ""
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Log Level: $LOG_LEVEL"
echo ""
echo "  Endpoints:"
echo "    GET  /health            - Health check"
echo "    POST /voice/design      - Generate voice from text"
echo "    POST /voice/synthesize  - Clone voice"
echo ""
echo "========================================"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, torch, soundfile, pydantic" 2>/dev/null || {
    echo ""
    echo "Warning: Some dependencies may be missing."
    echo "Install with: pip install -r requirements/services/tts.txt"
    echo ""
}

# Start the service
echo "Starting TTS Service..."
echo ""

exec uvicorn volsung.services.tts_service:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --access-log \
    "$@"
