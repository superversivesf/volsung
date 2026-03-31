#!/bin/bash
# Start Music Service
# 
# Standalone music generation service using MusicGen via audiocraft.
# Runs on port 8002 and provides HTTP API for music generation.
#
# Requirements:
#   - Python 3.9+
#   - Virtual environment with dependencies installed
# 
# Usage:
#   ./scripts/start-music.sh [--venv PATH] [--host HOST] [--port PORT] [--reload]
#
# Examples:
#   ./scripts/start-music.sh                    # Use default .venv-music
#   ./scripts/start-music.sh --venv .venv      # Use custom venv
#   ./scripts/start-music.sh --reload         # Enable auto-reload for development
#   ./scripts/start-music.sh --port 8003       # Run on different port

set -e

# Default configuration
VENV_PATH="${VENV_PATH:-.venv-music}"
HOST="${MUSIC_HOST:-0.0.0.0}"
PORT="${MUSIC_PORT:-8002}"
RELOAD=false
WORKERS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --venv PATH    Path to virtual environment (default: .venv-music)"
            echo "  --host HOST    Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT    Port to listen on (default: 8002)"
            echo "  --reload       Enable auto-reload for development"
            echo "  --workers N    Number of worker processes (default: 1)"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [[ ! -d "$VENV_PATH" ]]; then
    echo "Virtual environment not found: $VENV_PATH"
    echo ""
    echo "To create the virtual environment:"
    echo "  python -m venv $VENV_PATH"
    echo "  source $VENV_PATH/bin/activate"
    echo "  pip install -r requirements/services/music.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Verify dependencies are installed
if ! python -c "import audiocraft" 2>/dev/null; then
    echo "Warning: audiocraft not found. Installing dependencies..."
    pip install -r requirements/services/music.txt
fi

# Log startup info
echo "=========================================="
echo "Starting Volsung Music Service"
echo "=========================================="
echo "Host:     $HOST"
echo "Port:     $PORT"
echo "Venv:     $VENV_PATH"
echo "Reload:   $RELOAD"
echo "Workers:  $WORKERS"
echo "=========================================="
echo ""
echo "API Endpoints:"
echo "  GET  http://$HOST:$PORT/health"
echo "  GET  http://$HOST:$PORT/info"
echo "  POST http://$HOST:$PORT/music/generate"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Set environment variables for the service
export MUSIC_SERVICE_PORT="$PORT"
export MUSIC_SERVICE_HOST="$HOST"

# Run the service
if [[ "$RELOAD" == true ]]; then
    uvicorn volsung.services.music_service:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level info
else
    uvicorn volsung.services.music_service:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info
fi
