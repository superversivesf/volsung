#!/bin/bash
# Start all Volsung services in the correct order
#
# This script starts all microservices for the Volsung audio generation system:
# 1. TTS Service (port 8001) - Text-to-Speech generation
# 2. Music Service (port 8002) - Music generation
# 3. SFX Service (port 8003) - Sound effects generation
# 4. Coordinator (port 8000) - API Gateway and routing
#
# Usage:
#   ./scripts/start-all.sh              # Start all services
#   ./scripts/start-all.sh --daemon     # Run in background
#   ./scripts/start-all.sh --log-dir /var/log/volsung  # Custom log directory
#
# Environment Variables:
#   VOLSUNG_LOG_DIR    - Log directory (default: ./logs)
#   TTS_SERVICE_PORT   - TTS service port (default: 8001)
#   MUSIC_SERVICE_PORT - Music service port (default: 8002)
#   SFX_SERVICE_PORT   - SFX service port (default: 8003)
#   COORDINATOR_PORT   - Coordinator port (default: 8000)
#   SKIP_SERVICES      - Comma-separated list of services to skip
#
# Example:
#   SKIP_SERVICES=music ./scripts/start-all.sh  # Skip music service

set -e

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${VOLSUNG_LOG_DIR:-$PROJECT_DIR/logs}"
PID_FILE="$PROJECT_DIR/.volsung-pids"

# Service ports
TTS_PORT="${TTS_SERVICE_PORT:-8001}"
MUSIC_PORT="${MUSIC_SERVICE_PORT:-8002}"
SFX_PORT="${SFX_SERVICE_PORT:-8003}"
COORD_PORT="${COORDINATOR_PORT:-8000}"

# Parse skip list
SKIP_LIST="${SKIP_SERVICES:-}"
should_skip() {
    local service="$1"
    if [[ ",$SKIP_LIST," == *",$service,"* ]]; then
        return 0
    fi
    return 1
}

# Daemon mode
DAEMON_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon|-d)
            DAEMON_MODE=true
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Start all Volsung services."
            echo ""
            echo "Options:"
            echo "  --daemon, -d        Run services in background (daemon mode)"
            echo "  --log-dir PATH      Custom log directory (default: ./logs)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  VOLSUNG_LOG_DIR     Log directory"
            echo "  SKIP_SERVICES       Comma-separated services to skip (e.g., 'music,sfx')"
            echo "  TTS_SERVICE_PORT    TTS service port (default: 8001)"
            echo "  MUSIC_SERVICE_PORT  Music service port (default: 8002)"
            echo "  SFX_SERVICE_PORT    SFX service port (default: 8003)"
            echo "  COORDINATOR_PORT    Coordinator port (default: 8000)"
            echo ""
            echo "Examples:"
            echo "  $0                  # Start all services in foreground"
            echo "  $0 --daemon         # Start all services in background"
            echo "  SKIP_SERVICES=music $0  # Start all except music"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Helper Functions
# ==============================================================================

check_port() {
    local port=$1
    if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

wait_for_service() {
    local name="$1"
    local port="$2"
    local timeout="${3:-60}"
    local start_time=$(date +%s)

    echo -n "Waiting for $name on port $port..."
    
    while ! curl -s "http://localhost:$port/health" >/dev/null 2>&1; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -gt $timeout ]]; then
            echo " TIMEOUT"
            return 1
        fi
        
        echo -n "."
        sleep 1
    done
    
    echo " OK ($(( $(date +%s) - start_time ))s)"
    return 0
}

kill_service() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
}

cleanup() {
    echo ""
    echo "Shutting down services..."
    
    if [[ -f "$PID_FILE" ]]; then
        while read -r service pid; do
            echo "Stopping $service (PID $pid)..."
            kill_service "$pid"
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    
    echo "All services stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# ==============================================================================
# Main
# ==============================================================================

echo "========================================"
echo "  Volsung Service Launcher"
echo "========================================"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Check which services to start
echo "Service Configuration:"
should_skip "tts" && echo "  TTS Service:     SKIPPED (port $TTS_PORT)" || echo "  TTS Service:     port $TTS_PORT"
should_skip "music" && echo "  Music Service:   SKIPPED (port $MUSIC_PORT)" || echo "  Music Service:   port $MUSIC_PORT"
should_skip "sfx" && echo "  SFX Service:     SKIPPED (port $SFX_PORT)" || echo "  SFX Service:     port $SFX_PORT"
echo "  Coordinator:     port $COORD_PORT"
echo ""

# Check if ports are already in use
CONFLICTS=0
check_port $TTS_PORT && echo "ERROR: Port $TTS_PORT (TTS) is already in use" && CONFLICTS=$((CONFLICTS + 1))
check_port $MUSIC_PORT && echo "ERROR: Port $MUSIC_PORT (Music) is already in use" && CONFLICTS=$((CONFLICTS + 1))
check_port $SFX_PORT && echo "ERROR: Port $SFX_PORT (SFX) is already in use" && CONFLICTS=$((CONFLICTS + 1))
check_port $COORD_PORT && echo "ERROR: Port $COORD_PORT (Coordinator) is already in use" && CONFLICTS=$((CONFLICTS + 1))

if [[ $CONFLICTS -gt 0 ]]; then
    echo ""
    echo "Please stop conflicting services or change ports."
    exit 1
fi

echo "Log directory: $LOG_DIR"
echo ""

# Activate virtual environment if it exists
if [[ -d "$PROJECT_DIR/.venv" ]]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
elif [[ -d "$PROJECT_DIR/venv" ]]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Clear PID file
rm -f "$PID_FILE"

# Start TTS Service
if ! should_skip "tts"; then
    echo "Starting TTS Service..."
    if [[ "$DAEMON_MODE" == true ]]; then
        TTS_SERVICE_PORT=$TTS_PORT nohup "$SCRIPT_DIR/start-tts.sh" > "$LOG_DIR/tts.log" 2>&1 &
        echo "tts $!" >> "$PID_FILE"
    else
        TTS_SERVICE_PORT=$TTS_PORT "$SCRIPT_DIR/start-tts.sh" &
        echo "tts $!" >> "$PID_FILE"
    fi
    
    if ! wait_for_service "TTS" "$TTS_PORT" 120; then
        echo "ERROR: TTS Service failed to start"
        exit 1
    fi
    echo ""
fi

# Start Music Service
if ! should_skip "music"; then
    echo "Starting Music Service..."
    if [[ "$DAEMON_MODE" == true ]]; then
        MUSIC_SERVICE_PORT=$MUSIC_PORT nohup "$SCRIPT_DIR/start-music.sh" > "$LOG_DIR/music.log" 2>&1 &
        echo "music $!" >> "$PID_FILE"
    else
        MUSIC_SERVICE_PORT=$MUSIC_PORT "$SCRIPT_DIR/start-music.sh" &
        echo "music $!" >> "$PID_FILE"
    fi
    
    if ! wait_for_service "Music" "$MUSIC_PORT" 120; then
        echo "ERROR: Music Service failed to start"
        exit 1
    fi
    echo ""
fi

# Start SFX Service
if ! should_skip "sfx"; then
    echo "Starting SFX Service..."
    if [[ "$DAEMON_MODE" == true ]]; then
        SFX_SERVICE_PORT=$SFX_PORT nohup "$SCRIPT_DIR/start-sfx.sh" > "$LOG_DIR/sfx.log" 2>&1 &
        echo "sfx $!" >> "$PID_FILE"
    else
        SFX_SERVICE_PORT=$SFX_PORT "$SCRIPT_DIR/start-sfx.sh" &
        echo "sfx $!" >> "$PID_FILE"
    fi
    
    if ! wait_for_service "SFX" "$SFX_PORT" 120; then
        echo "ERROR: SFX Service failed to start"
        exit 1
    fi
    echo ""
fi

# Start Coordinator
echo "Starting Coordinator..."
cd "$PROJECT_DIR"

# Wait a moment for services to be ready
sleep 2

# Set environment for coordinator
export TTS_SERVICE_URL="http://localhost:$TTS_PORT"
export MUSIC_SERVICE_URL="http://localhost:$MUSIC_PORT"
export SFX_SERVICE_URL="http://localhost:$SFX_PORT"
export COORDINATOR_PORT="$COORD_PORT"

if [[ "$DAEMON_MODE" == true ]]; then
    nohup python -m volsung > "$LOG_DIR/coordinator.log" 2>&1 &
    COORD_PID=$!
    echo "coordinator $COORD_PID" >> "$PID_FILE"
else
    python -m volsung &
    COORD_PID=$!
    echo "coordinator $COORD_PID" >> "$PID_FILE"
fi

if ! wait_for_service "Coordinator" "$COORD_PORT" 30; then
    echo "ERROR: Coordinator failed to start"
    exit 1
fi

echo ""
echo "========================================"
echo "  All services started successfully!"
echo "========================================"
echo ""
echo "Access the coordinator at:"
echo "  http://localhost:$COORD_PORT"
echo ""
echo "Health check:"
echo "  curl http://localhost:$COORD_PORT/health"
echo ""

if [[ "$DAEMON_MODE" == true ]]; then
    echo "Services are running in background."
    echo "Logs: $LOG_DIR"
    echo "PIDs: $PID_FILE"
    echo ""
    echo "To stop:"
    echo "  kill $(cat $PID_FILE | awk '{print $2}')"
else
    echo "Services are running. Press Ctrl+C to stop."
    echo ""
    
    # Wait for coordinator to finish
    wait $COORD_PID
fi
