#!/bin/bash
# Start all Volsung microservices
#
# This script starts all 6 microservices for the Volsung audio generation system:
# 1. Coordinator (port 8000) - API Gateway and routing
# 2. Qwen Voice Service (port 8001) - Voice cloning and voice design
# 3. Qwen Base Service (port 8002) - Base TTS generation
# 4. StyleTTS2 Service (port 8003) - High-quality neural TTS
# 5. Music Service (port 8004) - Music generation
# 6. SFX Service (port 8005) - Sound effects generation
#
# Usage:
#   ./scripts/start-all.sh              # Start all services
#   ./scripts/start-all.sh --daemon     # Run in background
#   ./scripts/start-all.sh --docker     # Start using Docker Compose
#   ./scripts/start-all.sh --log-dir /var/log/volsung  # Custom log directory
#
# Environment Variables:
#   VOLSUNG_LOG_DIR        - Log directory (default: ./logs)
#   COORDINATOR_PORT       - Coordinator port (default: 8000)
#   QWEN_VOICE_PORT        - Qwen Voice port (default: 8001)
#   QWEN_BASE_PORT         - Qwen Base port (default: 8002)
#   STYLETTS_PORT          - StyleTTS2 port (default: 8003)
#   MUSIC_PORT             - Music port (default: 8004)
#   SFX_PORT               - SFX port (default: 8005)
#   SKIP_SERVICES          - Comma-separated list of services to skip
#
# Example:
#   SKIP_SERVICES=music,sfx ./scripts/start-all.sh  # Skip music and sfx services

set -e

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${VOLSUNG_LOG_DIR:-$PROJECT_DIR/logs}"
PID_FILE="$PROJECT_DIR/.volsung-pids"

# Service ports
COORD_PORT="${COORDINATOR_PORT:-8000}"
QWEN_VOICE_PORT="${QWEN_VOICE_PORT:-8001}"
QWEN_BASE_PORT="${QWEN_BASE_PORT:-8002}"
STYLETTS_PORT="${STYLETTS_PORT:-8003}"
MUSIC_PORT="${MUSIC_PORT:-8004}"
SFX_PORT="${SFX_PORT:-8005}"

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
DOCKER_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon|-d)
            DAEMON_MODE=true
            shift
            ;;
        --docker)
            DOCKER_MODE=true
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
            echo "  --docker            Start using Docker Compose"
            echo "  --log-dir PATH      Custom log directory (default: ./logs)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  VOLSUNG_LOG_DIR      Log directory"
            echo "  SKIP_SERVICES        Comma-separated services to skip (e.g., 'music,sfx')"
            echo "  COORDINATOR_PORT     Coordinator port (default: 8000)"
            echo "  QWEN_VOICE_PORT      Qwen Voice port (default: 8001)"
            echo "  QWEN_BASE_PORT       Qwen Base port (default: 8002)"
            echo "  STYLETTS_PORT        StyleTTS2 port (default: 8003)"
            echo "  MUSIC_PORT           Music port (default: 8004)"
            echo "  SFX_PORT             SFX port (default: 8005)"
            echo ""
            echo "Examples:"
            echo "  $0                   # Start all services in foreground"
            echo "  $0 --daemon          # Start all services in background"
            echo "  $0 --docker          # Start using Docker Compose"
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
# Docker Mode
# ==============================================================================

if [[ "$DOCKER_MODE" == true ]]; then
    echo "========================================"
    echo "  Volsung Docker Compose Launcher"
    echo "========================================"
    echo ""
    
    cd "$PROJECT_DIR"
    
    # Check if docker-compose exists
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo "ERROR: docker-compose is not installed"
        exit 1
    fi
    
    echo "Starting services with Docker Compose..."
    echo ""
    
    # Use docker-compose (legacy) or docker compose (modern)
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d --build
    else
        docker compose up -d --build
    fi
    
    echo ""
    echo "========================================"
    echo "  Services starting in Docker..."
    echo "========================================"
    echo ""
    echo "Access the coordinator at:"
    echo "  http://localhost:$COORD_PORT"
    echo ""
    echo "View logs:"
    echo "  docker-compose logs -f"
    echo ""
    echo "Stop services:"
    echo "  docker-compose down"
    echo ""
    
    exit 0
fi

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
    local timeout="${3:-120}"
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
echo "  Coordinator:     port $COORD_PORT"
should_skip "qwen-voice" && echo "  Qwen Voice:      SKIPPED (port $QWEN_VOICE_PORT)" || echo "  Qwen Voice:      port $QWEN_VOICE_PORT"
should_skip "qwen-base" && echo "  Qwen Base:       SKIPPED (port $QWEN_BASE_PORT)" || echo "  Qwen Base:       port $QWEN_BASE_PORT"
should_skip "styletts" && echo "  StyleTTS2:       SKIPPED (port $STYLETTS_PORT)" || echo "  StyleTTS2:       port $STYLETTS_PORT"
should_skip "music" && echo "  Music Service:   SKIPPED (port $MUSIC_PORT)" || echo "  Music Service:   port $MUSIC_PORT"
should_skip "sfx" && echo "  SFX Service:     SKIPPED (port $SFX_PORT)" || echo "  SFX Service:     port $SFX_PORT"
echo ""

# Check if ports are already in use
CONFLICTS=0
check_port $COORD_PORT && echo "ERROR: Port $COORD_PORT (Coordinator) is already in use" && CONFLICTS=$((CONFLICTS + 1))
should_skip "qwen-voice" || { check_port $QWEN_VOICE_PORT && echo "ERROR: Port $QWEN_VOICE_PORT (Qwen Voice) is already in use" && CONFLICTS=$((CONFLICTS + 1)); }
should_skip "qwen-base" || { check_port $QWEN_BASE_PORT && echo "ERROR: Port $QWEN_BASE_PORT (Qwen Base) is already in use" && CONFLICTS=$((CONFLICTS + 1)); }
should_skip "styletts" || { check_port $STYLETTS_PORT && echo "ERROR: Port $STYLETTS_PORT (StyleTTS2) is already in use" && CONFLICTS=$((CONFLICTS + 1)); }
should_skip "music" || { check_port $MUSIC_PORT && echo "ERROR: Port $MUSIC_PORT (Music) is already in use" && CONFLICTS=$((CONFLICTS + 1)); }
should_skip "sfx" || { check_port $SFX_PORT && echo "ERROR: Port $SFX_PORT (SFX) is already in use" && CONFLICTS=$((CONFLICTS + 1)); }

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

# Start Qwen Voice Service
if ! should_skip "qwen-voice"; then
    echo "Starting Qwen Voice Service..."
    cd "$PROJECT_DIR"
    
    if [[ "$DAEMON_MODE" == true ]]; then
        QWEN_VOICE_SERVICE_PORT=$QWEN_VOICE_PORT nohup python -m uvicorn volsung.services.qwen_voice_service:app --host 0.0.0.0 --port $QWEN_VOICE_PORT > "$LOG_DIR/qwen-voice.log" 2>&1 &
        echo "qwen-voice $!" >> "$PID_FILE"
    else
        python -m uvicorn volsung.services.qwen_voice_service:app --host 0.0.0.0 --port $QWEN_VOICE_PORT &
        echo "qwen-voice $!" >> "$PID_FILE"
    fi
    
    if ! wait_for_service "Qwen Voice" "$QWEN_VOICE_PORT" 180; then
        echo "ERROR: Qwen Voice Service failed to start"
        exit 1
    fi
    echo ""
fi

# Start Qwen Base Service
if ! should_skip "qwen-base"; then
    echo "Starting Qwen Base Service..."
    cd "$PROJECT_DIR"
    
    if [[ "$DAEMON_MODE" == true ]]; then
        QWEN_BASE_SERVICE_PORT=$QWEN_BASE_PORT nohup python -m uvicorn volsung.services.qwen_base_service:app --host 0.0.0.0 --port $QWEN_BASE_PORT > "$LOG_DIR/qwen-base.log" 2>&1 &
        echo "qwen-base $!" >> "$PID_FILE"
    else
        python -m uvicorn volsung.services.qwen_base_service:app --host 0.0.0.0 --port $QWEN_BASE_PORT &
        echo "qwen-base $!" >> "$PID_FILE"
    fi
    
    if ! wait_for_service "Qwen Base" "$QWEN_BASE_PORT" 180; then
        echo "ERROR: Qwen Base Service failed to start"
        exit 1
    fi
    echo ""
fi

# Start StyleTTS2 Service
if ! should_skip "styletts"; then
    echo "Starting StyleTTS2 Service..."
    cd "$PROJECT_DIR"
    
    if [[ "$DAEMON_MODE" == true ]]; then
        STYLETTS_SERVICE_PORT=$STYLETTS_PORT nohup python -m uvicorn volsung.services.styletts_service:app --host 0.0.0.0 --port $STYLETTS_PORT > "$LOG_DIR/styletts.log" 2>&1 &
        echo "styletts $!" >> "$PID_FILE"
    else
        python -m uvicorn volsung.services.styletts_service:app --host 0.0.0.0 --port $STYLETTS_PORT &
        echo "styletts $!" >> "$PID_FILE"
    fi
    
    if ! wait_for_service "StyleTTS2" "$STYLETTS_PORT" 180; then
        echo "ERROR: StyleTTS2 Service failed to start"
        exit 1
    fi
    echo ""
fi

# Start Music Service
if ! should_skip "music"; then
    echo "Starting Music Service..."
    cd "$PROJECT_DIR"
    
    if [[ "$DAEMON_MODE" == true ]]; then
        MUSIC_SERVICE_PORT=$MUSIC_PORT nohup python -m uvicorn volsung.services.music_service:app --host 0.0.0.0 --port $MUSIC_PORT > "$LOG_DIR/music.log" 2>&1 &
        echo "music $!" >> "$PID_FILE"
    else
        python -m uvicorn volsung.services.music_service:app --host 0.0.0.0 --port $MUSIC_PORT &
        echo "music $!" >> "$PID_FILE"
    fi
    
    if ! wait_for_service "Music" "$MUSIC_PORT" 180; then
        echo "ERROR: Music Service failed to start"
        exit 1
    fi
    echo ""
fi

# Start SFX Service
if ! should_skip "sfx"; then
    echo "Starting SFX Service..."
    cd "$PROJECT_DIR"
    
    if [[ "$DAEMON_MODE" == true ]]; then
        SFX_SERVICE_PORT=$SFX_PORT nohup python -m uvicorn volsung.services.sfx_service:app --host 0.0.0.0 --port $SFX_PORT > "$LOG_DIR/sfx.log" 2>&1 &
        echo "sfx $!" >> "$PID_FILE"
    else
        python -m uvicorn volsung.services.sfx_service:app --host 0.0.0.0 --port $SFX_PORT &
        echo "sfx $!" >> "$PID_FILE"
    fi
    
    if ! wait_for_service "SFX" "$SFX_PORT" 180; then
        echo "ERROR: SFX Service failed to start"
        exit 1
    fi
    echo ""
fi

# Start Coordinator (last, depends on all services)
echo "Starting Coordinator..."
cd "$PROJECT_DIR"

# Wait a moment for services to be ready
sleep 2

# Set environment for coordinator
export QWEN_VOICE_SERVICE_URL="http://localhost:$QWEN_VOICE_PORT"
export QWEN_BASE_SERVICE_URL="http://localhost:$QWEN_BASE_PORT"
export STYLETTS_SERVICE_URL="http://localhost:$STYLETTS_PORT"
export MUSIC_SERVICE_URL="http://localhost:$MUSIC_PORT"
export SFX_SERVICE_URL="http://localhost:$SFX_PORT"
export COORDINATOR_PORT="$COORD_PORT"

if [[ "$DAEMON_MODE" == true ]]; then
    nohup python -m uvicorn volsung.server:app --host 0.0.0.0 --port $COORD_PORT > "$LOG_DIR/coordinator.log" 2>&1 &
    COORD_PID=$!
    echo "coordinator $COORD_PID" >> "$PID_FILE"
else
    python -m uvicorn volsung.server:app --host 0.0.0.0 --port $COORD_PORT &
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
echo "Individual service health:"
should_skip "qwen-voice" || echo "  Qwen Voice:  curl http://localhost:$QWEN_VOICE_PORT/health"
should_skip "qwen-base" || echo "  Qwen Base:   curl http://localhost:$QWEN_BASE_PORT/health"
should_skip "styletts" || echo "  StyleTTS2:   curl http://localhost:$STYLETTS_PORT/health"
should_skip "music" || echo "  Music:       curl http://localhost:$MUSIC_PORT/health"
should_skip "sfx" || echo "  SFX:         curl http://localhost:$SFX_PORT/health"
echo ""

if [[ "$DAEMON_MODE" == true ]]; then
    echo "Services are running in background."
    echo "Logs: $LOG_DIR"
    echo "PIDs: $PID_FILE"
    echo ""
    echo "To stop:"
    echo "  kill \$(cat $PID_FILE | awk '{print \$2}')"
else
    echo "Services are running. Press Ctrl+C to stop."
    echo ""
    
    # Wait for coordinator to finish
    wait $COORD_PID
fi
