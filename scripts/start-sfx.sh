#!/bin/bash
# Start the SFX Service (AudioLDM2 Sound Effects Generation)
# Port: 8003

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=================================="
echo "Volsung SFX Service"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    echo "Activating virtual environment..."
    source "${PROJECT_ROOT}/.venv/bin/activate"
else
    echo "Warning: No virtual environment found at ${PROJECT_ROOT}/.venv"
    echo "Using system Python..."
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import fastapi, diffusers, torch" 2>/dev/null || {
    echo "Installing SFX service dependencies..."
    pip install -r "${PROJECT_ROOT}/requirements/services/sfx.txt"
}

# Set environment variables
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export SFX_MODEL_SIZE="${SFX_MODEL_SIZE:-base}"

# Optional: GPU settings
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"
else
    echo "CUDA not configured, will use CPU if no GPU available"
fi

echo ""
echo "Starting SFX Service on port 8003..."
echo "Model: AudioLDM2 (${SFX_MODEL_SIZE})"
echo ""
echo "Endpoints:"
echo "  GET  http://localhost:8003/health"
echo "  POST http://localhost:8003/sfx/generate"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================="
echo ""

# Run the service
cd "${PROJECT_ROOT}"
exec python -m volsung.services.sfx_service
