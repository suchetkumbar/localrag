#!/usr/bin/env bash
# LocalRAG bare-metal startup script
# Usage: ./scripts/start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "  LocalRAG — Local RAG Application"
echo "=========================================="

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

check_command() {
  if ! command -v "$1" &>/dev/null; then
    echo "❌ Required command not found: $1"
    echo "   Install it from: $2"
    exit 1
  fi
  echo "✅ $1 found"
}

echo ""
echo "🔍 Checking prerequisites..."
check_command python3 "https://python.org"
check_command ollama "https://ollama.com"
check_command node "https://nodejs.org"
check_command npm "https://nodejs.org"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_MINOR=11
if python3 -c "import sys; exit(0 if sys.version_info >= (3, $REQUIRED_MINOR) else 1)"; then
  echo "✅ Python $PYTHON_VERSION"
else
  echo "❌ Python 3.$REQUIRED_MINOR+ required (found $PYTHON_VERSION)"
  exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"

if [ ! -f ".env" ]; then
  echo ""
  echo "📋 Creating .env from template..."
  cp .env.example .env
  echo "✅ .env created — review it if needed"
fi

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

echo ""
echo "📁 Creating data directories..."
mkdir -p data/{chroma,db,watch_folder,uploads,logs}
echo "✅ Directories ready"

# ---------------------------------------------------------------------------
# Ollama models
# ---------------------------------------------------------------------------

echo ""
echo "🤖 Checking Ollama models..."

if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
  echo "⚠️  Ollama is not running. Starting it..."
  ollama serve &
  OLLAMA_PID=$!
  sleep 3
fi

for MODEL in llama3 nomic-embed-text; do
  if ollama list 2>/dev/null | grep -q "^$MODEL"; then
    echo "✅ $MODEL already pulled"
  else
    echo "⬇️  Pulling $MODEL (this may take a few minutes)..."
    ollama pull "$MODEL"
    echo "✅ $MODEL ready"
  fi
done

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------

echo ""
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt -q
echo "✅ Python dependencies installed"

# ---------------------------------------------------------------------------
# Frontend dependencies
# ---------------------------------------------------------------------------

echo ""
echo "📦 Installing frontend dependencies..."
cd frontend
npm install --silent
cd "$PROJECT_ROOT"
echo "✅ Frontend dependencies installed"

# ---------------------------------------------------------------------------
# Start services
# ---------------------------------------------------------------------------

echo ""
echo "🚀 Starting LocalRAG..."
echo ""

# Backend
echo "Starting FastAPI backend on port 8000..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-config /dev/null &
BACKEND_PID=$!

# Wait for backend
echo "Waiting for backend to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health &>/dev/null; then
    echo "✅ Backend ready"
    break
  fi
  sleep 1
done

# Frontend
echo "Starting Next.js frontend on port 3000..."
cd frontend
npm run dev -- --port 3000 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "  LocalRAG is running!"
echo "=========================================="
echo ""
echo "  Chat UI:   http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Health:    http://localhost:8000/health"
echo ""
echo "  Drop files into: ./data/watch_folder/"
echo ""
echo "  Press Ctrl+C to stop all services"
echo ""

# Trap cleanup
cleanup() {
  echo ""
  echo "🛑 Stopping services..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

wait
