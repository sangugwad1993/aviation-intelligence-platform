#!/usr/bin/env bash
# ============================================================================
# Aviation Intelligence Platform — One-Click Setup (macOS / Linux)
# ============================================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh            → Full setup (native Python)
#   chmod +x setup.sh && ./setup.sh --docker    → Docker-only setup
#   chmod +x setup.sh && ./setup.sh --help      → Show help
# ============================================================================

set -euo pipefail

# --- Colors ---------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Help ------------------------------------------------------------------
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo ""
    echo "Aviation Intelligence Platform — Setup Script"
    echo ""
    echo "Usage:"
    echo "  ./setup.sh             Full native setup (Python venv + deps + tests + launch)"
    echo "  ./setup.sh --docker    Docker-only setup (build image + launch container)"
    echo "  ./setup.sh --check     Check prerequisites only (no install)"
    echo ""
    echo "Prerequisites installed automatically:"
    echo "  • Python 3.10+  (via Homebrew on macOS, apt on Ubuntu/Debian)"
    echo "  • pip"
    echo "  • Docker Desktop (prompted if missing, for --docker mode)"
    echo ""
    echo "Default dashboard credentials:"
    echo "  • admin / aviation2024     (full access)"
    echo "  • viewer / readonly2024    (map + predictions only)"
    echo ""
    exit 0
fi

MODE="${1:-native}"

echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  Aviation Intelligence Platform — Setup${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""

# --- Detect OS -------------------------------------------------------------
detect_os() {
    case "$(uname -s)" in
        Darwin*)  OS="macos" ;;
        Linux*)   OS="linux" ;;
        MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
        *)        OS="unknown" ;;
    esac
    ARCH="$(uname -m)"
    info "Detected: ${BOLD}$OS${NC} ($ARCH)"
}

# --- Check / Install Python ------------------------------------------------
ensure_python() {
    info "Checking Python..."

    # Find a suitable Python (3.10+)
    PYTHON=""
    for candidate in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$candidate" &>/dev/null; then
            ver=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
                PYTHON="$candidate"
                break
            fi
        fi
    done

    if [[ -z "$PYTHON" ]]; then
        warn "Python 3.10+ not found. Installing..."
        if [[ "$OS" == "macos" ]]; then
            if ! command -v brew &>/dev/null; then
                info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python@3.12
            PYTHON="python3.12"
        elif [[ "$OS" == "linux" ]]; then
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -qq
                sudo apt-get install -y python3 python3-pip python3-venv
                PYTHON="python3"
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y python3 python3-pip
                PYTHON="python3"
            else
                fail "Cannot auto-install Python. Please install Python 3.10+ manually."
            fi
        else
            fail "Cannot auto-install Python on this OS. Please install Python 3.10+ manually."
        fi
    fi

    PYTHON_VER=$("$PYTHON" --version 2>&1)
    ok "$PYTHON_VER"
}

# --- Check / Install Docker ------------------------------------------------
ensure_docker() {
    info "Checking Docker..."

    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        DOCKER_VER=$(docker --version)
        ok "$DOCKER_VER"
        return 0
    fi

    if command -v docker &>/dev/null; then
        warn "Docker is installed but not running."
        if [[ "$OS" == "macos" ]]; then
            info "Starting Docker Desktop..."
            open -a Docker
            echo -n "  Waiting for Docker to start "
            for i in $(seq 1 30); do
                docker info &>/dev/null 2>&1 && break
                echo -n "."
                sleep 2
            done
            echo ""
            if docker info &>/dev/null 2>&1; then
                ok "Docker is now running."
                return 0
            fi
        fi
        fail "Docker is not running. Please start Docker Desktop and re-run this script."
    fi

    warn "Docker not found."
    echo ""
    echo "  Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    echo ""
    if [[ "$MODE" == "--docker" ]]; then
        fail "Docker is required for --docker mode."
    else
        warn "Skipping Docker (not needed for native setup)."
    fi
}

# --- Create virtual environment & install deps -----------------------------
setup_native() {
    info "Creating Python virtual environment..."
    if [[ ! -d ".venv" ]]; then
        "$PYTHON" -m venv .venv
        ok "Virtual environment created at .venv/"
    else
        ok "Virtual environment already exists."
    fi

    # Activate
    source .venv/bin/activate
    ok "Activated .venv ($(python --version))"

    # Upgrade pip
    info "Upgrading pip..."
    pip install --upgrade pip -q
    ok "pip upgraded"

    # Install dependencies
    info "Installing dependencies (this may take 2-3 minutes)..."
    pip install -r requirements-dev.txt -q
    ok "All Python dependencies installed"

    # Create data directory for user management
    mkdir -p data artifacts
    ok "Data directories ready"

    # Run tests
    echo ""
    info "Running test suite..."
    echo ""
    PYTHONPATH=. python -m pytest tests/ -v --no-cov --tb=short 2>&1 | tail -20
    TEST_EXIT=${PIPESTATUS[0]:-0}

    if [[ "$TEST_EXIT" -eq 0 ]]; then
        echo ""
        ok "All tests passed!"
    else
        echo ""
        warn "Some tests failed — the dashboard will still work. Review output above."
    fi
}

# --- Docker setup ----------------------------------------------------------
setup_docker() {
    info "Building Docker image (this may take 2-5 minutes on first run)..."
    docker compose build
    ok "Docker image built"

    info "Starting dashboard container..."
    docker compose up -d
    ok "Container running"

    echo ""
    info "Running tests inside container..."
    docker compose run --rm test 2>&1 | tail -5
    ok "Container tests done"
}

# --- Launch dashboard ------------------------------------------------------
launch_dashboard() {
    echo ""
    echo -e "${BOLD}=============================================${NC}"
    echo -e "${GREEN}  Setup Complete!${NC}"
    echo -e "${BOLD}=============================================${NC}"
    echo ""

    if [[ "$MODE" == "--docker" ]]; then
        echo "  Dashboard running at:  http://localhost:8501"
        echo ""
        echo "  Commands:"
        echo "    docker compose up -d          Start dashboard"
        echo "    docker compose down            Stop dashboard"
        echo "    docker compose run --rm test   Run all 192 tests"
        echo "    docker compose logs -f         View logs"
    else
        echo "  To start the dashboard:"
        echo ""
        echo "    source .venv/bin/activate"
        echo "    PYTHONPATH=. streamlit run dashboard/app.py"
        echo ""
        echo "  Or use Docker:"
        echo ""
        echo "    docker compose up --build -d"
    fi

    echo ""
    echo -e "  ${BOLD}Login credentials:${NC}"
    echo "    admin  / aviation2024     (full access + admin panel)"
    echo "    viewer / readonly2024     (flight map + predictions)"
    echo ""
    echo -e "  ${BOLD}Dashboard URL:${NC} http://localhost:8501"
    echo ""

    # Ask to launch now (native mode only)
    if [[ "$MODE" != "--docker" ]]; then
        read -p "  Launch dashboard now? [Y/n] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z "$REPLY" ]]; then
            info "Starting Streamlit dashboard..."
            source .venv/bin/activate
            PYTHONPATH=. streamlit run dashboard/app.py --server.port=8501 --server.address=localhost
        fi
    fi
}

# --- Main ------------------------------------------------------------------
detect_os

if [[ "$MODE" == "--docker" ]]; then
    ensure_docker
    setup_docker
elif [[ "$MODE" == "--check" ]]; then
    ensure_python
    ensure_docker
    echo ""
    ok "All prerequisites look good."
    exit 0
else
    ensure_python
    setup_native
fi

launch_dashboard
