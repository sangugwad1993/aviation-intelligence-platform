# Next-Generation Aviation Intelligence Platform Powered by AI

**M.Tech Data Science & Artificial Intelligence | PES University**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests 192/192](https://img.shields.io/badge/tests-192%2F192%20passed-brightgreen.svg)]()
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)

A cloud-native AI platform for aviation delay prediction, anomaly detection, and safety assurance — featuring **6 fully implemented AI/ML modules**, a **Streamlit dashboard** with authentication, and **Docker + Kubernetes** deployment.

---

## Quick Start (< 5 minutes)

### Option A: One-Click Setup (macOS / Linux)

```bash
git clone <repo-url>
cd aviation-intelligence-platform
chmod +x setup.sh
./setup.sh
```

### Option B: One-Click Setup (Windows)

```cmd
git clone <repo-url>
cd aviation-intelligence-platform
setup.bat
```

### Option C: Docker (any OS — just needs Docker Desktop)

```bash
git clone <repo-url>
cd aviation-intelligence-platform
docker compose up --build -d
```

Dashboard opens at **http://localhost:8501**

### Login Credentials

| Username | Password | Access |
|----------|----------|--------|
| `admin` | `aviation2024` | All pages + Admin Panel (manage users, change passwords) |
| `viewer` | `readonly2024` | Live Flight Map + Prediction Panel only |

---

## What's Inside

### 6 AI/ML Modules — All Fully Implemented & Tested

| # | Module | Source | Tests | Key Tech |
|---|--------|--------|-------|----------|
| 001 | **Liquid Neural Networks** | `src/models/liquid_nn/` | 18/18 | Neural ODEs, LTC cells, built-in Euler solver |
| 003 | **Multi-Agent System** | `src/agents/langgraph/` | 26/26 | 5 specialist agents, state-graph orchestration |
| 004 | **Temporal Fusion Transformer** | `src/models/tft/` | 26/26 | VSN, GRN, interpretable attention, quantile loss |
| 005 | **Neuro-Symbolic Reasoning** | `src/models/neuro_symbolic/` | 28/28 | Knowledge graphs, GNN, logic engine, fuzzy logic |
| 006 | **Constitutional AI Safety** | `src/safety/constitutional_ai/` | 24/24 | Safety constitution, self-critique, reward model |
| 008 | **Federated Learning** | `src/federated/` | 70/70 | FedAvg, FedProx, differential privacy (Opacus) |

**Total: 192/192 tests passing** (local + Docker)

### Dashboard Pages

| Page | Description | Access |
|------|-------------|--------|
| Live Flight Map | Real-time ADS-B positions from OpenSky Network | All users |
| Prediction Panel | AI delay predictions, anomaly detection, risk levels | All users |
| FL Training Monitor | Federated learning experiment results & convergence curves | Admin |
| System Health | Module implementation status, test counts | Admin |
| Admin Panel | User management: add/remove users, change passwords/roles | Admin |

---

## Prerequisites

### For Native Setup (without Docker)

| Requirement | Version | Auto-installed by `setup.sh`? |
|-------------|---------|-------------------------------|
| Python | 3.10+ | Yes (via Homebrew/apt) |
| pip | Latest | Yes |
| PyTorch | 2.0+ | Yes (via requirements.txt) |
| Streamlit | 1.37+ | Yes (via requirements.txt) |

### For Docker Setup

| Requirement | Version | Notes |
|-------------|---------|-------|
| Docker Desktop | Latest | [Download here](https://www.docker.com/products/docker-desktop/) |

**No other dependencies needed** — everything is bundled in the Docker image.

---

## Project Structure

```
aviation-intelligence-platform/
├── dashboard/
│   ├── app.py                    # Streamlit dashboard (main entry)
│   └── user_manager.py           # User auth & session management
├── src/
│   ├── models/
│   │   ├── liquid_nn/            # Feature 001: Liquid Neural Networks
│   │   ├── tft/                  # Feature 004: Temporal Fusion Transformer
│   │   └── neuro_symbolic/       # Feature 005: Neuro-Symbolic Reasoning
│   ├── agents/langgraph/         # Feature 003: Multi-Agent System
│   ├── safety/constitutional_ai/ # Feature 006: Constitutional AI
│   └── federated/                # Feature 008: Federated Learning
├── tests/                        # 192 tests across all modules
├── scripts/                      # Experiment runners
├── artifacts/                    # Experiment results (JSON)
├── k8s/                          # Kubernetes manifests
│   ├── namespace.yaml
│   ├── secret.yaml
│   ├── deployment.yaml           # 2 replicas, resource limits, probes
│   ├── service.yaml
│   ├── ingress.yaml              # Nginx + TLS
│   └── deploy.sh                 # One-command K8s deploy/teardown
├── docker-compose.yml            # Dashboard + test + experiment services
├── Dockerfile                    # Multi-stage build (Python 3.12-slim)
├── requirements.txt              # Python dependencies
├── setup.sh                      # One-click setup (macOS/Linux)
├── setup.bat                     # One-click setup (Windows)
└── README.md                     # This file
```

---

## Manual Setup (step by step)

If you prefer not to use the setup scripts:

```bash
# 1. Clone and enter
cd aviation-intelligence-platform

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests (all 192 should pass)
PYTHONPATH=. python -m pytest tests/ -v --no-cov

# 5. Launch dashboard
PYTHONPATH=. streamlit run dashboard/app.py --server.port=8501
```

---

## Docker Commands

```bash
# Build and start dashboard
docker compose up --build -d

# Run all 192 tests inside container
docker compose run --rm test

# Run all-module experiments
docker compose run --rm experiments

# View logs
docker compose logs -f

# Stop
docker compose down
```

---

## Kubernetes Deployment

```bash
# Deploy (creates namespace, secrets, 2-replica deployment, service, ingress)
./k8s/deploy.sh

# Port-forward to access locally
kubectl -n aviation-intelligence port-forward svc/aviation-dashboard-svc 8501:80

# Teardown
./k8s/deploy.sh --delete
```

Edit `k8s/ingress.yaml` to set your actual domain (replace `aviation.example.com`).

---

## Running Tests

```bash
# All tests
PYTHONPATH=. python -m pytest tests/ -v --no-cov

# Specific module
PYTHONPATH=. python -m pytest tests/test_federated/ -v --no-cov
PYTHONPATH=. python -m pytest tests/test_liquid_nn/ -v --no-cov
PYTHONPATH=. python -m pytest tests/test_multi_agent/ -v --no-cov
PYTHONPATH=. python -m pytest tests/test_tft/ -v --no-cov
PYTHONPATH=. python -m pytest tests/test_neuro_symbolic/ -v --no-cov
PYTHONPATH=. python -m pytest tests/test_constitutional_ai/ -v --no-cov
```

---

## Research References

1. **Liquid Neural Networks** — Hasani et al. (2021). "Liquid Time-constant Networks". AAAI. [arXiv:2006.04439](https://arxiv.org/abs/2006.04439)
2. **Temporal Fusion Transformers** — Lim et al. (2021). "TFT for Interpretable Multi-horizon Forecasting". [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
3. **Constitutional AI** — Bai et al. (2022). "Constitutional AI: Harmlessness from AI Feedback". [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
4. **Federated Learning** — McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks". AISTATS.
5. **FedProx** — Li et al. (2020). "Federated Optimization in Heterogeneous Networks". MLSys.
6. **Differential Privacy** — Abadi et al. (2016). "Deep Learning with Differential Privacy". CCS.

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError` | Ensure `PYTHONPATH=.` is set, or run from project root |
| OpenSky API timeout | Dashboard auto-falls back to demo data (50 flights) |
| Docker daemon not running | Start Docker Desktop, wait for it to initialize |
| Port 8501 already in use | `pkill -f streamlit` or `docker compose down` first |
| Login lost on reload | Fixed — sessions persist 24h via URL token |
| Tests fail in Docker | Rebuild: `docker compose build --no-cache` |

---

**Built for M.Tech Data Science & Artificial Intelligence | PES University**
**Last Updated**: April 2026 | **Version**: 2.0.0
