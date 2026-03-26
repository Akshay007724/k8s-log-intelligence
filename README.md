# Kubernetes Log Intelligence and Failure Resolution System

Diagnoses Kubernetes pod failures in **under 2 seconds** — tells you what is failing, why, and how to fix it.

```json
{
  "issue_type": "OOMKilled",
  "root_cause": "payment-svc exceeded its memory limit and was killed by the kernel (OOMKiller).",
  "confidence": 0.95,
  "evidence": [
    "payment-svc phase=Running restarts=9 reason=OOMKilled",
    "Log errors: OOMKilledx9, memory 97%"
  ],
  "suggested_fix": "Increase memory limit: `kubectl set resources deployment/payment-svc --limits=memory=512Mi -n production`"
}
```

---

## How It Works

A K-Agent orchestrator (`context-weaver`) runs a **THINK → ACT → OBSERVE → DECIDE** loop:

1. **THINK** — decides which data to fetch (pod status, logs, metrics)
2. **ACT** — calls up to 3 MCP servers in parallel (Kubernetes API, Logs, Metrics)
3. **OBSERVE** — receives compressed context shards (≤120 tokens each)
4. **DECIDE** — scores failure gravity, classifies root cause, returns fix

```
POST /diagnose {"service": "payment-svc", "namespace": "production"}
        │
        ▼
  context-weaver (port 8000)
        │
   ┌────┴────┐
   ▼         ▼
mcp-k8s   mcp-logs   mcp-metrics (optional)
(8001)    (8002)     (8003)
   │         │
   └────┬────┘
        ▼
   sub-agents (rule-based, sync)
   ├── log-analysis-agent
   ├── failure-classification-agent
   └── root-cause-agent
        │
        ▼
   IncidentReport (≤200 tokens)
```

**Failure types detected:** `OOMKilled` · `CrashLoopBackOff` · `ImagePullBackOff` · `DependencyTimeout` · `DeploymentRegression`

---

## Quick Start

### Option A — Docker Compose (recommended)

**Requirements:** Docker, a kubeconfig at `~/.kube/config`

```bash
git clone https://github.com/Akshay007724/k8s-log-intelligence.git
cd k8s-log-intelligence

cp .env.example .env        # optionally set ANTHROPIC_API_KEY

docker compose up --build
```

Diagnose a pod:

```bash
curl -s -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"service": "payment-svc", "namespace": "production"}' \
  | python3 -m json.tool
```

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

### Option B — Local Python (no Docker)

**Requirements:** Python 3.12+, `pip install -r requirements.txt`

```bash
pip install -r requirements.txt
```

Start all four servers (one terminal each):

```bash
# Terminal 1
IN_CLUSTER=false python3 -m uvicorn src.mcp_k8s_server:app --port 8001

# Terminal 2
IN_CLUSTER=false python3 -m uvicorn src.mcp_logs_server:app --port 8002

# Terminal 3
IN_CLUSTER=false python3 -m uvicorn src.mcp_metrics_server:app --port 8003

# Terminal 4
IN_CLUSTER=false \
MCP_K8S_URL=http://localhost:8001 \
MCP_LOGS_URL=http://localhost:8002 \
MCP_METRICS_URL=http://localhost:8003 \
python3 -m uvicorn src.context_weaver:app --port 8000
```

---

### Option C — Inside Kubernetes

```bash
# 1. Create namespace and secret
kubectl create namespace monitoring
kubectl create secret generic k8s-log-intelligence-secrets \
  --from-literal=anthropic_api_key=YOUR_KEY \
  -n monitoring

# 2. Build and push image
docker build -t YOUR_REGISTRY/k8s-log-intelligence:latest .
docker push YOUR_REGISTRY/k8s-log-intelligence:latest

# 3. Update image tag in kubernetes/deployment.yaml, then apply
kubectl apply -f kubernetes/

# 4. Verify rollout
kubectl rollout status deployment/context-weaver -n monitoring

# 5. Port-forward and test
kubectl port-forward svc/context-weaver 8000:80 -n monitoring
curl -X POST http://localhost:8000/diagnose \
  -d '{"service":"payment-svc","namespace":"production"}'
```

---

## API Reference

### `POST /diagnose`

Diagnose a failing service.

**Request:**
```json
{
  "service": "payment-svc",
  "namespace": "production"
}
```

**Response:**
```json
{
  "issue_type": "DependencyTimeout",
  "root_cause": "payment-svc cannot reach a required dependency (connection refused/timeout). Involved: redis:6379.",
  "confidence": 0.85,
  "evidence": [
    "ECONNREFUSED redis:6379 connection refused 12x",
    "payment-svc phase=Running restarts=5"
  ],
  "suggested_fix": "Check dependency reachability: `kubectl exec -it <pod> -- nc -zv <host> <port>`. Verify Service/Endpoints exist. If post-deploy, roll back: `kubectl rollout undo deployment/payment-svc -n production`"
}
```

**Issue types:**

| Type | Meaning |
|---|---|
| `OOMKilled` | Container exceeded memory limit, killed by kernel |
| `CrashLoopBackOff` | Pod crashing repeatedly on startup |
| `ImagePullBackOff` | Container image cannot be pulled |
| `DependencyTimeout` | Cannot reach a downstream service (DB, cache, etc.) |
| `DeploymentRegression` | Recent rollout broke pod availability |
| `Unknown` | Unclassified — inspect manually |

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## Configuration

All settings are via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `IN_CLUSTER` | `true` | Set to `false` for local dev |
| `MCP_K8S_URL` | `http://mcp-k8s-server:8001` | K8s MCP server URL |
| `MCP_LOGS_URL` | `http://mcp-logs-server:8002` | Logs MCP server URL |
| `MCP_METRICS_URL` | `http://mcp-metrics-server:8003` | Metrics MCP server URL |
| `FAILURE_GRAVITY_THRESHOLD` | `0.65` | Minimum score to surface an issue |
| `MAX_TOOL_CALLS` | `3` | Maximum MCP calls per diagnosis |
| `ANTHROPIC_API_KEY` | _(empty)_ | Optional — for future LLM integration |

Copy `.env.example` to `.env` and edit for local use.

---

## Project Structure

```
src/
├── context_weaver.py     # K-Agent orchestrator + /diagnose endpoint
├── sub_agents.py         # log-analysis, classification, root-cause agents
├── mcp_k8s_server.py     # MCP server: pod status, events, deployment diff
├── mcp_logs_server.py    # MCP server: log compression, error clustering
├── mcp_metrics_server.py # MCP server: CPU/memory anomaly detection
├── models.py             # Pydantic models (ContextShard, IncidentReport, …)
└── config.py             # Settings

kubernetes/
├── rbac.yaml             # ServiceAccount, ClusterRole (read-only), Binding
├── configmap.yaml        # Non-secret configuration
└── deployment.yaml       # 4 Deployments + 4 Services

tests/                    # 24 tests (unit + integration, no real K8s needed)
Dockerfile                # Multi-stage build (builder + runtime, non-root)
docker-compose.yml        # Local dev wiring all 4 services
```

---

## Running Tests

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

No Kubernetes cluster needed — MCP servers fall back to mock responses when unavailable.

Expected output: `24 passed`

---

## Without a Kubernetes Cluster

The system works without a real cluster. MCP servers return mock context shards, and the orchestrator still runs the full classification pipeline. This is useful for testing the logic locally before deploying.

---

## RBAC Permissions

The system requires read-only access to:
- `pods`, `pods/log`, `events`, `namespaces`
- `deployments`, `replicasets`
- `metrics.k8s.io/pods`

No write permissions are requested. See `kubernetes/rbac.yaml`.
