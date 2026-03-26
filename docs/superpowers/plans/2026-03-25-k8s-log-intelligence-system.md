# Kubernetes Log Intelligence and Failure Resolution System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lean, async AI orchestration system that diagnoses Kubernetes failures (CrashLoopBackOff, OOMKilled, etc.) in under 2 seconds, returning a structured root-cause + fix under 200 tokens.

**Architecture:** A FastAPI-based K-Agent orchestrator (`context_weaver.py`) runs a THINK→ACT→OBSERVE→DECIDE loop, delegating to three specialized sub-agents and calling up to three MCP servers (K8s, Logs, Metrics) only when needed. All MCP responses are compressed into context shards ≤120 tokens. The system runs inside the cluster via a ServiceAccount with read-only RBAC.

**Tech Stack:** Python 3.12, FastAPI, httpx (async), Anthropic Claude API (claude-sonnet-4-6), kubernetes-asyncio, Pydantic v2, Docker multi-stage, Kubernetes RBAC/ConfigMap/Deployment.

---

## File Map

| File | Responsibility |
|---|---|
| `src/context_weaver.py` | K-Agent orchestrator: THINK→ACT→OBSERVE→DECIDE loop, failure gravity scoring, final answer |
| `src/sub_agents.py` | Three sub-agents: log-analysis, failure-classification, root-cause |
| `src/mcp_k8s_server.py` | FastAPI MCP server: pod status, events, deployment changes via kubernetes-asyncio |
| `src/mcp_logs_server.py` | FastAPI MCP server: log compression, error pattern clustering |
| `src/mcp_metrics_server.py` | FastAPI MCP server (optional): resource anomaly detection |
| `src/models.py` | Pydantic models: ContextShard, FailureQuery, IncidentReport, AgentState |
| `src/config.py` | Settings from env vars (namespace, model ID, MCP endpoints, token budgets) |
| `tests/test_models.py` | Tests for Pydantic models and token budget assertions |
| `tests/test_sub_agents.py` | Unit tests for all three sub-agents |
| `tests/test_context_weaver.py` | Integration tests for orchestrator loop |
| `tests/test_mcp_servers.py` | Tests for MCP server endpoints |
| `tests/test_integration.py` | End-to-end scenario tests (OOM, Redis-down, ImagePull) |
| `tests/conftest.py` | Shared fixtures (mock K8s client, mock logs) |
| `Dockerfile` | Multi-stage build (builder + runtime) |
| `docker-compose.yml` | Local dev: all services wired together |
| `kubernetes/deployment.yaml` | Deployment + Service |
| `kubernetes/rbac.yaml` | ServiceAccount, ClusterRole, ClusterRoleBinding |
| `kubernetes/configmap.yaml` | ConfigMap for non-secret config |

---

## Task 1: Project Scaffold and Models

**Files:**
- Create: `src/__init__.py`
- Create: `src/models.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `requirements.txt`

- [ ] **Step 1: Write the failing model tests**

```python
# tests/test_models.py
from src.models import ContextShard, FailureQuery, IncidentReport

def test_context_shard_token_budget():
    shard = ContextShard(
        summary="Pod restarted 7 times due to DB timeout",
        severity=0.82,
        entities=["payment-service", "postgres"],
        timestamp="recent"
    )
    assert shard.severity <= 1.0
    import json
    token_estimate = len(json.dumps(shard.model_dump()).split())
    assert token_estimate < 120

def test_failure_query_required_fields():
    q = FailureQuery(service="payment-svc", namespace="production")
    assert q.service == "payment-svc"

def test_incident_report_token_budget():
    r = IncidentReport(
        issue_type="CrashLoopBackOff",
        root_cause="Redis unreachable after deployment",
        confidence=0.91,
        evidence=["7 restarts in 5m", "ECONNREFUSED redis:6379"],
        suggested_fix="kubectl rollout undo deployment/payment-svc"
    )
    import json
    token_estimate = len(json.dumps(r.model_dump()).split())
    assert token_estimate < 200
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/ad6/Ambient_layer && python -m pytest tests/test_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'src'`

- [ ] **Step 3: Write requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
httpx==0.27.0
pydantic==2.7.0
pydantic-settings==2.3.0
anthropic==0.28.0
kubernetes-asyncio==29.0.0
pytest==8.2.0
pytest-asyncio==0.23.7
```

- [ ] **Step 4: Write `src/models.py`**

```python
from pydantic import BaseModel, Field
from typing import Literal

class ContextShard(BaseModel):
    summary: str = Field(max_length=500)
    severity: float = Field(ge=0.0, le=1.0)
    entities: list[str]
    timestamp: str

class FailureQuery(BaseModel):
    service: str
    namespace: str

class IncidentReport(BaseModel):
    issue_type: Literal[
        "CrashLoopBackOff", "OOMKilled", "ImagePullBackOff",
        "DependencyTimeout", "DeploymentRegression", "Unknown"
    ]
    root_cause: str = Field(max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]  # capped at 3 items by root_cause_agent via slicing
    suggested_fix: str = Field(max_length=200)

class AgentState(BaseModel):
    query: FailureQuery
    shards: list[ContextShard] = []
    tool_calls_used: int = 0
    max_tool_calls: int = 3
```

- [ ] **Step 5: Write `src/config.py`**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    model_id: str = "claude-sonnet-4-6"
    mcp_k8s_url: str = "http://mcp-k8s-server:8001"
    mcp_logs_url: str = "http://mcp-logs-server:8002"
    mcp_metrics_url: str = "http://mcp-metrics-server:8003"
    in_cluster: bool = True
    failure_gravity_threshold: float = 0.65
    max_tool_calls: int = 3

    class Config:
        env_file = ".env"

settings = Settings()
```

- [ ] **Step 6: Write `src/__init__.py` and `tests/__init__.py`**

Both are empty files.

- [ ] **Step 7: Write `tests/conftest.py`**

```python
import pytest
from src.models import ContextShard, FailureQuery

@pytest.fixture
def sample_query():
    return FailureQuery(service="payment-svc", namespace="production")

@pytest.fixture
def crash_shard():
    return ContextShard(
        summary="Pod restarted 9x in 10m, OOMKilled, last exit code 137",
        severity=0.9,
        entities=["payment-svc"],
        timestamp="recent"
    )

@pytest.fixture
def dep_shard():
    return ContextShard(
        summary="ECONNREFUSED redis:6379, 15 connection errors in 2m",
        severity=0.75,
        entities=["payment-svc", "redis"],
        timestamp="recent"
    )
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
pip install -r requirements.txt
python -m pytest tests/test_models.py -v
```
Expected: 3 PASSED

- [ ] **Step 9: Commit**

```bash
git add src/ tests/ requirements.txt
git commit -m "feat: add pydantic models, config, and test scaffold"
```

---

## Task 2: MCP K8s Server

**Files:**
- Create: `src/mcp_k8s_server.py`
- Create: `tests/test_mcp_servers.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp_servers.py
import pytest
from httpx import AsyncClient, ASGITransport
from src.mcp_k8s_server import app as k8s_app

@pytest.mark.asyncio
async def test_get_pod_status_returns_shard():
    transport = ASGITransport(app=k8s_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/get_pod_status", json={
            "service": "payment-svc", "namespace": "production"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "severity" in data
    assert "entities" in data
    assert len(data["summary"].split()) < 120

@pytest.mark.asyncio
async def test_get_recent_events_returns_shard():
    transport = ASGITransport(app=k8s_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/get_recent_events", json={
            "service": "payment-svc", "namespace": "production"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_mcp_servers.py::test_get_pod_status_returns_shard -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `src/mcp_k8s_server.py`**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from src.models import ContextShard, FailureQuery

app = FastAPI(title="MCP K8s Server")

class K8sRequest(BaseModel):
    service: str
    namespace: str

async def _get_k8s_client():
    """Returns kubernetes async client, handles both in-cluster and local."""
    try:
        from kubernetes_asyncio import client, config as k8s_config
        try:
            await k8s_config.load_incluster_config()
        except Exception:
            await k8s_config.load_kube_config()
        return client.CoreV1Api(), client.AppsV1Api()
    except Exception:
        return None, None

def _compute_restart_severity(restart_count: int) -> float:
    if restart_count >= 10:
        return 0.95
    if restart_count >= 5:
        return 0.80
    if restart_count >= 2:
        return 0.60
    return 0.30

@app.post("/get_pod_status", response_model=ContextShard)
async def get_pod_status(req: K8sRequest) -> ContextShard:
    core_v1, _ = await _get_k8s_client()
    if core_v1 is None:
        return ContextShard(
            summary=f"K8s unavailable; mock: {req.service} may be failing",
            severity=0.5, entities=[req.service], timestamp="unknown"
        )
    try:
        pods = await core_v1.list_namespaced_pod(
            namespace=req.namespace,
            label_selector=f"app={req.service}"
        )
        if not pods.items:
            return ContextShard(
                summary=f"No pods found for {req.service} in {req.namespace}",
                severity=0.7, entities=[req.service], timestamp="recent"
            )
        pod = pods.items[0]
        restarts = sum(
            cs.restart_count for cs in (pod.status.container_statuses or [])
        )
        phase = pod.status.phase or "Unknown"
        reason = ""
        for cs in (pod.status.container_statuses or []):
            if cs.state and cs.state.waiting:
                reason = cs.state.waiting.reason or ""
            elif cs.last_state and cs.last_state.terminated:
                reason = cs.last_state.terminated.reason or ""
        severity = _compute_restart_severity(restarts)
        if reason in ("OOMKilled",):
            severity = max(severity, 0.9)
        summary = (
            f"{req.service} phase={phase} restarts={restarts}"
            + (f" reason={reason}" if reason else "")
        )
        return ContextShard(
            summary=summary[:300],
            severity=severity,
            entities=[req.service],
            timestamp="recent"
        )
    finally:
        await core_v1.api_client.close()

@app.post("/get_recent_events", response_model=ContextShard)
async def get_recent_events(req: K8sRequest) -> ContextShard:
    core_v1, _ = await _get_k8s_client()
    if core_v1 is None:
        return ContextShard(
            summary=f"No events available (mock mode) for {req.service}",
            severity=0.4, entities=[req.service], timestamp="unknown"
        )
    try:
        events = await core_v1.list_namespaced_event(
            namespace=req.namespace,
            field_selector=f"involvedObject.name={req.service}"
        )
        warning_events = [
            e for e in events.items if e.type == "Warning"
        ]
        if not warning_events:
            return ContextShard(
                summary=f"No warning events for {req.service}",
                severity=0.1, entities=[req.service], timestamp="recent"
            )
        top = warning_events[:3]
        msgs = "; ".join(
            f"{e.reason}: {(e.message or '')[:60]}" for e in top
        )
        return ContextShard(
            summary=msgs[:300],
            severity=0.7,
            entities=[req.service],
            timestamp="recent"
        )
    finally:
        await core_v1.api_client.close()

@app.post("/get_deployment_changes", response_model=ContextShard)
async def get_deployment_changes(req: K8sRequest) -> ContextShard:
    _, apps_v1 = await _get_k8s_client()
    if apps_v1 is None:
        return ContextShard(
            summary=f"Deployment history unavailable (mock) for {req.service}",
            severity=0.3, entities=[req.service], timestamp="unknown"
        )
    try:
        dep = await apps_v1.read_namespaced_deployment(
            name=req.service, namespace=req.namespace
        )
        gen = dep.metadata.generation or 0
        obs = dep.status.observed_generation or 0
        available = dep.status.available_replicas or 0
        desired = dep.spec.replicas or 1
        if gen != obs or available < desired:
            severity = 0.8
            summary = (
                f"{req.service} gen={gen} observed={obs} "
                f"available={available}/{desired} — possible rollout issue"
            )
        else:
            severity = 0.1
            summary = f"{req.service} deployment healthy gen={gen}"
        return ContextShard(
            summary=summary[:300],
            severity=severity,
            entities=[req.service],
            timestamp="recent"
        )
    finally:
        await apps_v1.api_client.close()
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_mcp_servers.py -v
```
Expected: PASSED (mock path taken since no cluster available)

- [ ] **Step 5: Commit**

```bash
git add src/mcp_k8s_server.py tests/test_mcp_servers.py
git commit -m "feat: add MCP K8s server with pod status, events, deployment endpoints"
```

---

## Task 3: MCP Logs Server

**Files:**
- Create: `src/mcp_logs_server.py`
- Modify: `tests/test_mcp_servers.py` (add log server tests)

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_mcp_servers.py
from src.mcp_logs_server import app as logs_app

@pytest.mark.asyncio
async def test_get_logs_summary_returns_shard():
    transport = ASGITransport(app=logs_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/get_logs_summary", json={
            "service": "payment-svc", "namespace": "production"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert len(data["summary"].split()) < 120

@pytest.mark.asyncio
async def test_get_error_patterns_returns_shard():
    transport = ASGITransport(app=logs_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/get_error_patterns", json={
            "service": "payment-svc", "namespace": "production"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "entities" in data
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_mcp_servers.py::test_get_logs_summary_returns_shard -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `src/mcp_logs_server.py`**

```python
import re
from collections import Counter
from fastapi import FastAPI
from pydantic import BaseModel
from src.models import ContextShard

app = FastAPI(title="MCP Logs Server")

class LogsRequest(BaseModel):
    service: str
    namespace: str
    tail_lines: int = 100

ERROR_PATTERNS = [
    (re.compile(r"OOMKilled|exit code 137", re.I), "OOMKilled", 0.95),
    (re.compile(r"ECONNREFUSED|connection refused", re.I), "ConnectionRefused", 0.85),
    (re.compile(r"timeout|timed out", re.I), "Timeout", 0.75),
    (re.compile(r"ImagePullBackOff|ErrImagePull", re.I), "ImagePullBackOff", 0.90),
    (re.compile(r"CrashLoopBackOff", re.I), "CrashLoopBackOff", 0.90),
    (re.compile(r"permission denied|unauthorized|403|401", re.I), "AuthError", 0.70),
    (re.compile(r"no such file|not found|404", re.I), "NotFound", 0.60),
    (re.compile(r"out of memory|oom", re.I), "OOM", 0.90),
]

async def _fetch_pod_logs(service: str, namespace: str, tail: int) -> str:
    """Fetch logs via kubernetes-asyncio; fallback to empty on error."""
    try:
        from kubernetes_asyncio import client, config as k8s_config
        try:
            await k8s_config.load_incluster_config()
        except Exception:
            await k8s_config.load_kube_config()
        v1 = client.CoreV1Api()
        try:
            pods = await v1.list_namespaced_pod(
                namespace=namespace, label_selector=f"app={service}"
            )
            if not pods.items:
                return ""
            pod_name = pods.items[0].metadata.name
            log = await v1.read_namespaced_pod_log(
                name=pod_name, namespace=namespace,
                tail_lines=tail, previous=True
            )
            return log or ""
        finally:
            await v1.api_client.close()
    except Exception:
        return ""

def _compress_logs(raw: str) -> tuple[str, list[str], float]:
    """Return (summary, entities, severity). Keep signal, drop noise."""
    lines = raw.splitlines()
    matched = []
    entities: set[str] = set()
    max_severity = 0.1

    for line in lines:
        for pattern, label, sev in ERROR_PATTERNS:
            if pattern.search(line):
                matched.append(label)
                max_severity = max(max_severity, sev)
                # Extract probable service names (word:port pattern)
                hosts = re.findall(r'[\w.-]+:\d{2,5}', line)
                entities.update(hosts[:2])
                break

    if not matched:
        return "No critical errors detected in logs", [], 0.1

    counts = Counter(matched)
    parts = [f"{label}x{cnt}" for label, cnt in counts.most_common(4)]
    summary = "Log errors: " + ", ".join(parts)
    return summary[:300], list(entities)[:5], max_severity

@app.post("/get_logs_summary", response_model=ContextShard)
async def get_logs_summary(req: LogsRequest) -> ContextShard:
    raw = await _fetch_pod_logs(req.service, req.namespace, req.tail_lines)
    summary, entities, severity = _compress_logs(raw)
    return ContextShard(
        summary=summary,
        severity=severity,
        entities=[req.service] + entities,
        timestamp="recent"
    )

@app.post("/get_error_patterns", response_model=ContextShard)
async def get_error_patterns(req: LogsRequest) -> ContextShard:
    raw = await _fetch_pod_logs(req.service, req.namespace, req.tail_lines)
    lines = raw.splitlines()
    error_lines = [l for l in lines if re.search(r'error|exception|fatal|panic', l, re.I)]
    if not error_lines:
        return ContextShard(
            summary="No error patterns found",
            severity=0.1,
            entities=[req.service],
            timestamp="recent"
        )
    # Deduplicate by stripping timestamps and taking unique messages
    cleaned = list({re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s]*\s*', '', l).strip()
                    for l in error_lines})[:5]
    summary = " | ".join(cleaned)[:300]
    _, entities, severity = _compress_logs(raw)
    return ContextShard(
        summary=summary,
        severity=severity,
        entities=[req.service] + entities,
        timestamp="recent"
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_mcp_servers.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mcp_logs_server.py tests/test_mcp_servers.py
git commit -m "feat: add MCP Logs server with compression and error pattern detection"
```

---

## Task 4: MCP Metrics Server (Optional)

**Files:**
- Create: `src/mcp_metrics_server.py`
- Modify: `tests/test_mcp_servers.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_mcp_servers.py
from src.mcp_metrics_server import app as metrics_app

@pytest.mark.asyncio
async def test_get_resource_anomalies_returns_shard():
    transport = ASGITransport(app=metrics_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/get_resource_anomalies", json={
            "service": "payment-svc", "namespace": "production"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "severity" in data
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_mcp_servers.py::test_get_resource_anomalies_returns_shard -v
```

- [ ] **Step 3: Write `src/mcp_metrics_server.py`**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.models import ContextShard

app = FastAPI(title="MCP Metrics Server")

class MetricsRequest(BaseModel):
    service: str
    namespace: str

async def _query_metrics_api(service: str, namespace: str) -> dict:
    """
    In production: query metrics-server or Prometheus via in-cluster API.
    Returns dict with cpu_usage_ratio and memory_usage_ratio (0.0-1.0).
    Fallback: return neutral values when metrics unavailable.
    """
    try:
        from kubernetes_asyncio import client, config as k8s_config
        try:
            await k8s_config.load_incluster_config()
        except Exception:
            await k8s_config.load_kube_config()
        # kubernetes metrics-server API
        custom = client.CustomObjectsApi()
        try:
            result = await custom.get_namespaced_custom_object(
                group="metrics.k8s.io", version="v1beta1",
                namespace=namespace, plural="pods", name=service
            )
            containers = result.get("containers", [])
            if containers:
                cpu_str = containers[0].get("usage", {}).get("cpu", "0")
                mem_str = containers[0].get("usage", {}).get("memory", "0")
                # Rough normalization (millicores / 1000, Mi / 1024)
                cpu = float(cpu_str.replace("n", "")) / 1e9 if "n" in cpu_str else float(cpu_str.replace("m", "")) / 1000
                mem = float(mem_str.replace("Ki", "")) / (1024 * 1024) if "Ki" in mem_str else 0.0
                return {"cpu_ratio": min(cpu, 1.0), "mem_ratio": min(mem, 1.0)}
        except Exception:
            pass
        finally:
            await custom.api_client.close()
    except Exception:
        pass
    return {"cpu_ratio": 0.0, "mem_ratio": 0.0}

@app.post("/get_resource_anomalies", response_model=ContextShard)
async def get_resource_anomalies(req: MetricsRequest) -> ContextShard:
    metrics = await _query_metrics_api(req.service, req.namespace)
    cpu = metrics["cpu_ratio"]
    mem = metrics["mem_ratio"]
    anomalies = []
    severity = 0.1
    if cpu > 0.85:
        anomalies.append(f"CPU spike {cpu:.0%}")
        severity = max(severity, 0.75)
    if mem > 0.85:
        anomalies.append(f"Memory high {mem:.0%}")
        severity = max(severity, 0.85)
    if mem > 0.95:
        anomalies.append("OOM imminent")
        severity = 0.95
    if not anomalies:
        summary = f"{req.service}: no resource anomalies detected"
    else:
        summary = f"{req.service}: " + ", ".join(anomalies)
    return ContextShard(
        summary=summary[:300],
        severity=severity,
        entities=[req.service],
        timestamp="recent"
    )
```

- [ ] **Step 4: Run all MCP tests**

```bash
python -m pytest tests/test_mcp_servers.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mcp_metrics_server.py tests/test_mcp_servers.py
git commit -m "feat: add optional MCP Metrics server for OOM/CPU anomaly detection"
```

---

## Task 5: Sub-Agents

**Files:**
- Create: `src/sub_agents.py`
- Create: `tests/test_sub_agents.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sub_agents.py
import pytest
from src.models import ContextShard, FailureQuery
from src.sub_agents import (
    log_analysis_agent,
    failure_classification_agent,
    root_cause_agent,
)

@pytest.fixture
def shards():
    return [
        ContextShard(summary="Pod restarted 9x OOMKilled exit 137",
                     severity=0.9, entities=["payment-svc"], timestamp="recent"),
        ContextShard(summary="Memory high 97% OOM imminent",
                     severity=0.95, entities=["payment-svc"], timestamp="recent"),
    ]

def test_log_analysis_extracts_signal(shards):
    result = log_analysis_agent(shards)
    assert isinstance(result, str)
    assert len(result.split()) < 100
    assert "OOM" in result or "memory" in result.lower() or "restart" in result.lower()

def test_failure_classification_oom(shards):
    result = failure_classification_agent(shards)
    assert result in [
        "CrashLoopBackOff", "OOMKilled", "ImagePullBackOff",
        "DependencyTimeout", "DeploymentRegression", "Unknown"
    ]
    assert result == "OOMKilled"

def test_failure_classification_dependency():
    shards = [ContextShard(
        summary="ECONNREFUSED redis:6379 connection refused 15 times",
        severity=0.8, entities=["payment-svc", "redis:6379"], timestamp="recent"
    )]
    result = failure_classification_agent(shards)
    assert result == "DependencyTimeout"

def test_root_cause_agent_produces_report(shards):
    from src.models import IncidentReport
    issue_type = "OOMKilled"
    signal = "OOMKilled x9, memory 97%"
    report = root_cause_agent(shards, issue_type, signal, FailureQuery(service="payment-svc", namespace="prod"))
    assert isinstance(report, IncidentReport)
    assert report.confidence > 0.5
    assert len(report.evidence) > 0
    assert len(report.suggested_fix) > 0
    import json
    token_estimate = len(json.dumps(report.model_dump()).split())
    assert token_estimate < 200
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_sub_agents.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `src/sub_agents.py`**

```python
"""
Sub-agents: pure, synchronous, deterministic — no LLM calls.
All classification logic is rule-based to stay under 2s latency budget.
"""
import re
from src.models import ContextShard, FailureQuery, IncidentReport

# ── 1. log-analysis-agent ──────────────────────────────────────────────────

def log_analysis_agent(shards: list[ContextShard]) -> str:
    """
    Compress shards into a single signal string ≤100 words.
    Returns highest-severity signals only.
    """
    sorted_shards = sorted(shards, key=lambda s: s.severity, reverse=True)
    top = sorted_shards[:3]
    parts = [f"[{s.severity:.2f}] {s.summary}" for s in top]
    return " | ".join(parts)[:400]


# ── 2. failure-classification-agent ───────────────────────────────────────

_CLASSIFICATION_RULES = [
    # (regex_on_combined_text, issue_type)
    (re.compile(r"OOMKilled|exit.?code.?137|out.?of.?memory|oom", re.I), "OOMKilled"),
    (re.compile(r"ImagePullBackOff|ErrImagePull|image.?pull", re.I), "ImagePullBackOff"),
    (re.compile(r"ECONNREFUSED|connection.?refused|timeout|unreachable", re.I), "DependencyTimeout"),
    (re.compile(r"CrashLoopBackOff", re.I), "CrashLoopBackOff"),
    (re.compile(r"rollout|generation.mismatch|available.*<.*desired|regression", re.I), "DeploymentRegression"),
]

def failure_classification_agent(shards: list[ContextShard]) -> str:
    """
    Return the most specific failure type from the shard summaries.
    Checks rules in priority order; highest-severity shard wins on tie.
    """
    sorted_shards = sorted(shards, key=lambda s: s.severity, reverse=True)
    combined = " ".join(s.summary for s in sorted_shards)

    for pattern, issue_type in _CLASSIFICATION_RULES:
        if pattern.search(combined):
            return issue_type
    return "Unknown"


# ── 3. root-cause-agent ───────────────────────────────────────────────────

_FIX_TEMPLATES: dict[str, str] = {
    "OOMKilled": (
        "Increase memory limit in deployment: "
        "`kubectl set resources deployment/{svc} --limits=memory=512Mi -n {ns}`"
        " or add HPA with memory trigger."
    ),
    "CrashLoopBackOff": (
        "Check logs: `kubectl logs {svc} -n {ns} --previous`. "
        "If config error, verify ConfigMap/Secret. Roll back if recent deploy: "
        "`kubectl rollout undo deployment/{svc} -n {ns}`"
    ),
    "ImagePullBackOff": (
        "Verify image tag exists and registry credentials: "
        "`kubectl describe pod -l app={svc} -n {ns}`. "
        "Fix imagePullSecret or correct image tag."
    ),
    "DependencyTimeout": (
        "Check dependency reachability: `kubectl exec -it <pod> -- nc -zv <host> <port>`. "
        "Verify Service/Endpoints exist. If post-deploy, roll back: "
        "`kubectl rollout undo deployment/{svc} -n {ns}`"
    ),
    "DeploymentRegression": (
        "Roll back: `kubectl rollout undo deployment/{svc} -n {ns}`. "
        "Check rollout status: `kubectl rollout status deployment/{svc} -n {ns}`"
    ),
    "Unknown": (
        "Inspect pod: `kubectl describe pod -l app={svc} -n {ns}` "
        "and `kubectl logs -l app={svc} -n {ns} --previous`"
    ),
}

def root_cause_agent(
    shards: list[ContextShard],
    issue_type: str,
    signal: str,
    query: FailureQuery,
) -> IncidentReport:
    """Produce final IncidentReport from classified failure and signal."""
    svc, ns = query.service, query.namespace

    # Evidence: top shard summaries, capped at 3
    sorted_shards = sorted(shards, key=lambda s: s.severity, reverse=True)
    evidence = [s.summary[:120] for s in sorted_shards[:3]]

    # Confidence from highest severity shard
    confidence = sorted_shards[0].severity if sorted_shards else 0.5
    confidence = round(min(confidence, 1.0), 2)

    # Root cause: derive from signal
    root_cause = _derive_root_cause(issue_type, signal, svc)

    # Fix: fill template
    template = _FIX_TEMPLATES.get(issue_type, _FIX_TEMPLATES["Unknown"])
    suggested_fix = template.format(svc=svc, ns=ns)[:200]

    return IncidentReport(
        issue_type=issue_type,  # type: ignore[arg-type]
        root_cause=root_cause[:300],
        confidence=confidence,
        evidence=evidence,
        suggested_fix=suggested_fix,
    )

def _derive_root_cause(issue_type: str, signal: str, service: str) -> str:
    causes = {
        "OOMKilled": f"{service} exceeded its memory limit and was killed by the kernel (OOMKiller).",
        "CrashLoopBackOff": f"{service} is crashing on startup repeatedly; likely bad config or unmet dependency.",
        "ImagePullBackOff": f"Container image for {service} cannot be pulled; tag missing or registry auth failed.",
        "DependencyTimeout": f"{service} cannot reach a required dependency (connection refused/timeout).",
        "DeploymentRegression": f"A recent deployment of {service} broke availability; pods not reaching ready state.",
        "Unknown": f"Failure pattern for {service} unclassified; manual inspection required.",
    }
    base = causes.get(issue_type, causes["Unknown"])
    # Append any specific entity from signal
    entities = re.findall(r'[\w.-]+:\d{2,5}', signal)
    if entities:
        base += f" Involved: {', '.join(entities[:2])}."
    return base
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_sub_agents.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/sub_agents.py tests/test_sub_agents.py
git commit -m "feat: add rule-based sub-agents for log analysis, classification, root-cause"
```

---

## Task 6: Context Weaver Orchestrator

**Files:**
- Create: `src/context_weaver.py`
- Create: `tests/test_context_weaver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_context_weaver.py
import pytest
from unittest.mock import AsyncMock, patch
from src.models import FailureQuery, ContextShard, IncidentReport
from src.context_weaver import ContextWeaverBrain, compute_failure_gravity

def test_failure_gravity_above_threshold():
    score = compute_failure_gravity(
        recency=1.0, restart_count=0.9, severity=0.9, blast_radius=0.8
    )
    assert score > 0.65

def test_failure_gravity_below_threshold():
    score = compute_failure_gravity(
        recency=0.1, restart_count=0.1, severity=0.1, blast_radius=0.1
    )
    assert score < 0.65

@pytest.mark.asyncio
async def test_orchestrator_returns_incident_report():
    mock_shard = ContextShard(
        summary="Pod restarted 9x OOMKilled exit 137 memory 97%",
        severity=0.92, entities=["payment-svc"], timestamp="recent"
    )
    brain = ContextWeaverBrain()
    with patch.object(brain, "_call_mcp_tool", new=AsyncMock(return_value=mock_shard)):
        report = await brain.run(FailureQuery(service="payment-svc", namespace="production"))
    assert isinstance(report, IncidentReport)
    assert report.issue_type in [
        "OOMKilled", "CrashLoopBackOff", "DependencyTimeout",
        "ImagePullBackOff", "DeploymentRegression", "Unknown"
    ]
    assert report.confidence > 0.0

@pytest.mark.asyncio
async def test_orchestrator_max_3_tool_calls():
    """Ensure THINK loop never exceeds 3 tool calls."""
    call_count = 0
    async def counting_mock(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return ContextShard(
            summary="some error", severity=0.7,
            entities=["svc"], timestamp="recent"
        )
    brain = ContextWeaverBrain()
    with patch.object(brain, "_call_mcp_tool", new=counting_mock):
        await brain.run(FailureQuery(service="test-svc", namespace="default"))
    assert call_count <= 3
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_context_weaver.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write `src/context_weaver.py`**

```python
"""
context-weaver-brain: K-Agent orchestrator.
THINK → ACT (≤3 MCP calls) → OBSERVE → DECIDE
"""
import asyncio
from typing import Literal
import httpx
from fastapi import FastAPI
from src.models import AgentState, ContextShard, FailureQuery, IncidentReport
from src.sub_agents import (
    log_analysis_agent,
    failure_classification_agent,
    root_cause_agent,
)
from src.config import settings

app = FastAPI(title="Context Weaver Brain")

MCPTool = Literal[
    "k8s/get_pod_status",
    "k8s/get_recent_events",
    "k8s/get_deployment_changes",
    "logs/get_logs_summary",
    "logs/get_error_patterns",
    "metrics/get_resource_anomalies",
]

_TOOL_URLS: dict[str, tuple[str, str]] = {
    "k8s/get_pod_status":          (settings.mcp_k8s_url,      "/get_pod_status"),
    "k8s/get_recent_events":       (settings.mcp_k8s_url,      "/get_recent_events"),
    "k8s/get_deployment_changes":  (settings.mcp_k8s_url,      "/get_deployment_changes"),
    "logs/get_logs_summary":       (settings.mcp_logs_url,     "/get_logs_summary"),
    "logs/get_error_patterns":     (settings.mcp_logs_url,     "/get_error_patterns"),
    "metrics/get_resource_anomalies": (settings.mcp_metrics_url, "/get_resource_anomalies"),
}

def compute_failure_gravity(
    recency: float, restart_count: float, severity: float, blast_radius: float
) -> float:
    return (
        recency * 0.35
        + restart_count * 0.25
        + severity * 0.25
        + blast_radius * 0.15
    )

class ContextWeaverBrain:
    async def _call_mcp_tool(self, tool: str, payload: dict) -> ContextShard:
        base_url, path = _TOOL_URLS[tool]
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.post(f"{base_url}{path}", json=payload)
            resp.raise_for_status()
            return ContextShard(**resp.json())

    async def _think(self, state: AgentState) -> list[str]:
        """Return ordered list of MCP tools to call based on current state."""
        if state.tool_calls_used >= state.max_tool_calls:
            return []
        if not state.shards:
            # Cold start: always get pod status + logs summary first
            return ["k8s/get_pod_status", "logs/get_logs_summary"]
        max_sev = max(s.severity for s in state.shards)
        if max_sev < 0.5:
            return []
        # If already have high-sev signal, optionally add metrics
        if max_sev > 0.8 and state.tool_calls_used < 2:
            return ["metrics/get_resource_anomalies"]
        return []

    async def run(self, query: FailureQuery) -> IncidentReport:
        state = AgentState(query=query)
        payload = {"service": query.service, "namespace": query.namespace}

        # THINK → ACT → OBSERVE loop (max 3 iterations)
        while state.tool_calls_used < state.max_tool_calls:
            tools_to_call = await self._think(state)
            if not tools_to_call:
                break

            remaining = state.max_tool_calls - state.tool_calls_used
            tools_to_call = tools_to_call[:remaining]

            shards = await asyncio.gather(
                *[self._call_mcp_tool(t, payload) for t in tools_to_call],
                return_exceptions=True
            )
            for shard in shards:
                if isinstance(shard, ContextShard):
                    state.shards.append(shard)
            state.tool_calls_used += len(tools_to_call)

        # DECIDE
        if not state.shards:
            return IncidentReport(
                issue_type="Unknown",
                root_cause=f"No data available for {query.service}",
                confidence=0.0,
                evidence=[],
                suggested_fix=(
                    f"kubectl describe pod -l app={query.service} "
                    f"-n {query.namespace}"
                ),
            )

        # Compute failure gravity for surfacing
        max_sev = max(s.severity for s in state.shards)
        gravity = compute_failure_gravity(
            recency=1.0,
            restart_count=min(max_sev, 1.0),
            severity=max_sev,
            blast_radius=0.7,
        )
        if gravity < settings.failure_gravity_threshold:
            return IncidentReport(
                issue_type="Unknown",
                root_cause=f"{query.service} appears healthy (gravity={gravity:.2f})",
                confidence=1.0 - gravity,
                evidence=[s.summary[:80] for s in state.shards[:2]],
                suggested_fix="No action required.",
            )

        signal = log_analysis_agent(state.shards)
        issue_type = failure_classification_agent(state.shards)
        return root_cause_agent(state.shards, issue_type, signal, query)


brain = ContextWeaverBrain()

@app.post("/diagnose", response_model=IncidentReport)
async def diagnose(query: FailureQuery) -> IncidentReport:
    return await brain.run(query)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_context_weaver.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/context_weaver.py tests/test_context_weaver.py
git commit -m "feat: add context-weaver K-Agent orchestrator with failure gravity scoring"
```

---

## Task 7: Dockerfile and docker-compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
# ── builder ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── runtime ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "src.context_weaver:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
version: "3.9"

services:
  context-weaver:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - IN_CLUSTER=false
      - MCP_K8S_URL=http://mcp-k8s-server:8001
      - MCP_LOGS_URL=http://mcp-logs-server:8002
      - MCP_METRICS_URL=http://mcp-metrics-server:8003
    depends_on:
      - mcp-k8s-server
      - mcp-logs-server
      - mcp-metrics-server

  mcp-k8s-server:
    build: .
    command: uvicorn src.mcp_k8s_server:app --host 0.0.0.0 --port 8001
    ports:
      - "8001:8001"
    environment:
      - IN_CLUSTER=false
    volumes:
      - ~/.kube:/root/.kube:ro

  mcp-logs-server:
    build: .
    command: uvicorn src.mcp_logs_server:app --host 0.0.0.0 --port 8002
    ports:
      - "8002:8002"
    environment:
      - IN_CLUSTER=false
    volumes:
      - ~/.kube:/root/.kube:ro

  mcp-metrics-server:
    build: .
    command: uvicorn src.mcp_metrics_server:app --host 0.0.0.0 --port 8003
    ports:
      - "8003:8003"
    environment:
      - IN_CLUSTER=false
    volumes:
      - ~/.kube:/root/.kube:ro
```

- [ ] **Step 3: Write `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
IN_CLUSTER=false
MCP_K8S_URL=http://localhost:8001
MCP_LOGS_URL=http://localhost:8002
MCP_METRICS_URL=http://localhost:8003
FAILURE_GRAVITY_THRESHOLD=0.65
MAX_TOOL_CALLS=3
```

- [ ] **Step 4: Verify Docker build**

```bash
docker build -t k8s-log-intelligence:dev .
```
Expected: Successfully built

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .env.example
git commit -m "feat: add multi-stage Dockerfile and docker-compose for local dev"
```

---

## Task 8: Kubernetes Manifests

**Files:**
- Create: `kubernetes/configmap.yaml`
- Create: `kubernetes/rbac.yaml`
- Create: `kubernetes/deployment.yaml`

- [ ] **Step 1: Write `kubernetes/rbac.yaml`**

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: k8s-log-intelligence
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: k8s-log-intelligence-reader
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "events", "namespaces"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["metrics.k8s.io"]
    resources: ["pods", "nodes"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: k8s-log-intelligence-binding
subjects:
  - kind: ServiceAccount
    name: k8s-log-intelligence
    namespace: monitoring
roleRef:
  kind: ClusterRole
  name: k8s-log-intelligence-reader
  apiGroup: rbac.authorization.k8s.io
```

- [ ] **Step 2: Write `kubernetes/configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: k8s-log-intelligence-config
  namespace: monitoring
data:
  IN_CLUSTER: "true"
  MCP_K8S_URL: "http://mcp-k8s-server.monitoring.svc.cluster.local:8001"
  MCP_LOGS_URL: "http://mcp-logs-server.monitoring.svc.cluster.local:8002"
  MCP_METRICS_URL: "http://mcp-metrics-server.monitoring.svc.cluster.local:8003"
  FAILURE_GRAVITY_THRESHOLD: "0.65"
  MAX_TOOL_CALLS: "3"
```

- [ ] **Step 3: Write `kubernetes/deployment.yaml`**

```yaml
# ── Context Weaver Brain ──────────────────────────────────────────────────
apiVersion: apps/v1
kind: Deployment
metadata:
  name: context-weaver
  namespace: monitoring
  labels:
    app: context-weaver
spec:
  replicas: 1
  selector:
    matchLabels:
      app: context-weaver
  template:
    metadata:
      labels:
        app: context-weaver
    spec:
      serviceAccountName: k8s-log-intelligence
      containers:
        - name: context-weaver
          image: k8s-log-intelligence:latest
          command: ["uvicorn", "src.context_weaver:app", "--host", "0.0.0.0", "--port", "8000"]
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: k8s-log-intelligence-config
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: k8s-log-intelligence-secrets
                  key: anthropic_api_key
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "256Mi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: context-weaver
  namespace: monitoring
spec:
  selector:
    app: context-weaver
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
---
# ── MCP K8s Server ────────────────────────────────────────────────────────
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-k8s-server
  namespace: monitoring
  labels:
    app: mcp-k8s-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mcp-k8s-server
  template:
    metadata:
      labels:
        app: mcp-k8s-server
    spec:
      serviceAccountName: k8s-log-intelligence
      containers:
        - name: mcp-k8s-server
          image: k8s-log-intelligence:latest
          command: ["uvicorn", "src.mcp_k8s_server:app", "--host", "0.0.0.0", "--port", "8001"]
          ports:
            - containerPort: 8001
          envFrom:
            - configMapRef:
                name: k8s-log-intelligence-config
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-k8s-server
  namespace: monitoring
spec:
  selector:
    app: mcp-k8s-server
  ports:
    - port: 8001
      targetPort: 8001
---
# ── MCP Logs Server ───────────────────────────────────────────────────────
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-logs-server
  namespace: monitoring
  labels:
    app: mcp-logs-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mcp-logs-server
  template:
    metadata:
      labels:
        app: mcp-logs-server
    spec:
      serviceAccountName: k8s-log-intelligence
      containers:
        - name: mcp-logs-server
          image: k8s-log-intelligence:latest
          command: ["uvicorn", "src.mcp_logs_server:app", "--host", "0.0.0.0", "--port", "8002"]
          ports:
            - containerPort: 8002
          envFrom:
            - configMapRef:
                name: k8s-log-intelligence-config
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-logs-server
  namespace: monitoring
spec:
  selector:
    app: mcp-logs-server
  ports:
    - port: 8002
      targetPort: 8002
---
# ── MCP Metrics Server ────────────────────────────────────────────────────
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-metrics-server
  namespace: monitoring
  labels:
    app: mcp-metrics-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mcp-metrics-server
  template:
    metadata:
      labels:
        app: mcp-metrics-server
    spec:
      serviceAccountName: k8s-log-intelligence
      containers:
        - name: mcp-metrics-server
          image: k8s-log-intelligence:latest
          command: ["uvicorn", "src.mcp_metrics_server:app", "--host", "0.0.0.0", "--port", "8003"]
          ports:
            - containerPort: 8003
          envFrom:
            - configMapRef:
                name: k8s-log-intelligence-config
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-metrics-server
  namespace: monitoring
spec:
  selector:
    app: mcp-metrics-server
  ports:
    - port: 8003
      targetPort: 8003
```

- [ ] **Step 4: Validate manifests**

```bash
kubectl apply --dry-run=client -f kubernetes/
```
Expected: All resources configured (dry-run)

- [ ] **Step 5: Commit**

```bash
git add kubernetes/
git commit -m "feat: add K8s deployment, RBAC, service, and configmap manifests"
```

---

## Task 9: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""
Integration tests: wire context-weaver + sub-agents + mock MCP shards.
No real K8s required.
"""
import pytest
from unittest.mock import AsyncMock, patch
from src.context_weaver import ContextWeaverBrain
from src.models import ContextShard, FailureQuery, IncidentReport

SCENARIOS = [
    (
        "oom-scenario",
        FailureQuery(service="payment-svc", namespace="production"),
        [
            ContextShard(summary="payment-svc phase=Running restarts=9 reason=OOMKilled",
                         severity=0.92, entities=["payment-svc"], timestamp="recent"),
            ContextShard(summary="Log errors: OOMKilledx9, memory 97%",
                         severity=0.95, entities=["payment-svc"], timestamp="recent"),
        ],
        "OOMKilled",
    ),
    (
        "redis-down-scenario",
        FailureQuery(service="order-svc", namespace="staging"),
        [
            ContextShard(summary="order-svc phase=Running restarts=5",
                         severity=0.75, entities=["order-svc"], timestamp="recent"),
            ContextShard(summary="ECONNREFUSED redis:6379 connection refused 12x",
                         severity=0.85, entities=["order-svc", "redis:6379"], timestamp="recent"),
        ],
        "DependencyTimeout",
    ),
    (
        "image-pull-scenario",
        FailureQuery(service="frontend", namespace="default"),
        [
            ContextShard(summary="frontend ImagePullBackOff ErrImagePull",
                         severity=0.90, entities=["frontend"], timestamp="recent"),
            ContextShard(summary="Log errors: ImagePullBackOffx3",
                         severity=0.88, entities=["frontend"], timestamp="recent"),
        ],
        "ImagePullBackOff",
    ),
]

@pytest.mark.asyncio
@pytest.mark.parametrize("name,query,mock_shards,expected_type", SCENARIOS)
async def test_scenario(name, query, mock_shards, expected_type):
    shard_iter = iter(mock_shards)

    async def mock_tool(*args, **kwargs):
        try:
            return next(shard_iter)
        except StopIteration:
            return mock_shards[-1]

    brain = ContextWeaverBrain()
    with patch.object(brain, "_call_mcp_tool", new=mock_tool):
        report = await brain.run(query)

    assert isinstance(report, IncidentReport), f"{name}: expected IncidentReport"
    assert report.issue_type == expected_type, (
        f"{name}: expected {expected_type}, got {report.issue_type}"
    )
    assert report.confidence > 0.6, f"{name}: confidence too low"
    assert len(report.evidence) > 0, f"{name}: no evidence"
    assert len(report.suggested_fix) > 0, f"{name}: no fix"
    import json
    token_count = len(json.dumps(report.model_dump()).split())
    assert token_count < 200, f"{name}: output {token_count} tokens exceeds budget"
```

- [ ] **Step 2: Run integration tests**

```bash
python -m pytest tests/test_integration.py -v
```
Expected: All 3 scenarios PASSED

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests PASSED, no failures

- [ ] **Step 4: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for OOM, Redis-down, ImagePull scenarios"
```

---

## Deployment Checklist

- [ ] `kubectl create namespace monitoring`
- [ ] `kubectl create secret generic k8s-log-intelligence-secrets --from-literal=anthropic_api_key=$ANTHROPIC_API_KEY -n monitoring`
- [ ] Build and push image: `docker build -t <registry>/k8s-log-intelligence:latest . && docker push ...`
- [ ] Update image in `kubernetes/deployment.yaml`
- [ ] `kubectl apply -f kubernetes/`
- [ ] `kubectl rollout status deployment/context-weaver -n monitoring`
- [ ] Test: `curl http://<svc-ip>/diagnose -d '{"service":"my-svc","namespace":"production"}'`
