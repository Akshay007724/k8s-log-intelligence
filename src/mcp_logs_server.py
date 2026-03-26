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
