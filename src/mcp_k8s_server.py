from fastapi import FastAPI
from pydantic import BaseModel
from src.models import ContextShard

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
