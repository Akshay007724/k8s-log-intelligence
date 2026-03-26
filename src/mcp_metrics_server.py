from fastapi import FastAPI
from pydantic import BaseModel
from src.models import ContextShard

app = FastAPI(title="MCP Metrics Server")

class MetricsRequest(BaseModel):
    service: str
    namespace: str

async def _query_metrics_api(service: str, namespace: str) -> dict:
    """
    Query kubernetes metrics-server API.
    Returns dict with cpu_ratio and mem_ratio (0.0-1.0).
    Falls back to neutral values when metrics unavailable.
    """
    try:
        from kubernetes_asyncio import client, config as k8s_config
        try:
            await k8s_config.load_incluster_config()
        except Exception:
            await k8s_config.load_kube_config()
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
                if "n" in cpu_str:
                    cpu = float(cpu_str.replace("n", "")) / 1e9
                else:
                    cpu = float(cpu_str.replace("m", "")) / 1000
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
