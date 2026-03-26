import pytest
from httpx import AsyncClient, ASGITransport
from src.mcp_k8s_server import app as k8s_app
from src.mcp_logs_server import app as logs_app
from src.mcp_metrics_server import app as metrics_app

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

@pytest.mark.asyncio
async def test_get_deployment_changes_returns_shard():
    transport = ASGITransport(app=k8s_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/get_deployment_changes", json={
            "service": "payment-svc", "namespace": "production"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "severity" in data

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
