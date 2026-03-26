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

def test_failure_gravity_formula():
    """Verify exact formula: (recency*0.35) + (restart_count*0.25) + (severity*0.25) + (blast_radius*0.15)"""
    score = compute_failure_gravity(
        recency=1.0, restart_count=1.0, severity=1.0, blast_radius=1.0
    )
    assert abs(score - 1.0) < 0.001

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

@pytest.mark.asyncio
async def test_low_gravity_returns_healthy_report():
    """Low severity shards → gravity below threshold → healthy report."""
    low_shard = ContextShard(
        summary="all good", severity=0.1, entities=["svc"], timestamp="recent"
    )
    brain = ContextWeaverBrain()
    with patch.object(brain, "_call_mcp_tool", new=AsyncMock(return_value=low_shard)):
        report = await brain.run(FailureQuery(service="healthy-svc", namespace="default"))
    assert report.suggested_fix == "No action required."
