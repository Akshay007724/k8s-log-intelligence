# tests/test_integration.py
"""
Integration tests: wire context-weaver + sub-agents + mock MCP shards.
No real K8s required.
"""
import json
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
    token_count = len(json.dumps(report.model_dump()).split())
    assert token_count < 200, f"{name}: output {token_count} tokens exceeds budget"
