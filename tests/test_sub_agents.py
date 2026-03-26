import pytest
from src.models import ContextShard, FailureQuery
from src.sub_agents import (
    log_analysis_agent,
    failure_classification_agent,
    root_cause_agent,
)

@pytest.fixture
def oom_shards():
    return [
        ContextShard(summary="Pod restarted 9x OOMKilled exit 137",
                     severity=0.9, entities=["payment-svc"], timestamp="recent"),
        ContextShard(summary="Memory high 97% OOM imminent",
                     severity=0.95, entities=["payment-svc"], timestamp="recent"),
    ]

def test_log_analysis_extracts_signal(oom_shards):
    result = log_analysis_agent(oom_shards)
    assert isinstance(result, str)
    assert len(result.split()) < 100
    assert any(word in result.lower() for word in ["oom", "memory", "restart", "0.9", "0.95"])

def test_failure_classification_oom(oom_shards):
    result = failure_classification_agent(oom_shards)
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

def test_failure_classification_image_pull():
    shards = [ContextShard(
        summary="ImagePullBackOff ErrImagePull registry.io/app:v2.3",
        severity=0.88, entities=["frontend"], timestamp="recent"
    )]
    result = failure_classification_agent(shards)
    assert result == "ImagePullBackOff"

def test_root_cause_agent_produces_report(oom_shards):
    from src.models import IncidentReport
    report = root_cause_agent(
        oom_shards, "OOMKilled", "OOMKilledx9, memory 97%",
        FailureQuery(service="payment-svc", namespace="prod")
    )
    assert isinstance(report, IncidentReport)
    assert report.confidence > 0.5
    assert len(report.evidence) > 0
    assert "memory" in report.suggested_fix.lower() or "limit" in report.suggested_fix.lower()
    import json
    token_count = len(json.dumps(report.model_dump()).split())
    assert token_count < 200

def test_root_cause_agent_fix_mentions_service(oom_shards):
    report = root_cause_agent(
        oom_shards, "OOMKilled", "OOMKilledx9",
        FailureQuery(service="payment-svc", namespace="prod")
    )
    assert "payment-svc" in report.suggested_fix
