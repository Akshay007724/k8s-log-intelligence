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
