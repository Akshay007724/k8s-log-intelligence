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
