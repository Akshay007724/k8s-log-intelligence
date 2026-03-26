import re as _re
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class ContextShard(BaseModel):
    summary: str = Field(max_length=500)
    severity: float = Field(ge=0.0, le=1.0)
    entities: list[str]
    timestamp: str

class FailureQuery(BaseModel):
    service: str
    namespace: str

    @field_validator("service", "namespace")
    @classmethod
    def validate_k8s_name(cls, v: str) -> str:
        if not _re.match(r'^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$', v) or len(v) > 63:
            raise ValueError(
                "Must be a valid Kubernetes DNS label (lowercase alphanumeric and hyphens)"
            )
        return v

class IncidentReport(BaseModel):
    issue_type: Literal[
        "CrashLoopBackOff", "OOMKilled", "ImagePullBackOff",
        "DependencyTimeout", "DeploymentRegression", "Unknown"
    ]
    root_cause: str = Field(max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default=[], max_length=3)
    suggested_fix: str = Field(max_length=200)

class AgentState(BaseModel):
    query: FailureQuery
    shards: list[ContextShard] = []
    tool_calls_used: int = 0
    max_tool_calls: int = 3
