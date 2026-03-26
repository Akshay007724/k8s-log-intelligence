"""
context-weaver-brain: K-Agent orchestrator.
THINK → ACT (≤3 MCP calls) → OBSERVE → DECIDE
"""
import asyncio
import logging
from typing import Literal

logger = logging.getLogger(__name__)
import httpx
from fastapi import FastAPI
from src.models import AgentState, ContextShard, FailureQuery, IncidentReport
from src.sub_agents import (
    log_analysis_agent,
    failure_classification_agent,
    root_cause_agent,
)
from src.config import settings

app = FastAPI(title="Context Weaver Brain")

MCPTool = Literal[
    "k8s/get_pod_status",
    "k8s/get_recent_events",
    "k8s/get_deployment_changes",
    "logs/get_logs_summary",
    "logs/get_error_patterns",
    "metrics/get_resource_anomalies",
]

_TOOL_URLS: dict[str, tuple[str, str]] = {
    "k8s/get_pod_status":             (settings.mcp_k8s_url,     "/get_pod_status"),
    "k8s/get_recent_events":          (settings.mcp_k8s_url,     "/get_recent_events"),
    "k8s/get_deployment_changes":     (settings.mcp_k8s_url,     "/get_deployment_changes"),
    "logs/get_logs_summary":          (settings.mcp_logs_url,    "/get_logs_summary"),
    "logs/get_error_patterns":        (settings.mcp_logs_url,    "/get_error_patterns"),
    "metrics/get_resource_anomalies": (settings.mcp_metrics_url, "/get_resource_anomalies"),
}

def compute_failure_gravity(
    recency: float, restart_count: float, severity: float, blast_radius: float
) -> float:
    return (
        recency * 0.35
        + restart_count * 0.25
        + severity * 0.25
        + blast_radius * 0.15
    )

class ContextWeaverBrain:
    async def _call_mcp_tool(self, tool: str, payload: dict) -> ContextShard:
        base_url, path = _TOOL_URLS[tool]
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.post(f"{base_url}{path}", json=payload)
            resp.raise_for_status()
            return ContextShard(**resp.json())

    async def _think(self, state: AgentState) -> list[str]:
        """Return ordered list of MCP tools to call based on current state."""
        if state.tool_calls_used >= state.max_tool_calls:
            return []
        if not state.shards:
            # Cold start: always get pod status + logs summary first
            return ["k8s/get_pod_status", "logs/get_logs_summary"]
        max_sev = max(s.severity for s in state.shards)
        if max_sev < 0.5:
            return []
        # High-severity signal → optionally add metrics
        if max_sev > 0.8 and state.tool_calls_used < 2:
            return ["metrics/get_resource_anomalies"]
        return []

    async def run(self, query: FailureQuery) -> IncidentReport:
        state = AgentState(query=query)
        payload = {"service": query.service, "namespace": query.namespace}

        # THINK → ACT → OBSERVE loop (max 3 iterations)
        while state.tool_calls_used < state.max_tool_calls:
            tools_to_call = await self._think(state)
            if not tools_to_call:
                break

            remaining = state.max_tool_calls - state.tool_calls_used
            tools_to_call = tools_to_call[:remaining]

            shards = await asyncio.gather(
                *[self._call_mcp_tool(t, payload) for t in tools_to_call],
                return_exceptions=True
            )
            for shard in shards:
                if isinstance(shard, ContextShard):
                    state.shards.append(shard)
                elif isinstance(shard, Exception):
                    logger.warning("MCP tool call failed: %s", shard)
            state.tool_calls_used += len(tools_to_call)

        # DECIDE
        if not state.shards:
            return IncidentReport(
                issue_type="Unknown",
                root_cause=f"No data available for {query.service}",
                confidence=0.0,
                evidence=[],
                suggested_fix=(
                    f"kubectl describe pod -l app={query.service} "
                    f"-n {query.namespace}"
                ),
            )

        max_sev = max(s.severity for s in state.shards)
        # recency: 1.0 = all shards are "recent" (data from current query)
        # restart_count: derived from max severity (proxy for how bad the restart pattern is)
        # severity: max severity from shards
        # blast_radius: fraction of shards with severity > 0.7 (proxy for breadth of impact)
        blast_radius = sum(1 for s in state.shards if s.severity > 0.7) / max(len(state.shards), 1)
        gravity = compute_failure_gravity(
            recency=1.0,
            restart_count=min(max_sev, 1.0),
            severity=max_sev,
            blast_radius=blast_radius,
        )
        if gravity < settings.failure_gravity_threshold:
            return IncidentReport(
                issue_type="Unknown",
                root_cause=f"{query.service} appears healthy (gravity={gravity:.2f})",
                confidence=round(1.0 - gravity, 2),
                evidence=[s.summary[:80] for s in state.shards[:2]],
                suggested_fix="No action required.",
            )

        signal = log_analysis_agent(state.shards)
        issue_type = failure_classification_agent(state.shards)
        return root_cause_agent(state.shards, issue_type, signal, query)


brain = ContextWeaverBrain()

@app.post("/diagnose", response_model=IncidentReport)
async def diagnose(query: FailureQuery) -> IncidentReport:
    try:
        return await asyncio.wait_for(brain.run(query), timeout=2.0)
    except asyncio.TimeoutError:
        return IncidentReport(
            issue_type="Unknown",
            root_cause=f"Diagnosis timed out for {query.service}",
            confidence=0.0,
            evidence=[],
            suggested_fix=(
                f"kubectl describe pod -l app={query.service} -n {query.namespace}"
            ),
        )

@app.get("/health")
async def health():
    return {"status": "ok"}
