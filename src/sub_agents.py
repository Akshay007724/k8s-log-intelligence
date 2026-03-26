"""
Sub-agents: pure, synchronous, deterministic — no LLM calls.
All classification logic is rule-based to stay under 2s latency budget.
"""
import re
from src.models import ContextShard, FailureQuery, IncidentReport

# ── 1. log-analysis-agent ──────────────────────────────────────────────────

def log_analysis_agent(shards: list[ContextShard]) -> str:
    """
    Compress shards into a single signal string ≤100 words.
    Returns highest-severity signals only.
    """
    sorted_shards = sorted(shards, key=lambda s: s.severity, reverse=True)
    top = sorted_shards[:3]
    parts = [f"[{s.severity:.2f}] {s.summary}" for s in top]
    return " | ".join(parts)[:400]


# ── 2. failure-classification-agent ───────────────────────────────────────

_CLASSIFICATION_RULES = [
    # (regex_on_combined_text, issue_type)
    (re.compile(r"OOMKilled|exit.?code.?137|out.?of.?memory|oom", re.I), "OOMKilled"),
    (re.compile(r"ImagePullBackOff|ErrImagePull|image.?pull", re.I), "ImagePullBackOff"),
    (re.compile(r"ECONNREFUSED|connection.?refused|timeout|unreachable", re.I), "DependencyTimeout"),
    (re.compile(r"CrashLoopBackOff", re.I), "CrashLoopBackOff"),
    (re.compile(r"rollout|generation.mismatch|available.*<.*desired|regression", re.I), "DeploymentRegression"),
]

def failure_classification_agent(shards: list[ContextShard]) -> str:
    """
    Return the most specific failure type from the shard summaries.
    Checks rules in priority order; highest-severity shard wins on tie.
    """
    sorted_shards = sorted(shards, key=lambda s: s.severity, reverse=True)
    combined = " ".join(s.summary for s in sorted_shards)

    for pattern, issue_type in _CLASSIFICATION_RULES:
        if pattern.search(combined):
            return issue_type
    return "Unknown"


# ── 3. root-cause-agent ───────────────────────────────────────────────────

_FIX_TEMPLATES: dict[str, str] = {
    "OOMKilled": (
        "Increase memory limit in deployment: "
        "`kubectl set resources deployment/{svc} --limits=memory=512Mi -n {ns}`"
        " or add HPA with memory trigger."
    ),
    "CrashLoopBackOff": (
        "Check logs: `kubectl logs {svc} -n {ns} --previous`. "
        "If config error, verify ConfigMap/Secret. Roll back if recent deploy: "
        "`kubectl rollout undo deployment/{svc} -n {ns}`"
    ),
    "ImagePullBackOff": (
        "Verify image tag exists and registry credentials: "
        "`kubectl describe pod -l app={svc} -n {ns}`. "
        "Fix imagePullSecret or correct image tag."
    ),
    "DependencyTimeout": (
        "Check dependency reachability: `kubectl exec -it <pod> -- nc -zv <host> <port>`. "
        "Verify Service/Endpoints exist. If post-deploy, roll back: "
        "`kubectl rollout undo deployment/{svc} -n {ns}`"
    ),
    "DeploymentRegression": (
        "Roll back: `kubectl rollout undo deployment/{svc} -n {ns}`. "
        "Check rollout status: `kubectl rollout status deployment/{svc} -n {ns}`"
    ),
    "Unknown": (
        "Inspect pod: `kubectl describe pod -l app={svc} -n {ns}` "
        "and `kubectl logs -l app={svc} -n {ns} --previous`"
    ),
}

def root_cause_agent(
    shards: list[ContextShard],
    issue_type: str,
    signal: str,
    query: FailureQuery,
) -> IncidentReport:
    """Produce final IncidentReport from classified failure and signal."""
    svc, ns = query.service, query.namespace

    sorted_shards = sorted(shards, key=lambda s: s.severity, reverse=True)
    evidence = [s.summary[:120] for s in sorted_shards[:3]]

    confidence = sorted_shards[0].severity if sorted_shards else 0.5
    confidence = round(min(confidence, 1.0), 2)

    root_cause = _derive_root_cause(issue_type, signal, svc)

    template = _FIX_TEMPLATES.get(issue_type, _FIX_TEMPLATES["Unknown"])
    suggested_fix = template.format(svc=svc, ns=ns)[:200]

    return IncidentReport(
        issue_type=issue_type,  # type: ignore[arg-type]
        root_cause=root_cause[:300],
        confidence=confidence,
        evidence=evidence,
        suggested_fix=suggested_fix,
    )

def _derive_root_cause(issue_type: str, signal: str, service: str) -> str:
    causes = {
        "OOMKilled": f"{service} exceeded its memory limit and was killed by the kernel (OOMKiller).",
        "CrashLoopBackOff": f"{service} is crashing on startup repeatedly; likely bad config or unmet dependency.",
        "ImagePullBackOff": f"Container image for {service} cannot be pulled; tag missing or registry auth failed.",
        "DependencyTimeout": f"{service} cannot reach a required dependency (connection refused/timeout).",
        "DeploymentRegression": f"A recent deployment of {service} broke availability; pods not reaching ready state.",
        "Unknown": f"Failure pattern for {service} unclassified; manual inspection required.",
    }
    base = causes.get(issue_type, causes["Unknown"])
    entities = re.findall(r'[\w.-]+:\d{2,5}', signal)
    if entities:
        base += f" Involved: {', '.join(entities[:2])}."
    return base
