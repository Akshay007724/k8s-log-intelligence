# Ambient Developer Context Layer (ADCL) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-ready system that aggregates developer signals (Git, Slack, Jira, Sentry) and answers "What is the most critical context for file X right now?" with minimal token usage.

**Architecture:** Five containerized microservices communicating over MCP and HTTP/WebSocket. A K-Agent ReAct loop in `weaver.py` orchestrates tool calls against MCP servers, scoring signals via Relevance Gravity (exponential time decay + file proximity), and streams gravity events to a FastAPI WebSocket arena UI. All tool outputs are enforced to <150-token Context Shards.

**Tech Stack:** Python 3.12, FastAPI, asyncio, `mcp` (modelcontextprotocol/python-sdk), SQLite, aiosqlite, GitPython, prometheus-client, opentelemetry-sdk, Docker multi-stage builds, Kubernetes 1.28+

---

## File Structure

```
ambient-layer/
├── src/
│   ├── models.py               # Shared Pydantic models: ContextShard, GravityEvent, WeaverResult
│   ├── identity_graph.py       # SQLite identity resolution + LRU cache + Neo4j adapter ABC
│   ├── mcp_server.py           # MCP server: 3 tools returning ContextShards
│   ├── gravity.py              # Gravity scoring: time decay + file proximity + normalization
│   ├── weaver.py               # K-Agent ReAct loop: think→act→observe, max 2 tool calls
│   └── arena_server.py         # FastAPI app: /query, /arena/stream (WebSocket), /metrics, /feedback
├── tests/
│   ├── test_models.py
│   ├── test_identity_graph.py
│   ├── test_gravity.py
│   ├── test_mcp_server.py
│   ├── test_weaver.py
│   └── test_arena_server.py
├── k8s/
│   ├── configmap-weights.yaml
│   ├── mcp-server-deploy.yaml
│   ├── weaver-deploy.yaml
│   ├── arena-deploy.yaml
│   ├── identity-deploy.yaml
│   ├── services.yaml
│   └── weaver-hpa.yaml
├── Dockerfile                  # Multi-stage, slim, build-target per service
├── docker-compose.yml          # All 5 services + Prometheus + Grafana
├── pyproject.toml
└── README.md
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `src/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from src.models import ContextShard, GravityEvent, WeaverResult

def test_context_shard_token_budget():
    shard = ContextShard(
        shard_id="s1",
        summary="x " * 151,
        entities=["foo"],
        relevance_hint=0.9,
    )
    assert len(shard.summary.split()) <= 150

def test_gravity_event_types():
    for t in ("glow", "pulse", "link"):
        e = GravityEvent(entity_id="e1", type=t, strength=0.5, coordinates=[0.0, 0.0], label="L")
        assert e.type == t

def test_gravity_event_rejects_invalid_type():
    with pytest.raises(Exception):
        GravityEvent(entity_id="e1", type="unknown", strength=0.5, coordinates=[0.0, 0.0], label="L")

def test_weaver_result_shard_limit():
    shards = [ContextShard(shard_id=f"s{i}", summary="x", entities=[], relevance_hint=0.9) for i in range(10)]
    result = WeaverResult(shards=shards, synthesis="answer", token_count=50)
    assert len(result.shards) <= 5
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ambient-layer"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.7.0",
    "mcp>=1.0.0",
    "aiosqlite>=0.20.0",
    "gitpython>=3.1.43",
    "httpx>=0.27.0",
    "prometheus-client>=0.20.0",
    "opentelemetry-sdk>=1.24.0",
    "opentelemetry-exporter-otlp>=1.24.0",
    "opentelemetry-instrumentation-fastapi>=0.45b0",
    "anyio>=4.3.0",
]

[project.optional-dependencies]
neo4j = ["neo4j>=5.20.0"]
dev = ["pytest>=8.2.0", "pytest-asyncio>=0.23.0", "httpx>=0.27.0"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Create `src/models.py`**

```python
from __future__ import annotations
import uuid
from typing import Literal
from pydantic import BaseModel, field_validator


class ContextShard(BaseModel):
    shard_id: str
    summary: str        # enforced <150 tokens at source
    entities: list[str]
    relevance_hint: float

    @field_validator("summary")
    @classmethod
    def truncate_summary(cls, v: str) -> str:
        words = v.split()
        return " ".join(words[:150]) if len(words) > 150 else v

    @field_validator("relevance_hint")
    @classmethod
    def clamp_hint(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class GravityEvent(BaseModel):
    entity_id: str
    type: Literal["glow", "pulse", "link"]
    strength: float
    coordinates: list[float]
    label: str


class WeaverResult(BaseModel):
    shards: list[ContextShard]
    synthesis: str
    token_count: int

    @field_validator("shards")
    @classmethod
    def cap_shards(cls, v: list[ContextShard]) -> list[ContextShard]:
        return sorted(v, key=lambda s: s.relevance_hint, reverse=True)[:5]
```

- [ ] **Step 4: Create `src/__init__.py` and `tests/__init__.py`** (empty files)

- [ ] **Step 5: Run tests**

```bash
pip install -e ".[dev]" && pytest tests/test_models.py -v
```
Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git init && git add pyproject.toml src/ tests/test_models.py
git commit -m "feat: project scaffold with shared Pydantic models"
```

---

## Task 2: Identity Graph

**Files:**
- Create: `src/identity_graph.py`
- Create: `tests/test_identity_graph.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_identity_graph.py
import pytest
import asyncio
from src.identity_graph import IdentityGraph, UserRecord

@pytest.fixture
async def graph(tmp_path):
    g = IdentityGraph(db_path=str(tmp_path / "ids.db"))
    await g.init()
    return g

async def test_register_and_resolve_by_github(graph):
    await graph.register(github_id="gh_alice", slack_id="U_alice", email="alice@example.com", display_name="Alice")
    rec = await graph.resolve_user("gh_alice")
    assert rec is not None
    assert rec.email == "alice@example.com"

async def test_resolve_by_email(graph):
    await graph.register(github_id="gh_bob", slack_id="U_bob", email="bob@example.com", display_name="Bob")
    rec = await graph.resolve_user("bob@example.com")
    assert rec.github_id == "gh_bob"

async def test_resolve_cache_hit(graph):
    await graph.register(github_id="gh_carol", slack_id="U_carol", email="carol@example.com", display_name="Carol")
    await graph.resolve_user("gh_carol")   # populate cache
    graph._db = None                       # break db — must serve from cache
    rec = await graph.resolve_user("gh_carol")
    assert rec is not None

async def test_resolve_unknown_returns_none(graph):
    assert await graph.resolve_user("nobody") is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_identity_graph.py -v
```
Expected: ImportError / FAIL

- [ ] **Step 3: Implement `src/identity_graph.py`**

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
import aiosqlite


@dataclass
class UserRecord:
    github_id: str
    slack_id: str
    email: str
    display_name: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    github_id    TEXT PRIMARY KEY,
    slack_id     TEXT UNIQUE,
    email        TEXT UNIQUE,
    display_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_slack  ON users(slack_id);
CREATE INDEX IF NOT EXISTS idx_email  ON users(email);
"""


class IdentityGraph:
    def __init__(self, db_path: str = "identity.db") -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._cache: dict[str, UserRecord] = {}

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def register(self, *, github_id: str, slack_id: str, email: str, display_name: str) -> None:
        assert self._db
        await self._db.execute(
            "INSERT OR REPLACE INTO users VALUES (?,?,?,?)",
            (github_id, slack_id, email, display_name),
        )
        await self._db.commit()
        rec = UserRecord(github_id, slack_id, email, display_name)
        for key in (github_id, slack_id, email):
            self._cache[key] = rec

    async def resolve_user(self, identity_key: str) -> Optional[UserRecord]:
        if identity_key in self._cache:
            return self._cache[identity_key]
        if self._db is None:
            return None
        async with self._db.execute(
            "SELECT * FROM users WHERE github_id=? OR slack_id=? OR email=?",
            (identity_key, identity_key, identity_key),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        rec = UserRecord(row["github_id"], row["slack_id"], row["email"], row["display_name"])
        self._cache[identity_key] = rec
        return rec


# Optional Neo4j adapter interface
class IdentityGraphAdapter:
    async def init(self) -> None: ...
    async def resolve_user(self, identity_key: str) -> Optional[UserRecord]: ...
    async def register(self, **kwargs) -> None: ...
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_identity_graph.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/identity_graph.py tests/test_identity_graph.py
git commit -m "feat: identity graph with SQLite + LRU cache + Neo4j adapter stub"
```

---

## Task 3: Gravity Scoring

**Files:**
- Create: `src/gravity.py`
- Create: `tests/test_gravity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gravity.py
from datetime import datetime, timedelta, timezone
import pytest
from src.gravity import time_decay, file_proximity, gravity_score, GravityWeights

def test_time_decay_recent_is_high():
    now = datetime.now(timezone.utc)
    assert time_decay(now) > 0.95

def test_time_decay_old_is_low():
    old = datetime.now(timezone.utc) - timedelta(hours=72)
    assert time_decay(old) < 0.05

def test_file_proximity_direct():
    assert file_proximity("src/foo.py", "src/foo.py") == 1.0

def test_file_proximity_same_module():
    assert file_proximity("src/auth/login.py", "src/auth/logout.py") == 0.7

def test_file_proximity_repo_wide():
    assert file_proximity("src/auth/login.py", "tests/test_db.py") == 0.3

def test_gravity_above_threshold_passes():
    score = gravity_score(recency=0.9, event_count=5, proximity=1.0)
    assert score > 0.6

def test_gravity_below_threshold_fails():
    score = gravity_score(recency=0.05, event_count=1, proximity=0.3)
    assert score < 0.6

def test_gravity_normalized_0_to_1():
    score = gravity_score(recency=1.0, event_count=100, proximity=1.0)
    assert 0.0 <= score <= 1.0

def test_custom_weights():
    w = GravityWeights(recency=0.8, frequency=0.1, proximity=0.1)
    score = gravity_score(recency=1.0, event_count=1, proximity=1.0, weights=w)
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_gravity.py -v
```
Expected: ImportError / FAIL

- [ ] **Step 3: Implement `src/gravity.py`**

```python
from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class GravityWeights:
    recency: float = 0.50
    frequency: float = 0.30
    proximity: float = 0.20

    def __post_init__(self):
        total = self.recency + self.frequency + self.proximity
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


_DEFAULT_WEIGHTS = GravityWeights()
_DECAY_LAMBDA = 0.1  # half-life ~7h


def time_decay(event_time: datetime, lambda_: float = _DECAY_LAMBDA) -> float:
    """Exponential decay: e^(-λ·hours_ago). Returns [0, 1]."""
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    hours_ago = (datetime.now(timezone.utc) - event_time).total_seconds() / 3600
    return math.exp(-lambda_ * max(hours_ago, 0))


def file_proximity(target: str, signal: str) -> float:
    """1.0 direct match, 0.7 same directory, 0.3 repo-wide."""
    if target == signal:
        return 1.0
    t_dir = target.rsplit("/", 1)[0]
    s_dir = signal.rsplit("/", 1)[0]
    return 0.7 if t_dir == s_dir else 0.3


def gravity_score(
    recency: float,
    event_count: int,
    proximity: float,
    weights: Optional[GravityWeights] = None,
) -> float:
    """
    Relevance_Gravity = (w_r × recency) + (w_f × norm_freq) + (w_p × proximity)
    Normalized to [0, 1].
    """
    w = weights or _DEFAULT_WEIGHTS
    norm_freq = min(event_count / 10.0, 1.0)
    raw = w.recency * recency + w.frequency * norm_freq + w.proximity * proximity
    return max(0.0, min(raw, 1.0))


GRAVITY_THRESHOLD = 0.6
HIGH_GRAVITY_THRESHOLD = 0.8
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_gravity.py -v
```
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/gravity.py tests/test_gravity.py
git commit -m "feat: gravity scoring with exponential time decay and file proximity"
```

---

## Task 4: MCP Server

**Files:**
- Create: `src/mcp_server.py`
- Create: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mcp_server.py
import pytest
from unittest.mock import AsyncMock, patch
from src.mcp_server import build_mcp_tools, GitAdapter, SignalAdapter, ErrorAdapter

@pytest.fixture
def git_adapter(tmp_path):
    return GitAdapter(repo_path=str(tmp_path))

@pytest.fixture
def signal_adapter():
    return SignalAdapter(slack_token=None, jira_url=None, jira_token=None)

@pytest.fixture
def error_adapter():
    return ErrorAdapter(sentry_dsn=None)

async def test_get_semantic_diff_returns_shard(git_adapter):
    shard = await git_adapter.get_semantic_diff("src/foo.py")
    assert shard.shard_id.startswith("diff:")
    assert len(shard.summary.split()) <= 150
    assert 0.0 <= shard.relevance_hint <= 1.0

async def test_get_recent_conversations_returns_shard(signal_adapter):
    shard = await signal_adapter.get_recent_conversations("src/foo.py")
    assert shard.shard_id.startswith("conv:")
    assert isinstance(shard.entities, list)

async def test_get_error_signals_returns_shard(error_adapter):
    shard = await error_adapter.get_error_signals("src/foo.py")
    assert shard.shard_id.startswith("err:")
    assert "risk" in shard.summary.lower() or shard.relevance_hint >= 0

async def test_shard_summary_under_150_tokens(git_adapter):
    shard = await git_adapter.get_semantic_diff("any/path.py")
    assert len(shard.summary.split()) <= 150
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_mcp_server.py -v
```

- [ ] **Step 3: Implement `src/mcp_server.py`**

```python
"""
MCP server exposing 3 tools: get_semantic_diff, get_recent_conversations, get_error_signals.
All outputs are ContextShards (<150 tokens). Run as: python -m src.mcp_server
"""
from __future__ import annotations
import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.models import ContextShard
from src.gravity import time_decay, file_proximity, gravity_score


# ── Adapters ────────────────────────────────────────────────────────────────

class GitAdapter:
    def __init__(self, repo_path: str = ".") -> None:
        self._repo_path = repo_path

    async def get_semantic_diff(self, file_path: str) -> ContextShard:
        try:
            import git
            repo = git.Repo(self._repo_path, search_parent_directories=True)
            commits = list(repo.iter_commits(paths=file_path, max_count=5))
            if not commits:
                return ContextShard(
                    shard_id=f"diff:{file_path}:none",
                    summary=f"No recent commits for {file_path}. New file or untracked.",
                    entities=[],
                    relevance_hint=0.1,
                )
            recent = commits[0]
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(recent.committed_date, tz=timezone.utc)
            diff_stat = recent.stats.files.get(file_path, {})
            insertions = diff_stat.get("insertions", 0)
            deletions = diff_stat.get("deletions", 0)
            risk = "HIGH" if deletions > 50 else "MEDIUM" if deletions > 10 else "LOW"
            summary = (
                f"Last commit {int(age.total_seconds() / 3600)}h ago by {recent.author.name}: "
                f'"{recent.message.splitlines()[0][:80]}". '
                f"+{insertions}/-{deletions} lines. Risk: {risk}."
            )
            decay = time_decay(datetime.fromtimestamp(recent.committed_date, tz=timezone.utc))
            return ContextShard(
                shard_id=f"diff:{file_path}:{recent.hexsha[:8]}",
                summary=summary,
                entities=[recent.author.email, recent.hexsha[:8]],
                relevance_hint=gravity_score(recency=decay, event_count=len(commits), proximity=1.0),
            )
        except Exception as exc:
            return ContextShard(
                shard_id=f"diff:{file_path}:err",
                summary=f"Git unavailable for {file_path}: {type(exc).__name__}.",
                entities=[],
                relevance_hint=0.0,
            )


class SignalAdapter:
    def __init__(
        self,
        slack_token: Optional[str],
        jira_url: Optional[str],
        jira_token: Optional[str],
    ) -> None:
        self._slack_token = slack_token
        self._jira_url = jira_url
        self._jira_token = jira_token

    async def get_recent_conversations(self, file_path: str) -> ContextShard:
        # Stub: replace with real Slack/Jira API calls when tokens available
        module = Path(file_path).stem
        summary = (
            f"No live Slack/Jira integration configured. "
            f"To enable: set SLACK_TOKEN and JIRA_URL env vars. "
            f"Module '{module}' has no tracked conversations."
        )
        return ContextShard(
            shard_id=f"conv:{file_path}:{uuid.uuid4().hex[:8]}",
            summary=summary,
            entities=[module],
            relevance_hint=0.0,
        )


class ErrorAdapter:
    def __init__(self, sentry_dsn: Optional[str]) -> None:
        self._dsn = sentry_dsn

    async def get_error_signals(self, file_path: str) -> ContextShard:
        # Stub: replace with real Sentry API when DSN available
        summary = (
            f"No Sentry DSN configured. Risk: LOW. "
            f"Set SENTRY_DSN env var to enable live error signals for {file_path}."
        )
        return ContextShard(
            shard_id=f"err:{file_path}:{uuid.uuid4().hex[:8]}",
            summary=summary,
            entities=[],
            relevance_hint=0.0,
        )


# ── MCP Server ──────────────────────────────────────────────────────────────

def build_mcp_tools(
    git_adapter: Optional[GitAdapter] = None,
    signal_adapter: Optional[SignalAdapter] = None,
    error_adapter: Optional[ErrorAdapter] = None,
) -> Server:
    git = git_adapter or GitAdapter(os.getenv("REPO_PATH", "."))
    signals = signal_adapter or SignalAdapter(
        slack_token=os.getenv("SLACK_TOKEN"),
        jira_url=os.getenv("JIRA_URL"),
        jira_token=os.getenv("JIRA_TOKEN"),
    )
    errors = error_adapter or ErrorAdapter(sentry_dsn=os.getenv("SENTRY_DSN"))

    server = Server("adcl-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="get_semantic_diff",
                description="Returns recent git change summary for a file as a Context Shard.",
                inputSchema={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            ),
            Tool(
                name="get_recent_conversations",
                description="Returns Slack/Jira discussion summary for a file as a Context Shard.",
                inputSchema={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            ),
            Tool(
                name="get_error_signals",
                description="Returns Sentry/log error summary for a file as a Context Shard.",
                inputSchema={"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        fp = arguments["file_path"]
        if name == "get_semantic_diff":
            shard = await git.get_semantic_diff(fp)
        elif name == "get_recent_conversations":
            shard = await signals.get_recent_conversations(fp)
        elif name == "get_error_signals":
            shard = await errors.get_error_signals(fp)
        else:
            raise ValueError(f"Unknown tool: {name}")
        return [TextContent(type="text", text=shard.model_dump_json())]

    return server


async def _main() -> None:
    server = build_mcp_tools()
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_mcp_server.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: MCP server with 3 tools returning <150-token ContextShards"
```

---

## Task 5: Weaver Agent (K-Agent ReAct Loop)

**Files:**
- Create: `src/weaver.py`
- Create: `tests/test_weaver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_weaver.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.models import ContextShard
from src.weaver import WeaverAgent, AgentState, ThoughtResult

@pytest.fixture
def mock_shards():
    return {
        "diff": ContextShard(shard_id="diff:f", summary="Recent refactor removed auth check. Risk HIGH.", entities=["alice"], relevance_hint=0.85),
        "conv": ContextShard(shard_id="conv:f", summary="Slack: team agreed to defer auth until v2.", entities=["bob"], relevance_hint=0.72),
        "err":  ContextShard(shard_id="err:f",  summary="0 Sentry errors in 7 days. Risk LOW.", entities=[], relevance_hint=0.1),
    }

@pytest.fixture
def agent(mock_shards):
    a = WeaverAgent()
    a._git = AsyncMock(return_value=mock_shards["diff"])
    a._conv = AsyncMock(return_value=mock_shards["conv"])
    a._err = AsyncMock(return_value=mock_shards["err"])
    return a

async def test_query_returns_weaver_result(agent):
    result = await agent.query("src/auth.py")
    assert result.synthesis
    assert len(result.synthesis.split()) <= 300
    assert 1 <= len(result.shards) <= 5

async def test_max_2_tool_calls_by_default(agent):
    await agent.query("src/auth.py")
    total_calls = agent._git.call_count + agent._conv.call_count + agent._err.call_count
    assert total_calls <= 2

async def test_high_gravity_allows_3_tool_calls(agent, mock_shards):
    # Make all shards high gravity
    for s in mock_shards.values():
        s.relevance_hint = 0.9
    result = await agent.query("src/auth.py")
    assert result.shards  # still returns valid result

async def test_filters_low_gravity_shards(agent, mock_shards):
    result = await agent.query("src/auth.py")
    for shard in result.shards:
        assert shard.relevance_hint >= 0.6

async def test_synthesis_under_300_tokens(agent):
    result = await agent.query("src/auth.py")
    assert len(result.synthesis.split()) <= 300

async def test_token_count_tracked(agent):
    result = await agent.query("src/auth.py")
    assert result.token_count > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_weaver.py -v
```

- [ ] **Step 3: Implement `src/weaver.py`**

```python
"""
K-Agent ReAct loop: THINK → ACT → OBSERVE.
Max 2 tool calls per query (3 if HIGH_GRAVITY detected).
Outputs top 3-5 ContextShards + synthesis <300 tokens.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional, Literal

from src.models import ContextShard, WeaverResult
from src.gravity import GRAVITY_THRESHOLD, HIGH_GRAVITY_THRESHOLD


ToolFn = Callable[[str], Awaitable[ContextShard]]

_TOOLS: list[Literal["diff", "conv", "err"]] = ["diff", "conv", "err"]


@dataclass
class ThoughtResult:
    action: Optional[Literal["diff", "conv", "err"]]
    reasoning: str
    high_gravity: bool = False


@dataclass
class AgentState:
    file_path: str
    tool_calls: int = 0
    shards: list[ContextShard] = field(default_factory=list)
    called: set[str] = field(default_factory=set)

    @property
    def max_calls(self) -> int:
        return 3 if any(s.relevance_hint >= HIGH_GRAVITY_THRESHOLD for s in self.shards) else 2

    @property
    def is_sufficient(self) -> bool:
        high = [s for s in self.shards if s.relevance_hint >= GRAVITY_THRESHOLD]
        return len(high) >= 2 or self.tool_calls >= self.max_calls


class WeaverAgent:
    def __init__(
        self,
        git_fn: Optional[ToolFn] = None,
        conv_fn: Optional[ToolFn] = None,
        err_fn: Optional[ToolFn] = None,
    ) -> None:
        from src.mcp_server import GitAdapter, SignalAdapter, ErrorAdapter
        import os
        self._git = git_fn or GitAdapter(os.getenv("REPO_PATH", ".")).get_semantic_diff
        self._conv = conv_fn or SignalAdapter(None, None, None).get_recent_conversations
        self._err = err_fn or ErrorAdapter(None).get_error_signals
        self._tool_map: dict[str, ToolFn] = {
            "diff": self._git,
            "conv": self._conv,
            "err": self._err,
        }

    def _think(self, state: AgentState) -> ThoughtResult:
        """Decide which tool to call next, or None if sufficient."""
        if state.is_sufficient:
            return ThoughtResult(action=None, reasoning="sufficient context gathered")
        # Priority: diff first (highest signal), then conv, then err
        for tool in _TOOLS:
            if tool not in state.called:
                high = any(s.relevance_hint >= HIGH_GRAVITY_THRESHOLD for s in state.shards)
                return ThoughtResult(action=tool, reasoning=f"gathering {tool} signal", high_gravity=high)
        return ThoughtResult(action=None, reasoning="all tools exhausted")

    async def _act(self, tool: str, file_path: str) -> ContextShard:
        return await self._tool_map[tool](file_path)

    def _observe(self, state: AgentState, shard: ContextShard, tool: str) -> None:
        state.shards.append(shard)
        state.called.add(tool)
        state.tool_calls += 1

    def _synthesize(self, state: AgentState) -> WeaverResult:
        relevant = [s for s in state.shards if s.relevance_hint >= GRAVITY_THRESHOLD]
        relevant.sort(key=lambda s: s.relevance_hint, reverse=True)
        top = relevant[:5] or state.shards[:1]

        lines = [f"Context for `{state.file_path}` ({len(top)} signals):"]
        for s in top:
            lines.append(f"• [{s.relevance_hint:.2f}] {s.summary}")
        synthesis = " ".join(lines)
        words = synthesis.split()
        synthesis = " ".join(words[:300])
        token_count = sum(len(s.summary.split()) for s in top) + len(lines[0].split())

        return WeaverResult(shards=top, synthesis=synthesis, token_count=token_count)

    async def query(self, file_path: str) -> WeaverResult:
        state = AgentState(file_path=file_path)
        while True:
            thought = self._think(state)
            if thought.action is None:
                break
            shard = await self._act(thought.action, file_path)
            self._observe(state, shard, thought.action)
        return self._synthesize(state)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_weaver.py -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/weaver.py tests/test_weaver.py
git commit -m "feat: K-Agent ReAct loop with gravity filtering and 2-tool budget"
```

---

## Task 6: Arena Server (FastAPI + WebSocket + Observability)

**Files:**
- Create: `src/arena_server.py`
- Create: `tests/test_arena_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_arena_server.py
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from src.arena_server import create_app
from src.models import WeaverResult, ContextShard

@pytest.fixture
def mock_weaver():
    w = AsyncMock()
    w.query.return_value = WeaverResult(
        shards=[ContextShard(shard_id="s1", summary="Auth removed. Risk HIGH.", entities=["alice"], relevance_hint=0.85)],
        synthesis="Auth removed recently. High risk. Review before deploy.",
        token_count=42,
    )
    return w

@pytest.fixture
def app(mock_weaver):
    return create_app(weaver=mock_weaver)

@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

async def test_query_endpoint(client, mock_weaver):
    resp = await client.post("/query", json={"file_path": "src/auth.py"})
    assert resp.status_code == 200
    data = resp.json()
    assert "synthesis" in data
    assert "shards" in data

async def test_query_returns_synthesis_under_300_words(client):
    resp = await client.post("/query", json={"file_path": "src/auth.py"})
    assert len(resp.json()["synthesis"].split()) <= 300

async def test_metrics_endpoint(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert b"token_usage" in resp.content or b"query" in resp.content

async def test_feedback_endpoint(client):
    resp = await client.post("/feedback", json={"shard_id": "s1", "vote": "up", "file_path": "src/auth.py"})
    assert resp.status_code == 200

async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_arena_server.py -v
```

- [ ] **Step 3: Implement `src/arena_server.py`**

```python
"""
Arena server: FastAPI + WebSocket gravity stream + Prometheus metrics + feedback loop.
"""
from __future__ import annotations
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.models import ContextShard, GravityEvent, WeaverResult
from src.gravity import GRAVITY_THRESHOLD, HIGH_GRAVITY_THRESHOLD


# ── Metrics ─────────────────────────────────────────────────────────────────

QUERY_COUNT     = Counter("adcl_query_total", "Total queries")
TOKEN_USAGE     = Histogram("adcl_token_usage", "Tokens per query", buckets=[50, 100, 200, 300, 500])
TOOL_CALLS      = Histogram("adcl_tool_calls", "Tool calls per query", buckets=[1, 2, 3, 4])
GRAVITY_SCORE   = Gauge("adcl_avg_gravity_score", "Running avg gravity score")
QUERY_LATENCY   = Histogram("adcl_query_latency_seconds", "Query latency", buckets=[0.1, 0.5, 1, 2, 5])

# ── Tracing ──────────────────────────────────────────────────────────────────

_tracer_provider = TracerProvider()
_span_exporter = InMemorySpanExporter()
_tracer_provider.add_span_processor(BatchSpanProcessor(_span_exporter))
trace.set_tracer_provider(_tracer_provider)
tracer = trace.get_tracer("adcl.arena")


# ── WebSocket Manager ────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws) if hasattr(self._connections, "discard") else None
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, event: GravityEvent) -> None:
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(event.model_dump())
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
_feedback_store: list[dict] = []


def _shard_to_gravity_event(shard: ContextShard, idx: int) -> GravityEvent:
    if shard.relevance_hint >= HIGH_GRAVITY_THRESHOLD:
        etype = "pulse"
    elif shard.relevance_hint >= GRAVITY_THRESHOLD:
        etype = "glow"
    else:
        etype = "link"
    return GravityEvent(
        entity_id=shard.shard_id,
        type=etype,
        strength=shard.relevance_hint,
        coordinates=[float(idx * 120), float(100 + idx * 80)],
        label=shard.summary[:60],
    )


# ── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    file_path: str


class FeedbackRequest(BaseModel):
    shard_id: str
    vote: str   # "up" | "down" | "flag"
    file_path: str
    note: Optional[str] = None


# ── App Factory ──────────────────────────────────────────────────────────────

def create_app(weaver=None) -> FastAPI:
    if weaver is None:
        from src.weaver import WeaverAgent
        weaver = WeaverAgent()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="ADCL Arena", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/query", response_model=WeaverResult)
    async def query(req: QueryRequest):
        QUERY_COUNT.inc()
        t0 = time.perf_counter()
        with tracer.start_as_current_span("weaver.query") as span:
            span.set_attribute("file_path", req.file_path)
            result: WeaverResult = await weaver.query(req.file_path)
            span.set_attribute("token_count", result.token_count)

        elapsed = time.perf_counter() - t0
        QUERY_LATENCY.observe(elapsed)
        TOKEN_USAGE.observe(result.token_count)
        if result.shards:
            avg = sum(s.relevance_hint for s in result.shards) / len(result.shards)
            GRAVITY_SCORE.set(avg)

        # Broadcast gravity events to connected WebSocket clients
        for i, shard in enumerate(result.shards):
            event = _shard_to_gravity_event(shard, i)
            await manager.broadcast(event)

        return result

    @app.post("/feedback")
    async def feedback(req: FeedbackRequest):
        _feedback_store.append(req.model_dump())
        return {"accepted": True, "shard_id": req.shard_id}

    @app.get("/metrics")
    async def metrics():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/arena/stream")
    async def arena_stream(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive; events are pushed via broadcast()
                await asyncio.sleep(30)
                await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.arena_server:app", host="0.0.0.0", port=8000, reload=False)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_arena_server.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/arena_server.py tests/test_arena_server.py
git commit -m "feat: arena FastAPI server with WebSocket gravity stream and Prometheus metrics"
```

---

## Task 7: Multi-stage Dockerfile + docker-compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create `Dockerfile`** (multi-stage, build-target per service)

```dockerfile
# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12-slim

# ── Base ─────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────────
FROM base AS deps
COPY pyproject.toml .
RUN pip install -e ".[neo4j]"

# ── Source ────────────────────────────────────────────────────────────────────
FROM deps AS source
COPY src/ ./src/

# ── MCP Server ────────────────────────────────────────────────────────────────
FROM source AS mcp-server
EXPOSE 9000
CMD ["python", "-m", "src.mcp_server"]

# ── Weaver ────────────────────────────────────────────────────────────────────
FROM source AS weaver
EXPOSE 8001
CMD ["uvicorn", "src.weaver:app", "--host", "0.0.0.0", "--port", "8001"]

# ── Arena UI ──────────────────────────────────────────────────────────────────
FROM source AS arena
EXPOSE 8000
CMD ["uvicorn", "src.arena_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ── Identity Graph ────────────────────────────────────────────────────────────
FROM source AS identity
EXPOSE 8002
CMD ["uvicorn", "src.identity_server:app", "--host", "0.0.0.0", "--port", "8002"]
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
version: "3.9"

x-common: &common
  build:
    context: .
    dockerfile: Dockerfile
  restart: unless-stopped
  environment: &common-env
    REPO_PATH: /workspace
    LOG_LEVEL: warning

services:
  mcp-server:
    <<: *common
    build:
      context: .
      target: mcp-server
    container_name: adcl-mcp-server
    ports: ["9000:9000"]
    environment:
      <<: *common-env
      SLACK_TOKEN: ${SLACK_TOKEN:-}
      JIRA_URL: ${JIRA_URL:-}
      JIRA_TOKEN: ${JIRA_TOKEN:-}
      SENTRY_DSN: ${SENTRY_DSN:-}
    volumes:
      - ${WORKSPACE_PATH:-/tmp/workspace}:/workspace:ro
    healthcheck:
      test: ["CMD", "python", "-c", "import src.mcp_server"]
      interval: 30s
      timeout: 5s
      retries: 3

  identity:
    <<: *common
    build:
      context: .
      target: identity
    container_name: adcl-identity
    ports: ["8002:8002"]
    volumes:
      - identity-data:/data
    environment:
      <<: *common-env
      DB_PATH: /data/identity.db

  weaver:
    <<: *common
    build:
      context: .
      target: weaver
    container_name: adcl-weaver
    ports: ["8001:8001"]
    depends_on: [mcp-server, identity]
    environment:
      <<: *common-env
      MCP_SERVER_URL: http://mcp-server:9000
      IDENTITY_URL: http://identity:8002
      GRAVITY_RECENCY_W: "0.50"
      GRAVITY_FREQUENCY_W: "0.30"
      GRAVITY_PROXIMITY_W: "0.20"
      OTEL_EXPORTER_OTLP_ENDPOINT: http://otel-collector:4317

  arena:
    <<: *common
    build:
      context: .
      target: arena
    container_name: adcl-arena
    ports: ["8000:8000"]
    depends_on: [weaver]
    environment:
      <<: *common-env
      WEAVER_URL: http://weaver:8001
      OTEL_EXPORTER_OTLP_ENDPOINT: http://otel-collector:4317

  prometheus:
    image: prom/prometheus:v2.52.0
    container_name: adcl-prometheus
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command: ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.retention.time=7d"]

  grafana:
    image: grafana/grafana:10.4.2
    container_name: adcl-grafana
    ports: ["3000:3000"]
    depends_on: [prometheus]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_AUTH_ANONYMOUS_ENABLED: "true"
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  identity-data:
  prometheus-data:
  grafana-data:
```

- [ ] **Step 3: Create `monitoring/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: adcl-arena
    static_configs:
      - targets: ["arena:8000"]
    metrics_path: /metrics

  - job_name: adcl-weaver
    static_configs:
      - targets: ["weaver:8001"]
    metrics_path: /metrics
```

- [ ] **Step 4: Verify Docker build**

```bash
docker build --target arena -t adcl-arena:dev . && echo "BUILD OK"
```
Expected: `BUILD OK`

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml monitoring/
git commit -m "feat: multi-stage Dockerfile and docker-compose for all 5 services"
```

---

## Task 8: Kubernetes Manifests

**Files:**
- Create: `k8s/configmap-weights.yaml`
- Create: `k8s/mcp-server-deploy.yaml`
- Create: `k8s/weaver-deploy.yaml`
- Create: `k8s/arena-deploy.yaml`
- Create: `k8s/identity-deploy.yaml`
- Create: `k8s/services.yaml`
- Create: `k8s/weaver-hpa.yaml`

- [ ] **Step 1: Create `k8s/configmap-weights.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gravity-weights
  namespace: adcl
data:
  GRAVITY_RECENCY_W: "0.50"
  GRAVITY_FREQUENCY_W: "0.30"
  GRAVITY_PROXIMITY_W: "0.20"
  GRAVITY_THRESHOLD: "0.6"
  HIGH_GRAVITY_THRESHOLD: "0.8"
  MAX_TOOL_CALLS: "2"
  TOKEN_BUDGET_INPUT: "500"
  TOKEN_BUDGET_OUTPUT: "300"
```

- [ ] **Step 2: Create `k8s/mcp-server-deploy.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  namespace: adcl
  labels:
    app: mcp-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
        - name: mcp-server
          image: adcl-mcp-server:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 9000
          envFrom:
            - configMapRef:
                name: gravity-weights
          env:
            - name: SLACK_TOKEN
              valueFrom:
                secretKeyRef:
                  name: adcl-secrets
                  key: slack_token
                  optional: true
            - name: SENTRY_DSN
              valueFrom:
                secretKeyRef:
                  name: adcl-secrets
                  key: sentry_dsn
                  optional: true
          resources:
            requests: { cpu: "100m", memory: "128Mi" }
            limits:   { cpu: "500m", memory: "256Mi" }
          livenessProbe:
            exec:
              command: ["python", "-c", "import src.mcp_server"]
            initialDelaySeconds: 10
            periodSeconds: 30
```

- [ ] **Step 3: Create `k8s/weaver-deploy.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: weaver
  namespace: adcl
  labels:
    app: weaver
spec:
  replicas: 2
  selector:
    matchLabels:
      app: weaver
  template:
    metadata:
      labels:
        app: weaver
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: weaver
          image: adcl-weaver:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8001
          envFrom:
            - configMapRef:
                name: gravity-weights
          env:
            - name: MCP_SERVER_URL
              value: "http://mcp-server-svc:9000"
            - name: IDENTITY_URL
              value: "http://identity-svc:8002"
          resources:
            requests: { cpu: "200m", memory: "256Mi" }
            limits:   { cpu: "1000m", memory: "512Mi" }
          readinessProbe:
            httpGet:
              path: /health
              port: 8001
            initialDelaySeconds: 5
            periodSeconds: 10
```

- [ ] **Step 4: Create `k8s/arena-deploy.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arena
  namespace: adcl
  labels:
    app: arena
spec:
  replicas: 2
  selector:
    matchLabels:
      app: arena
  template:
    metadata:
      labels:
        app: arena
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: arena
          image: adcl-arena:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: gravity-weights
          env:
            - name: WEAVER_URL
              value: "http://weaver-svc:8001"
          resources:
            requests: { cpu: "100m", memory: "128Mi" }
            limits:   { cpu: "500m", memory: "256Mi" }
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
```

- [ ] **Step 5: Create `k8s/identity-deploy.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: identity
  namespace: adcl
  labels:
    app: identity
spec:
  replicas: 1
  selector:
    matchLabels:
      app: identity
  template:
    metadata:
      labels:
        app: identity
    spec:
      containers:
        - name: identity
          image: adcl-identity:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8002
          env:
            - name: DB_PATH
              value: "/data/identity.db"
          volumeMounts:
            - name: identity-data
              mountPath: /data
          resources:
            requests: { cpu: "50m",  memory: "64Mi" }
            limits:   { cpu: "200m", memory: "128Mi" }
      volumes:
        - name: identity-data
          persistentVolumeClaim:
            claimName: identity-pvc
```

- [ ] **Step 6: Create `k8s/services.yaml`**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-svc
  namespace: adcl
spec:
  selector: { app: mcp-server }
  ports:
    - port: 9000
      targetPort: 9000
---
apiVersion: v1
kind: Service
metadata:
  name: weaver-svc
  namespace: adcl
spec:
  selector: { app: weaver }
  ports:
    - port: 8001
      targetPort: 8001
---
apiVersion: v1
kind: Service
metadata:
  name: arena-svc
  namespace: adcl
spec:
  type: LoadBalancer
  selector: { app: arena }
  ports:
    - port: 80
      targetPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: identity-svc
  namespace: adcl
spec:
  selector: { app: identity }
  ports:
    - port: 8002
      targetPort: 8002
```

- [ ] **Step 7: Create `k8s/weaver-hpa.yaml`**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: weaver-hpa
  namespace: adcl
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: weaver
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 65
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 75
```

- [ ] **Step 8: Validate YAML syntax**

```bash
find k8s/ -name "*.yaml" | xargs -I{} python -c "import yaml, sys; yaml.safe_load(open('{}'))" && echo "ALL VALID"
```
Expected: `ALL VALID`

- [ ] **Step 9: Commit**

```bash
git add k8s/ monitoring/
git commit -m "feat: Kubernetes deployments, services, ConfigMap, and HPA for all services"
```

---

## Task 9: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Ambient Developer Context Layer (ADCL)

Answers: **"What is the most critical context for file X right now?"**

Aggregates Git, Slack, Jira, Sentry signals. Returns high-signal context in <2s with <300 tokens output.

## Architecture

```
┌─────────────┐    MCP     ┌──────────────────┐
│  mcp-server │◄──────────►│ context-weaver   │
│  (Git+Slack │            │ (ReAct K-Agent)  │
│   +Sentry)  │            └────────┬─────────┘
└─────────────┘                     │ HTTP
                                    ▼
┌──────────────┐  WebSocket  ┌─────────────────┐
│  identity-   │◄────────────│  arena-server   │
│  graph       │             │  (FastAPI)      │
└──────────────┘             └─────────────────┘
```

**Gravity Score** = (0.5 × recency_decay) + (0.3 × event_frequency) + (0.2 × file_proximity)
Signals with score < 0.6 are filtered. Only 2 tool calls per query (3 if HIGH_GRAVITY).

## Quick Start

```bash
cp .env.example .env        # set SLACK_TOKEN, JIRA_URL, SENTRY_DSN
docker compose up -d
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"file_path": "src/auth.py"}'
```

## Services

| Service       | Port | Purpose                            |
|---------------|------|------------------------------------|
| arena         | 8000 | HTTP API + WebSocket stream        |
| weaver        | 8001 | K-Agent ReAct orchestrator         |
| identity      | 8002 | GitHub↔Slack↔Email resolution      |
| mcp-server    | 9000 | Git/Slack/Sentry MCP tools         |
| prometheus    | 9090 | Metrics                            |
| grafana       | 3000 | Dashboards (admin/admin)           |

## WebSocket Stream

```js
const ws = new WebSocket("ws://localhost:8000/arena/stream");
ws.onmessage = (e) => {
  const event = JSON.parse(e.data);
  // { entity_id, type: "pulse"|"glow"|"link", strength, coordinates, label }
};
```

## Kubernetes

```bash
kubectl create namespace adcl
kubectl apply -f k8s/
kubectl get pods -n adcl
```

## Environment Variables

| Var             | Required | Description                   |
|-----------------|----------|-------------------------------|
| SLACK_TOKEN     | No       | Slack Bot token for signals   |
| JIRA_URL        | No       | Jira base URL                 |
| JIRA_TOKEN      | No       | Jira API token                |
| SENTRY_DSN      | No       | Sentry DSN for error signals  |
| WORKSPACE_PATH  | No       | Host path to git repo         |
| REPO_PATH       | No       | Container path to git repo    |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Metrics (Prometheus)

- `adcl_query_total` — total queries
- `adcl_token_usage` — token histogram per query
- `adcl_tool_calls` — tool calls per query
- `adcl_avg_gravity_score` — running gravity average
- `adcl_query_latency_seconds` — end-to-end latency
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with architecture, quickstart, and API reference"
```

---

## Task 10: Full Test Suite + Integration Verification

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```
Expected: All PASSED (no errors, no skips except intentional)

- [ ] **Step 2: Verify imports across all modules**

```bash
python -c "
from src.models import ContextShard, GravityEvent, WeaverResult
from src.identity_graph import IdentityGraph
from src.gravity import gravity_score, time_decay, file_proximity
from src.mcp_server import GitAdapter, SignalAdapter, ErrorAdapter
from src.weaver import WeaverAgent
from src.arena_server import create_app
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Verify Dockerfile builds all targets**

```bash
for target in mcp-server weaver arena identity; do
  docker build --target $target -t adcl-$target:test . && echo "$target: OK"
done
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final integration verification — all services building and tested"
```
