from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    model_id: str = "claude-sonnet-4-6"
    mcp_k8s_url: str = "http://mcp-k8s-server:8001"
    mcp_logs_url: str = "http://mcp-logs-server:8002"
    mcp_metrics_url: str = "http://mcp-metrics-server:8003"
    in_cluster: bool = True
    failure_gravity_threshold: float = 0.65
    max_tool_calls: int = 3

    class Config:
        env_file = ".env"

settings = Settings()
