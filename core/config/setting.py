"""
System Configuration Module using Pydantic-Settings.

This module provides a centralized, type-safe configuration management
system for the application. It loads settings from environment variables
and .env files.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Load Environment Variables
LLM_BASE_URL = os.getenv("LLM_API_BASE_URL", None)


class AgentSystemConfig(BaseSettings):
    """
    Defines the system's configuration schema.

    Pydantic-Settings automatically maps environment variables to these fields.
    For example, the environment variable `AGENT_DEBUG` will map to the `debug` field.
    """
    model_config = SettingsConfigDict(
        env_prefix='AGENT_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # Environment
    ENVIRONMENT: str = Field("development", description="Execution environment")
    DEBUG: bool = Field(True, description="Debug mode")

    # FastAPI Server
    API_HOST: str = Field("0.0.0.0", description="Host for the API server")
    API_PORT: int = Field(8080, description="Port for the API server")
    API_VERSION: str = Field("2.1.0", description="API Version")

    # Logging
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FILE: Optional[str] = Field("logs/agent_system.log", description="Log file path")

    # Graph Settings
    GRAPH_YAML_PATH: Path = Field("graph/schemas/graph.yaml", description="Path to the graph YAML file")
    MAX_GRAPH_ITERATIONS: int = Field(10, description="Default max iterations for the graph")

    # MCP (Mission Control Protocol)
    MCP_URL: HttpUrl = Field("http://localhost:8888/mcp/", description="URL for the MCP server")
    MCP_CONNECTION_RETRIES: int = Field(5, description="Number of retries to connect to MCP")
    MCP_CONNECTION_TIMEOUT: int = Field(2, description="Seconds to wait between MCP connection retries")

    # LLM Provider
    LLM_PROVIDER: str = Field("ollama", description="The provider for the LLM (e.g., 'ollama', 'openai')")
    LLM_MODEL: str = Field("qwen3:8b", description="The default model name for the LLM.")
    LLM_API_BASE_URL: Optional[HttpUrl] = Field(LLM_BASE_URL, description="The base URL for the LLM API, if applicable.")
    LLM_TEMPERATURE: float = Field(0.7, ge=0.0, le=2.0, description="Default LLM temperature.")
    LLM_MAX_TOKENS: int = Field(4096, ge=1, description="Default LLM max tokens.")
    LLM_TIMEOUT: int = Field(180, ge=1, description="Default request timeout in seconds for LLM calls.")

    # Agent Registry
    AGENTS_MODULE_PATH: str = Field("agent.implementations", description="Python path to discover agents")


# Create a single, globally accessible settings instance.
# This instance is imported by other modules to access configuration values.
settings = AgentSystemConfig()

