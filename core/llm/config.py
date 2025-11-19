"""
LLM Configuration Module
"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

from core.config.setting import settings

class LLMConfig(BaseModel):
    """
    Configuration for Large Language Model interactions.
    
    Inherits defaults from the global settings and can be overridden
    by agent-specific configurations.
    """
    model: str = Field(settings.LLM_MODEL, description="The model name to use for the LLM.")
    temperature: float = Field(settings.LLM_TEMPERATURE, ge=0.0, le=2.0, description="LLM temperature.")
    max_tokens: int = Field(settings.LLM_MAX_TOKENS, ge=1, description="LLM max tokens.")
    base_url: Optional[str] = Field(str(settings.LLM_API_BASE_URL) if settings.LLM_API_BASE_URL else None, description="The base URL for the LLM API.")
    timeout: int = Field(settings.LLM_TIMEOUT, ge=1, description="Request timeout in seconds.")

    class Config:
        # This allows the model to be created from a dictionary that has extra keys
        extra = 'ignore'
        # Pydantic v2 needs this to allow HttpUrl to be used
        arbitrary_types_allowed = True