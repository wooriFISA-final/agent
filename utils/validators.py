"""Validation utilities"""
from typing import Any, Dict

def validate_state(state: Dict[str, Any], required_keys: list) -> bool:
    """Validate state has required keys"""
    return all(key in state for key in required_keys)