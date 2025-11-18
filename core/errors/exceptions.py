"""Custom exceptions"""

class AgentSystemError(Exception):
    """Base exception"""
    pass

class AgentNotFoundError(AgentSystemError):
    """Agent not found"""
    pass

class InvalidStateError(AgentSystemError):
    """Invalid state"""
    pass

class ExecutionTimeoutError(AgentSystemError):
    """Execution timeout"""
    pass

class ValidationError(AgentSystemError):
    """Validation failed"""
    pass