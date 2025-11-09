"""Helper utilities"""
import uuid
from datetime import datetime

def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"session_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
