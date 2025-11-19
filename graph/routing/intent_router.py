"""
A concrete implementation of a router based on intent.
"""
from typing import Dict, Any
from .router_base import RouterBase
from .router_registry import RouterRegistry
from core.logging.logger import setup_logger

logger = setup_logger()

@RouterRegistry.register("IntentRouter")
class IntentRouter(RouterBase):
    """
    A simple router that directs the graph based on an 'intent'
    field in the agent state.
    """
    def route(self, state: Dict[str, Any]) -> str:
        """
        Determines the next node by checking the 'intent' in the state.

        Args:
            state: The current agent state.

        Returns:
            The name of the next node to execute. Defaults to '__end__' if
            intent is not found or not a string.
        """
        intent = state.get("intent")
        logger.info(f"IntentRouter routing based on intent: '{intent}'")

        if isinstance(intent, str) and intent:
            # The path map in the graph factory will determine the final destination.
            # This router's job is just to return the key for that map.
            return intent
        
        logger.warning("No valid intent found in state. Routing to '__end__'.")
        return "__end__"
