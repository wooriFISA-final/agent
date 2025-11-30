import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to sys.path
sys.path.append(os.getcwd())

logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"sys.path: {sys.path}")

try:
    logger.info("Attempting to import agents.implementations.report_template...")
    import agents.implementations.report_template
    logger.info("✅ Successfully imported agents.implementations.report_template")
except Exception as e:
    logger.error(f"❌ Failed to import agents.implementations.report_template: {e}")
    import traceback
    traceback.print_exc()

try:
    from agents.registry.agent_registry import AgentRegistry
    logger.info("Checking AgentRegistry...")
    agents = AgentRegistry.list_agents()
    logger.info(f"Registered agents: {agents}")
    
    if "report_agent" in agents:
        logger.info("✅ 'report_agent' is found in registry!")
    else:
        logger.warning("⚠️ 'report_agent' is NOT found in registry.")
        
    # Try auto discovery manually
    logger.info("Attempting auto_discover('agents.implementations')...")
    AgentRegistry.auto_discover("agents.implementations")
    agents_after = AgentRegistry.list_agents()
    logger.info(f"Registered agents after discovery: {agents_after}")

except Exception as e:
    logger.error(f"❌ Error checking registry: {e}")
    import traceback
    traceback.print_exc()
