import sys
import os
sys.path.append(os.getcwd())

# Mock settings if needed, or rely on .env
# We need to make sure core.config.setting can be imported
try:
    from core.config.setting import settings
    print(f"Loaded settings. AGENTS_MODULE_PATH: {settings.AGENTS_MODULE_PATH}")
except Exception as e:
    print(f"Warning: Could not load settings: {e}")

from agents.config.agent_config_loader import AgentConfigLoader
from agents.registry.agent_registry import AgentRegistry

print("--- 1. Testing Import ---")
try:
    import agents.implementations.report_template
    print("✅ Successfully imported report_template")
except Exception as e:
    print(f"❌ Failed to import report_template: {e}")
    import traceback
    traceback.print_exc()

print("\n--- 2. Testing Config Loading ---")
try:
    # We need to point to the correct yaml file for report_agent
    # The loader might look in default locations or we need to specify it
    # In lifespan.py, it loads from specific path.
    # But AgentRegistry.register decorator uses AgentConfigLoader.get_agent_config_from_current(agent_name)
    # which likely looks at agents/config/agents.yaml or similar.
    # Let's see if we can load it.
    
    # Force load the report_agents.yaml if possible, or check if it's loaded by default
    # AgentConfigLoader implementation details are needed, but let's try default first
    config = AgentConfigLoader.get_agent_config_from_current("report_agent")
    if config:
        print(f"✅ Successfully loaded config for report_agent. Enabled: {config.enabled}")
    else:
        print("⚠️ Config not found via default path. Trying to load specific file...")
        loader = AgentConfigLoader("agents/config/report_agents.yaml")
        config = loader.get_agent_config("report_agent")
        if config:
             print(f"✅ Successfully loaded config from report_agents.yaml. Enabled: {config.enabled}")
        else:
             print("❌ Config still not found")

except Exception as e:
    print(f"❌ Failed to load config: {e}")
    import traceback
    traceback.print_exc()

print("\n--- 3. Testing Auto Discovery ---")
try:
    # We need to ensure the config is loaded so that 'enabled' check passes
    # If auto_discover relies on get_agent_config_from_current, we might need to setup the loader first
    # In lifespan.py, it does:
    # AgentRegistry.auto_discover()
    # But before that, it doesn't seem to load specific yamls globally?
    # Wait, AgentRegistry.register uses AgentConfigLoader.get_agent_config_from_current(agent_name)
    # If that returns None, it might default to enabled=True or False?
    # Let's check AgentConfigLoader.get_agent_config_from_current implementation if this fails.
    
    AgentRegistry.auto_discover()
    if "report_agent" in AgentRegistry._agents:
        print("✅ report_agent successfully registered via auto_discover")
    else:
        print(f"❌ report_agent NOT registered. Registered agents: {list(AgentRegistry._agents.keys())}")
except Exception as e:
    print(f"❌ Auto-discovery failed: {e}")
    import traceback
    traceback.print_exc()
