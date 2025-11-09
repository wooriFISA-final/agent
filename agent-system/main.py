import asyncio
from graph.builder.graph_builder import GraphBuilder
from agents.registry.agent_registry import AgentRegistry
from core.logging.logger import setup_logger
from mystatechema import MyStateSchema
import logging


async def main():
    # 로깅 설정
    logger = setup_logger()
    
    # Agent 자동 발견
    AgentRegistry.auto_discover("agents.implementations")
    
    logger.info(f"✅ Registered agents: {AgentRegistry.list_agents()}")
    
    # 그래프 빌드
    builder = GraphBuilder(MyStateSchema)
    builder.add_agent_node("research", "research") \
        .add_agent_node("analyze", "analysis") \
        .add_edge("research", "analyze") \
        .set_entry_point("research") \
        .set_finish_point("analyze")
    
    graph = builder.build()
    
    # ✅ dict로 전달 (BaseModel이 아닌)
    initial_state = {
        "query": "AI agents architecture"
    }
    
    result = await graph.ainvoke(initial_state)
    
    logger.info(f"✅ Final Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())