"""
LLM Workflow Example
Ollama를 사용한 완전한 워크플로우
"""
import asyncio
from typing import TypedDict, Optional
from graph.builder.graph_builder import GraphBuilder
from agents.registry.agent_registry import AgentRegistry
from core.logging.logger import setup_logger
from core.llm.llm_manger import LLMManager
from langchain_core.messages import HumanMessage
import logging


class LLMStateSchema(TypedDict, total=False):
    """LLM 워크플로우용 상태 스키마"""
    query: str
    research_result: Optional[str]
    analysis_result: Optional[str]
    final_report: Optional[str]


async def test_ollama_connection():
    """Ollama 연결 테스트"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🔌 Testing Ollama Connection...")
    logger.info("=" * 60)
    
    try:
        # LLM 인스턴스 초기화 (provider를 ollama로 명시)
        LLMManager.reset()  # 기존 인스턴스 초기화
        llm = LLMManager.get_llm(provider="ollama", model="qwen3:8b")
        
        # 연결 테스트
        response = await llm.ainvoke([HumanMessage(content="Hello")])
        
        logger.info(f"✅ Ollama is ready! Response: {response.content[:50]}...")
        return True
            
    except Exception as e:
        logger.error(f"❌ Connection test error: {e}")
        logger.info("""
💡 Troubleshooting:
1. Install Ollama: https://ollama.ai
2. Start Ollama: ollama serve
3. Pull a model: ollama pull llama3.2
4. Check if running: curl http://localhost:11434
""")
        return False


async def run_llm_workflow():
    """LLM 기반 워크플로우 실행"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🚀 LLM Research Workflow Starting...")
    logger.info("=" * 60)
    
    # Agent 자동 발견
    AgentRegistry.auto_discover("agents.implementations")
    
    registered_agents = AgentRegistry.list_agents()
    logger.info(f"✅ Registered agents: {registered_agents}")
    
    # 필수 Agent 확인
    required_agents = ["intent_classifier"]
    missing = [a for a in required_agents if a not in registered_agents]
    
    if missing:
        logger.warning(f"⚠️ Missing required agents: {missing}")
        logger.info("💡 Using available agents only. Some agents may not be available.")
        logger.info("💡 To create missing agents, implement them in agents/implementations/")
    
    # 사용 가능한 Agent로 그래프 빌드
    builder = GraphBuilder(LLMStateSchema)
    
    # LLM Agent용 설정 (타임아웃을 120초로 늘림)
    llm_agent_config = {
        "timeout": 120,  # LLM 호출은 시간이 걸릴 수 있으므로 타임아웃을 늘림
        "max_retries": 2  # 재시도 횟수는 줄임
    }
    
    # intent_classifier 있으면 추가
    entry_point = None
    finish_point = None
    
    if "intent_classifier" in registered_agents:
        builder.add_agent_node("intent", "intent_classifier", config=llm_agent_config)
        entry_point = "intent"
        finish_point = "intent"
    else:
        logger.error("❌ intent_classifier agent not found!")
        return None
    
    # 그래프 설정
    if entry_point and finish_point:
        builder.set_entry_point(entry_point)
        builder.set_finish_point(finish_point)
    else:
        logger.error("❌ Cannot build graph: entry_point or finish_point is not set")
        return None
    
    # 그래프 컴파일
    graph = builder.build()
    
    logger.info("=" * 60)
    logger.info("📊 Graph Structure:")
    logger.info(f"   Entry: {entry_point}")
    logger.info(f"   Finish: {finish_point}")
    logger.info("=" * 60)
    
    # 초기 상태 설정
    initial_state = {
        "query": "계획을 수정하고 싶어"
    }
    
    logger.info(f"🔍 Starting workflow with query: '{initial_state['query']}'")
    logger.info("=" * 60)
    
    # 워크플로우 실행
    try:
        result = await graph.ainvoke(initial_state)
        
        logger.info("=" * 60)
        logger.info("✅ Workflow completed successfully!")
        logger.info("=" * 60)
        logger.info("📝 Results:")
        
        if "intent_result" in result:
            logger.info(f"   intent: {result['intent_result'][:100]}...")

        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def main():
    """메인 실행 함수"""
    # 로깅 설정
    logger = setup_logger()
    
    logger.info("=" * 60)
    logger.info("🚀 LLM Workflow Main")
    logger.info("=" * 60)
    
    # Ollama 연결 테스트
    connection_ok = await test_ollama_connection()
    
    if not connection_ok:
        logger.error("❌ Cannot proceed without Ollama connection")
        logger.info("💡 Please start Ollama and try again")
        return
    
    logger.info("")
    
    # 워크플로우 실행
    result = await run_llm_workflow()
    
    if result:
        logger.info("✅ All done!")
    else:
        logger.error("❌ Workflow failed")


if __name__ == "__main__":
    asyncio.run(main())