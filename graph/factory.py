"""
Graph Factory Module
YAML 파일에서 그래프 구조를 읽어 GraphBuilder를 통해 LangGraph를 생성
"""
from typing import Optional, Any
import yaml
from pathlib import Path

from graph.builder.graph_builder import GraphBuilder
from graph.routing.router_base import RouterBase
from core.logging.logger import setup_logger

logger = setup_logger()


def mk_graph(yaml_path: str, checkpointer: Optional[Any] = None):
    """
    YAML 파일로부터 Agent 그래프를 생성
    
    Args:
        yaml_path: YAML 파일 경로
        checkpointer: Checkpointer 인스턴스
                     - None이면 내부에서 새로 생성 (테스트용)
                     - 프로덕션에서는 전역 Checkpointer 전달 권장
        
    Returns:
        컴파일된 LangGraph 객체 또는 None
    """
    try:
        # 1. YAML 파일 로드
        config = _load_yaml_config(yaml_path)
        if not config:
            logger.error("Failed to load YAML config")
            return None
        
        # 2. GraphBuilder 생성
        builder = GraphBuilder()
        
        # 3. 노드 추가
        nodes = config.get("nodes", [])
        if not nodes:
            logger.error("No nodes defined in YAML")
            return None
        
        for node in nodes:
            node_name = node.get("name")
            agent_name = node.get("agent")
            node_config = node.get("config", {})
            
            if not node_name or not agent_name:
                logger.warning(f"Invalid node definition: {node}")
                continue
            
            builder.add_agent_node(
                node_name=node_name,
                agent_name=agent_name,
                config=node_config
            )
            logger.info(f"Added node: {node_name} (agent: {agent_name})")
        
        # 4. 엣지 추가
        edges = config.get("edges", [])
        for edge in edges:
            from_node = edge.get("from")
            to_node = edge.get("to")
            
            if not from_node or not to_node:
                logger.warning(f"Invalid edge definition: {edge}")
                continue
            
            builder.add_edge(from_node, to_node)
            logger.info(f"Added edge: {from_node} → {to_node}")
        
        # 5. 조건부 엣지 추가
        conditional_edges = config.get("conditional_edges", [])
        for ce in conditional_edges:
            from_node = ce.get("from")
            router_class_name = ce.get("router")
            path_map = ce.get("paths", {})
            
            if not from_node or not router_class_name or not path_map:
                logger.warning(f"Invalid conditional edge: {ce}")
                continue
            
            # Router 인스턴스 생성
            try:
                router = _create_router_instance(router_class_name)
                builder.add_conditional_edge(
                    from_node=from_node,
                    router=router,
                    path_map=path_map
                )
                logger.info(f"Added conditional edge from {from_node}")
            except Exception as e:
                logger.error(f"Failed to create router {router_class_name}: {e}")
                continue
        
        # 6. Entry/Finish 포인트 설정
        entry_point = config.get("entry_point")
        if entry_point:
            builder.set_entry_point(entry_point)
            logger.info(f"Set entry point: {entry_point}")
        
        finish_points = config.get("finish_points", [])
        for finish in finish_points:
            builder.set_finish_point(finish)
            logger.info(f"Set finish point: {finish}")
        
        # 7. 그래프 빌드 (Checkpointer 전달)
        logger.info("Building graph...")
        if checkpointer:
            logger.info(f"Using provided checkpointer: {type(checkpointer).__name__}")
        else:
            logger.warning("No checkpointer provided. Creating new MemorySaver (not recommended for production)")
        
        graph = builder.build(checkpointer=checkpointer)
        
        # 8. 그래프 구조 출력
        logger.info("\n" + builder.visualize_structure())
        
        return graph
        
    except Exception as e:
        logger.error(f"Failed to create graph from YAML: {e}", exc_info=True)
        return None


def _load_yaml_config(yaml_path: str) -> dict:
    """YAML 파일 로드"""
    try:
        path = Path(yaml_path)
        if not path.exists():
            logger.error(f"YAML file not found: {yaml_path}")
            return {}
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded YAML config from: {yaml_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load YAML file {yaml_path}: {e}")
        return {}


def _create_router_instance(router_class_name: str) -> RouterBase:
    """
    Router 클래스 이름으로부터 인스턴스 생성
    
    Args:
        router_class_name: Router 클래스 이름
        
    Returns:
        RouterBase 인스턴스
    """
    # 동적 import
    import importlib
    
    # graph.routing 패키지에서 찾기
    try:
        module = importlib.import_module("graph.routing")
        router_class = getattr(module, router_class_name)
        return router_class()
    except (ImportError, AttributeError) as e:
        logger.error(f"Router class {router_class_name} not found: {e}")
        raise


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    from agents.config.base_config import StateBuilder
    from langgraph.checkpoint.memory import MemorySaver
    
    async def test_graph():
        """그래프 테스트 예시"""
        # 전역 Checkpointer 생성
        checkpointer = MemorySaver()
        
        # 그래프 생성 (Checkpointer 전달)
        graph = mk_graph("graph.yaml", checkpointer=checkpointer)
        
        if not graph:
            print("❌ Failed to create graph")
            return
        
        print("✅ Graph created successfully")
        
        # 초기 상태
        initial_state = StateBuilder.create_initial_state(
            messages=[HumanMessage(content="김철수(25세) 등록해줘")],
            session_id="test-session",
            max_iterations=10
        )
        
        # 그래프 실행 (첫 번째 메시지)
        config = {"configurable": {"thread_id": "test-session"}}
        print("\n🚀 Executing first message...")
        result1 = await graph.ainvoke(initial_state, config=config)
        print(f"✅ First response: {result1.get('last_result')}")
        print(f"📊 Messages: {len(result1.get('messages', []))}")
        
        # 같은 세션에서 두 번째 메시지 (이전 대화 유지됨)
        follow_up_state = {
            "messages": [HumanMessage(content="방금 등록한 사람 조회해줘")]
        }
        print("\n🚀 Executing second message (continuing conversation)...")
        result2 = await graph.ainvoke(follow_up_state, config=config)
        print(f"✅ Second response: {result2.get('last_result')}")
        print(f"📊 Total messages: {len(result2.get('messages', []))}")
        
        # 새 세션으로 테스트
        new_session_config = {"configurable": {"thread_id": "new-session"}}
        new_state = StateBuilder.create_initial_state(
            messages=[HumanMessage(content="이영희(30세) 등록해줘")],
            session_id="new-session",
            max_iterations=10
        )
        print("\n🚀 Executing with new session...")
        result3 = await graph.ainvoke(new_state, config=new_session_config)
        print(f"✅ New session response: {result3.get('last_result')}")
        print(f"📊 New session messages: {len(result3.get('messages', []))}")
    
    asyncio.run(test_graph())