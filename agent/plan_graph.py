from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from plan_agents.intent_classifier import IntentClassifierAgent

class AgentState(MessagesState):
    intent: str = ""
    confidence: float = 0.0
    reason: str = ""
    plan: list[str] = []
    past_steps: list = []
    response: str = ""

def create_graph():
    # IntentClassifierAgent 초기화
    intent_agent = IntentClassifierAgent().create_intent_node()

    # StateGraph 생성
    graph = StateGraph(AgentState)

    # intent_classifier 노드 추가
    graph.add_node("intent_classifier", intent_agent)

    # 시작 → intent_classifier → 종료
    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", END)

    # MemorySaver 체크포인트 포함
    return graph.compile(checkpointer=MemorySaver())

