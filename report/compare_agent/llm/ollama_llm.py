from langchain_ollama import ChatOllama

# Ollama 모델 초기화
ollama_llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.3
)
