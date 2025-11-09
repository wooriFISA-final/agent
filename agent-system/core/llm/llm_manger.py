"""
LLM Manager Module
LLM 설정 및 관리 (Ollama, Anthropic, OpenAI 지원)
"""
from typing import Optional, Dict, Any, List
from langchain_ollama import ChatOllama
# from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from core.config.setting import Settings
import logging

logger = logging.getLogger(__name__)


class LLMManager:
    """
    LLM 관리 클래스 (싱글톤)
    
    지원 Provider:
    - Ollama (로컬)
    - Anthropic (Claude)
    - OpenAI (GPT)
    """
    
    _instance: Optional['LLMManager'] = None
    _llm: Optional[BaseChatModel] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_llm(cls, **kwargs) -> BaseChatModel:
        """
        LLM 인스턴스 가져오기
        
        Args:
            **kwargs: LLM 설정 오버라이드
            
        Returns:
            LLM 인스턴스
        """
        if cls._llm is None:
            cls._llm = cls._create_llm(**kwargs)
        return cls._llm
    
    @classmethod
    def _create_llm(cls, **kwargs) -> BaseChatModel:
        """LLM 인스턴스 생성"""
        config = Settings.get_config()
        
        provider = kwargs.get("provider") or config.llm_provider
        model = kwargs.get("model") or config.llm_model
        temperature = kwargs.get("temperature") or config.llm_temperature
        
        # provider가 ollama가 아니거나 설정되지 않은 경우 기본값으로 ollama 사용
        if provider.lower() != "ollama":
            logger.warning(f"⚠️ Provider '{provider}' is not supported or not configured. Using Ollama as default.")
            provider = "ollama"
            # ollama 기본 모델 설정 (kwargs에 model이 없거나 config 모델과 같으면)
            if not kwargs.get("model") or model == config.llm_model:
                model = "qwen3:8b"  # ollama 기본 모델
        
        logger.info(f"🤖 Creating LLM: provider={provider}, model={model}")
        
        if provider.lower() == "ollama":
            # kwargs에서 이미 처리한 인자들 제거
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ["provider", "model", "temperature"]}
            return cls._create_ollama(model, temperature, **filtered_kwargs)
        # elif provider.lower() == "anthropic":
        #     return cls._create_anthropic(model, temperature, **kwargs)
        # elif provider.lower() == "openai":
        #     return cls._create_openai(model, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @classmethod
    def _create_ollama(
        cls, 
        model: str, 
        temperature: float,
        **kwargs
    ) -> ChatOllama:
        """
        Ollama LLM 생성
        
        기본 설정:
        - base_url: http://localhost:11434
        - model: 설정 모델
        """
        base_url = kwargs.get("base_url", "http://localhost:11434")
        
        llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            **{k: v for k, v in kwargs.items() 
               if k not in ["provider", "model", "temperature", "base_url"]}
        )
        
        logger.info(f"✅ Ollama LLM created: {model} at {base_url}")
        return llm
    
    # @classmethod
    # def _create_anthropic(
    #     cls,
    #     model: str,
    #     temperature: float,
    #     **kwargs
    # ) -> ChatAnthropic:
    #     """Anthropic Claude LLM 생성"""
    #     config = Settings.get_config()
    #     api_key = kwargs.get("api_key") or config.llm_api_key
        
    #     if not api_key:
    #         raise ValueError("Anthropic API key is required")
        
    #     llm = ChatAnthropic(
    #         model=model,
    #         temperature=temperature,
    #         api_key=api_key,
    #         **{k: v for k, v in kwargs.items() 
    #            if k not in ["provider", "model", "temperature", "api_key"]}
    #     )
        
    #     logger.info(f"✅ Anthropic LLM created: {model}")
    #     return llm
    
    # @classmethod
    # def _create_openai(
    #     cls,
    #     model: str,
    #     temperature: float,
    #     **kwargs
    # ) -> ChatOpenAI:
    #     """OpenAI GPT LLM 생성"""
    #     config = Settings.get_config()
    #     api_key = kwargs.get("api_key") or config.llm_api_key
        
    #     if not api_key:
    #         raise ValueError("OpenAI API key is required")
        
    #     llm = ChatOpenAI(
    #         model=model,
    #         temperature=temperature,
    #         api_key=api_key,
    #         **{k: v for k, v in kwargs.items() 
    #            if k not in ["provider", "model", "temperature", "api_key"]}
    #     )
        
    #     logger.info(f"✅ OpenAI LLM created: {model}")
    #     return llm
    
    @classmethod
    def reset(cls):
        """LLM 인스턴스 초기화 (재생성 시 사용)"""
        cls._llm = None
        logger.info("🔄 LLM instance reset")
    
    @classmethod
    async def test_connection(cls) -> bool:
        """
        LLM 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            llm = cls.get_llm()
            response = await llm.ainvoke([HumanMessage(content="Hello")])
            logger.info(f"✅ LLM connection test successful: {response.content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"❌ LLM connection test failed: {e}")
            return False


class LLMHelper:
    """LLM 사용을 위한 헬퍼 함수들"""
    
    @staticmethod
    async def invoke(
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        간단한 LLM 호출
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            **kwargs: LLM 설정 오버라이드
            
        Returns:
            LLM 응답 텍스트
        """
        llm = LLMManager.get_llm(**kwargs)
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = await llm.ainvoke(messages)
        return response.content
    
    @staticmethod
    async def invoke_with_history(
        prompt: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        대화 히스토리를 포함한 LLM 호출
        
        Args:
            prompt: 현재 프롬프트
            history: 대화 히스토리 [{"role": "user/assistant", "content": "..."}]
            system_prompt: 시스템 프롬프트
            **kwargs: LLM 설정
            
        Returns:
            LLM 응답
        """
        llm = LLMManager.get_llm(**kwargs)
        
        messages = []
        
        # 시스템 프롬프트
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # 히스토리 추가
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        
        # 현재 프롬프트
        messages.append(HumanMessage(content=prompt))
        
        response = await llm.ainvoke(messages)
        return response.content
    
    @staticmethod
    async def stream_invoke(
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        스트리밍 LLM 호출
        
        Args:
            prompt: 프롬프트
            system_prompt: 시스템 프롬프트
            **kwargs: LLM 설정
            
        Yields:
            응답 청크
        """
        llm = LLMManager.get_llm(**kwargs)
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        async for chunk in llm.astream(messages):
            yield chunk.content