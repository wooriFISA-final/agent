"""
LLM Manager Module
LLM 설정 및 관리 (Ollama HTTP API 직접 호출, LangChain 의존성 제거)
"""
from typing import Optional, Dict, Any, List
import requests
from core.logging.logger import setup_logger

logger = setup_logger()


class LLMManager:
    """
    LLM 관리 클래스 (싱글톤)
    Ollama Chat API를 직접 호출하는 방식 (LangChain 불필요)
    """
    
    _instance: Optional['LLMManager'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_config(cls, **kwargs) -> Dict[str, Any]:
        """
        LLM 설정 가져오기
        
        Args:
            **kwargs: LLM 설정 오버라이드
            
        Returns:
            LLM 설정 딕셔너리
        """
        if cls._config is None:
            cls._config = cls._create_config(**kwargs)
        return cls._config
    
    @classmethod
    def _create_config(cls, **kwargs) -> Dict[str, Any]:
        """LLM 설정 생성"""
        provider = kwargs.get("provider", "ollama")
        model = kwargs.get("model", "qwen3:8b")
        temperature = kwargs.get("temperature", 0.3)
        base_url = kwargs.get("base_url", "http://localhost:11434")
        timeout = kwargs.get("timeout", 180)
        
        llm_config = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "base_url": base_url,
            "timeout": timeout
        }
        
        logger.info(f"🤖 LLM Config: provider={provider}, model={model}, base_url={base_url}")
        return llm_config
    
    @classmethod
    def reset(cls):
        """LLM 설정 초기화"""
        cls._config = None
        logger.info("🔄 LLM config reset")
    
    @classmethod
    def test_connection(cls) -> bool:
        """
        LLM 연결 테스트
        
        Returns:
            연결 성공 여부
        """
        try:
            config = cls.get_config()
            response = cls._call_ollama_chat(
                messages=[{"role": "user", "content": "Hello"}],
                model=config["model"],
                base_url=config["base_url"],
                timeout=10,
                temperature=0.1,
                stream=False
            )
            logger.info(f"✅ LLM connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ LLM connection test failed: {e}")
            return False
    
    @classmethod
    def _call_ollama_chat(
        cls,
        messages: List[Dict[str, str]],
        model: str,
        base_url: str,
        timeout: int = 180,
        temperature: float = 0.3,
        stream: bool = False,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ollama Chat API 호출
        
        Args:
            messages: 메시지 리스트 [{"role": "user/assistant/system", "content": "..."}]
            model: 모델 이름
            base_url: Ollama 서버 URL
            timeout: 타임아웃 (초)
            temperature: 온도 설정
            stream: 스트리밍 여부
            format: 응답 포맷 (json 등)
            options: 추가 옵션
            
        Returns:
            LLM 응답 텍스트
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options or {"temperature": temperature}
        }
        
        if format:
            payload["format"] = format
        
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(f"chat ollama result : {result}")
            # stream=False인 경우 message.content 반환
            if not stream:
                return result.get('message', {}).get('content', '').strip()
            
            # stream=True인 경우는 별도 처리 필요
            return result
            
        except requests.exceptions.Timeout:
            error_msg = f"❌ Ollama 타임아웃 ({timeout}초 초과). 모델: {model}"
            logger.error(error_msg)
            raise TimeoutError(error_msg)
            
        except requests.exceptions.ConnectionError:
            error_msg = f"❌ Ollama 서버 연결 실패: {base_url}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Ollama API 호출 오류: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


class LLMHelper:
    """LLM 사용을 위한 헬퍼 함수들 (LangChain 불필요)"""
    
    @staticmethod
    def invoke(
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
        config = LLMManager.get_config(**kwargs)
        
        # kwargs에서 설정 추출
        model = kwargs.get("model", config["model"])
        base_url = kwargs.get("base_url", config["base_url"])
        temperature = kwargs.get("temperature", config["temperature"])
        timeout = kwargs.get("timeout", config["timeout"])
        format_type = kwargs.get("format")
        options = kwargs.get("options")
        
        # 메시지 구성
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return LLMManager._call_ollama_chat(
            messages=messages,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            stream=False,
            format=format_type,
            options=options
        )
    
    @staticmethod
    def invoke_with_history(
        prompt: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        대화 히스토리를 포함한 LLM 호출
        
        Args:
            prompt: 현재 프롬프트
            history: 대화 히스토리 [{"role": "user/assistant/system", "content": "..."}]
            system_prompt: 시스템 프롬프트
            **kwargs: LLM 설정
            
        Returns:
            LLM 응답
        """
        config = LLMManager.get_config(**kwargs)
        
        model = kwargs.get("model", config["model"])
        base_url = kwargs.get("base_url", config["base_url"])
        temperature = kwargs.get("temperature", config["temperature"])
        timeout = kwargs.get("timeout", config["timeout"])
        format_type = kwargs.get("format")
        options = kwargs.get("options")
        
        # 메시지 구성
        messages = []
        
        # 시스템 프롬프트
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 히스토리 추가
        messages.extend(history)
        
        # 현재 프롬프트
        messages.append({"role": "user", "content": prompt})
        
        return LLMManager._call_ollama_chat(
            messages=messages,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            stream=False,
            format=format_type,
            options=options
        )
    
    @staticmethod
    def stream_invoke(
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
        config = LLMManager.get_config(**kwargs)
        
        model = kwargs.get("model", config["model"])
        base_url = kwargs.get("base_url", config["base_url"])
        temperature = kwargs.get("temperature", config["temperature"])
        timeout = kwargs.get("timeout", config["timeout"])
        format_type = kwargs.get("format")
        options = kwargs.get("options")
        
        # 메시지 구성
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options or {"temperature": temperature}
        }
        
        if format_type:
            payload["format"] = format_type
        
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    message = chunk.get('message', {})
                    content = message.get('content', '')
                    if content:
                        yield content
                        
        except Exception as e:
            logger.error(f"❌ 스트리밍 오류: {e}")
            raise