"""
LLM Manager Module
LLM 설정 및 관리 (Ollama HTTP API 직접 호출)
"""
from typing import Optional, Dict, Any, List
import requests
from core.logging.logger import setup_logger
from core.config.setting import settings

logger = setup_logger()


class LLMManager:
    """
    LLM 관리 클래스 (싱글톤)
    Ollama Chat API를 직접 호출하는 방식
    """
    
    _instance: Optional['LLMManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        전역 기본 LLM 설정 가져오기 (settings 기반)
        
        Returns:
            기본 LLM 설정 딕셔너리
        """
        return {
            "base_url": str(settings.LLM_API_BASE_URL),
            "provider": settings.LLM_PROVIDER,
            "model": settings.LLM_MODEL,
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": settings.LLM_TOP_P,
            "top_k": settings.LLM_TOP_K,
            "num_ctx": settings.LLM_NUM_CTX,
            "stream": settings.LLM_STREAM,
            "format": settings.LLM_FORMAT,
            "timeout": settings.LLM_TIMEOUT
        }
    
    @classmethod
    def merge_config(cls, **overrides) -> Dict[str, Any]:
        """
        기본 설정과 오버라이드 병합
        
        Args:
            **overrides: 덮어쓸 설정값들
            
        Returns:
            병합된 LLM 설정
        """
        config = cls.get_default_config()
        
        # overrides에 있는 값만 업데이트
        for key in config.keys():
            if key in overrides and overrides[key] is not None:
                config[key] = overrides[key]
        
        logger.debug(f"🤖 Merged LLM Config: {config}")
        return config
    
    @classmethod
    def test_connection(cls, **config_overrides) -> bool:
        """
        LLM 연결 테스트
        
        Args:
            **config_overrides: 테스트용 설정 오버라이드
            
        Returns:
            연결 성공 여부
        """
        try:
            config = cls.merge_config(**config_overrides)
            response = cls._call_ollama_chat(
                messages=[{"role": "user", "content": "Hello"}],
                model=config["model"],
                base_url=config["base_url"],
                timeout=10,
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                num_ctx=config["num_ctx"],
                stream=False
            )
            logger.info(f"✅ LLM connection test successful: {config['base_url']}")
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
        stream: bool = False,
        format: str = "",
        **kwargs  # ✅ 추가 파라미터 받기
    ) -> str:
        """
        API 호출 
        
        Args:
            messages: 메시지 리스트 [{"role": "user/assistant/system", "content": "..."}]
            model: 모델 이름
            base_url: Ollama 서버 URL
            timeout: 타임아웃 (초)
            stream: 스트리밍 여부
            format: 응답 포맷 (json 등)
            **kwargs: temperature, top_k, top_p, num_ctx 등
            
        Returns:
            LLM 응답 텍스트
        """
        # ✅ options 객체 생성
        options = {}
        ollama_option_keys = [
            'temperature', 'top_k', 'top_p', 'min_p',
            'num_ctx', 'num_predict', 'seed', 'stop'
        ]
        
        for key in ollama_option_keys:
            if key in kwargs and kwargs[key] is not None:
                options[key] = kwargs[key]
        
        # Payload 구성
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # options가 있으면 추가
        if options:
            payload["options"] = options
        
        # format이 있으면 추가
        if format:
            payload["format"] = format
        
        logger.debug(f"🤖 Ollama API Request: {payload}")
        
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Ollama API result: {result}")
            
            # stream=False인 경우 message.content 반환
            if not stream:
                content = (
                    result.get("message", {}).get("content")  # Ollama 기본 구조
                    or (
                        result.get("choices", [{}])[0]  # OpenAI 호환 구조
                        .get("message", {})
                        .get("content")
                    )
                    or ""
                )
                return content.strip()
            
            # stream=True인 경우는 별도 처리 필요
            return result
            
        except requests.exceptions.Timeout:
            error_msg = f"❌ Ollama 타임아웃 ({timeout}초 초과). 모델: {model}, URL: {base_url}"
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
    """LLM 사용을 위한 헬퍼 함수들"""
    
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
        config = LLMManager.merge_config(**kwargs)
        
        # 메시지 구성
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return LLMManager._call_ollama_chat(
            messages=messages,
            model=config["model"],
            base_url=config["base_url"],
            timeout=config["timeout"],
            stream=kwargs.get("stream", False),
            format=kwargs.get("format", ""),
            # ✅ options 대신 개별 파라미터 전달
            temperature=kwargs.get("temperature", config["temperature"]),
            top_k=kwargs.get("top_k", config["top_k"]),
            top_p=kwargs.get("top_p", config["top_p"]),
            num_ctx=kwargs.get("num_ctx", config["num_ctx"])
        )
    
    @staticmethod
    def invoke_with_history(
        history: List[Dict[str, str]],
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
        config = LLMManager.merge_config(**kwargs)
        
        # 메시지 구성
        messages = []
        
        
        # 히스토리 추가
        messages.extend(history)
        
        # # 현재 프롬프트
        # if prompt:
        #     messages.append({"role": "user", "content": prompt})
        
        return LLMManager._call_ollama_chat(
            messages=messages,
            model=config["model"],
            base_url=config["base_url"],
            timeout=config["timeout"],
            stream=kwargs.get("stream", False),
            format=kwargs.get("format", ""),
            # ✅ options 대신 개별 파라미터 전달
            temperature=kwargs.get("temperature", config["temperature"]),
            top_k=kwargs.get("top_k", config["top_k"]),
            top_p=kwargs.get("top_p", config["top_p"]),
            num_ctx=kwargs.get("num_ctx", config["num_ctx"])
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
        config = LLMManager.merge_config(**kwargs)
        
        # 메시지 구성
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # ✅ options 객체 생성
        options = {
            "temperature": kwargs.get("temperature", config["temperature"]),
            "top_k": kwargs.get("top_k", config["top_k"]),
            "top_p": kwargs.get("top_p", config["top_p"]),
            "num_ctx": kwargs.get("num_ctx", config["num_ctx"])
        }
        
        payload = {
            "model": config["model"],
            "messages": messages,
            "stream": True,
            "options": options
        }
        
        if kwargs.get("format"):
            payload["format"] = kwargs["format"]
        
        try:
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                json=payload,
                timeout=config["timeout"],
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