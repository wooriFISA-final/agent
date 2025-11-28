"""
LLM Manager Module
LLM 설정 및 관리 (AWS Bedrock Converse API)
"""
from typing import Optional, Dict, Any, List, Tuple, Union
import boto3
from botocore.exceptions import ClientError
from core.logging.logger import setup_logger
from core.config.setting import settings

logger = setup_logger()


class LLMManager:
    """
    LLM 관리 클래스 (싱글톤)
    AWS Bedrock Converse API 사용
    """
    
    _instance: Optional['LLMManager'] = None
    _bedrock_client = None  # boto3 클라이언트 캐시
    _current_region = None  # 현재 설정된 리전
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def _get_bedrock_client(cls, region: str):
        """
        boto3 Bedrock 클라이언트 가져오기 (재사용)
        
        Args:
            region: AWS 리전
            
        Returns:
            boto3 Bedrock Runtime 클라이언트
        """
        # 리전이 변경되었거나 클라이언트가 없으면 새로 생성
        if cls._bedrock_client is None or cls._current_region != region:
            logger.info(f"새로운 Bedrock 클라이언트를 생성합니다. 리전: {region}")
            cls._bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region
            )
            cls._current_region = region
            logger.info("Bedrock 클라이언트가 생성되고 캐시되었습니다.")
        
        return cls._bedrock_client
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        전역 기본 LLM 설정 가져오기 (AWS Bedrock 기반)
        
        Returns:
            기본 LLM 설정 딕셔너리
        """
        return {
            "region": str(settings.AWS_REGION),
            "model_id": settings.BEDROCK_MODEL_ID,
            "temperature": settings.LLM_TEMPERATURE,
            "top_p": settings.LLM_TOP_P,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": settings.LLM_STREAM,
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
        
        logger.debug(f"병합된 LLM 설정: {config}")
        return config
    
    @classmethod
    def test_connection(cls, **config_overrides) -> bool:
        """
        LLM 연결 테스트 (Bedrock)
        
        Args:
            **config_overrides: 테스트용 설정 오버라이드
            
        Returns:
            연결 성공 여부
        """
        try:
            config = cls.merge_config(**config_overrides)
            response = cls._call_bedrock_converse(
                messages=[{"role": "user", "content": "Hello"}],
                model_id=config["model_id"],
                region=config["region"],
                timeout=10,
                temperature=config["temperature"],
                top_p=config["top_p"],
                max_tokens=config.get("max_tokens", 10000)
            )
            logger.info(f"Bedrock 연결 테스트 성공: region={config['region']}, model={config['model_id']}")
            return True
        except Exception as e:
            logger.error(f"Bedrock 연결 테스트 실패: {e}")
            return False
    
    @classmethod
    def _prepare_bedrock_messages(
        cls,
        messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Ollama 포맷 메시지를 Bedrock Converse 포맷으로 변환
        
        Args:
            messages: Ollama 포맷 메시지 리스트
            
        Returns:
            Tuple[system_messages, conversation_messages]
        """
        system_messages = []
        conversation_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                # System 메시지는 별도 배열로
                system_messages.append({"text": content})
                
            elif role in ["user", "assistant"]:
                # content가 이미 Bedrock 형식의 리스트인지 확인
                # (예: [{"toolUse": {...}}, {"text": "..."}])
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    # reasoningContent 블록 필터링 (Extended Thinking 모델용)
                    # toolUse, text, image 등만 유지
                    filtered_content = [
                        block for block in content 
                        if not isinstance(block, dict) or "reasoningContent" not in block
                    ]
                    
                    # 필터링 후 content가 비어있으면 빈 텍스트 블록 추가
                    if not filtered_content:
                        filtered_content = [{"text": ""}]
                    
                    # 이미 Bedrock 형식이면 필터링된 content 사용
                    conversation_messages.append({
                        "role": role,
                        "content": filtered_content
                    })
                else:
                    # 일반 텍스트 메시지
                    conversation_messages.append({
                        "role": role,
                        "content": [{"text": str(content)}]
                    })
                
            elif role == "tool":
                # ToolMessage 처리
                # LangChain ToolMessage에서 tool_call_id와 content 추출
                tool_use_id = msg.get("tool_call_id", "unknown")
                
                # Tool 결과를 파싱 시도 (JSON인지 확인)
                try:
                    import json
                    result_json = json.loads(content)
                    tool_content = [{"json": result_json}]
                except:
                    # JSON이 아니면 텍스트로 처리
                    tool_content = [{"text": content}]
                
                # toolResult 블록 생성
                tool_result_block = {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": tool_content
                    }
                }
                
                # 이전 메시지가 toolResult를 포함하는 user 메시지인지 확인
                if (conversation_messages and 
                    conversation_messages[-1]["role"] == "user" and 
                    "toolResult" in conversation_messages[-1]["content"][0]):
                    
                    # 기존 메시지에 추가
                    conversation_messages[-1]["content"].append(tool_result_block)
                else:
                    # 새로운 user 메시지 생성
                    conversation_messages.append({
                        "role": "user",
                        "content": [tool_result_block]
                    })
        
        return system_messages, conversation_messages
    
    @classmethod
    def _handle_tool_response(cls, response: Dict) -> Tuple[bool, Optional[Dict]]:
        """
        Bedrock 응답에서 tool_use 확인 및 정보 추출
        
        Args:
            response: Bedrock converse() 응답
            
        Returns:
            Tuple[is_tool_requested, tool_request_info]
            - is_tool_requested: Tool이 요청되었는지 여부
            - tool_request_info: Tool 요청 정보 (toolUseId, name, input)
        """
        stop_reason = response.get("stopReason")
        
        if stop_reason == "tool_use":
            # Tool 요청 추출
            message = response.get("output", {}).get("message", {})
            content = message.get("content", [])
            
            for block in content:
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    return True, {
                        "tool_use_id": tool_use["toolUseId"],
                        "tool_name": tool_use["name"],
                        "tool_input": tool_use.get("input", {})
                    }
        
        return False, None
    
    @classmethod
    def _call_bedrock_converse(
        cls,
        messages: List[Dict[str, str]],
        model_id: str,
        region: str,
        timeout: int = 180,
        tool_config: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        **kwargs
    ) -> Dict:
        """
        AWS Bedrock Converse API 호출
        
        Args:
            messages: 메시지 리스트
            model_id: Bedrock 모델 ID
            region: AWS 리전
            timeout: 타임아웃 (초)
            tool_config: Bedrock toolConfig (선택)
            tool_choice: Bedrock toolChoice (선택, 예: {"any": {}}, {"auto": {}}, {"tool": {"name": "tool_name"}})
            **kwargs: temperature, top_p, max_tokens 등
            
        Returns:
            Dict: 전체 Bedrock 응답 (stopReason, output, usage 등 포함)
        """
        # 메시지 변환
        system_messages, conversation_messages = cls._prepare_bedrock_messages(messages)
        
        logger.info(f"Bedrock API 호출 준비")
        logger.info(f"   Region: {region}")
        logger.info(f"   Model ID: {model_id}")
        logger.info(f"   System messages: {len(system_messages)}")
        logger.info(f"   Conversation messages: {len(conversation_messages)}")
        
        # AWS 환경 변수 확인
        import os
        logger.info(f"AWS_BEARER_TOKEN_BEDROCK: {'설정됨' if os.getenv('AWS_BEARER_TOKEN_BEDROCK') else '없음'}")
        
        # Bedrock 클라이언트 가져오기 (재사용)
        client = cls._get_bedrock_client(region)
        
        # inferenceConfig 구성
        inference_config = {}
        if "temperature" in kwargs:
            inference_config["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            inference_config["topP"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            inference_config["maxTokens"] = kwargs["max_tokens"]
        
        # API 요청 파라미터
        request_params = {
            "modelId": model_id,
            "messages": conversation_messages
        }
        
        if system_messages:
            request_params["system"] = system_messages
        
        # toolConfig 추가
        if tool_config:
            request_params["toolConfig"] = tool_config
            logger.info(f"✅ toolConfig 추가: {len(tool_config.get('tools', []))}개의 도구")
            
            # toolChoice 추가 (toolConfig가 있을 때만)
            if tool_choice:
                request_params["toolConfig"]["toolChoice"] = tool_choice
                logger.info(f"✅ toolChoice 추가: {tool_choice}")
        
        if inference_config:
            request_params["inferenceConfig"] = inference_config
        
        logger.debug(f"요청 파라미터 키: {list(request_params.keys())}")
        
        try:
            logger.info("Bedrock API 호출 시도...")
            response = client.converse(**request_params)
            logger.info("Bedrock API 호출 성공")
            logger.debug(f"응답 키: {list(response.keys())}")
            
            # 토큰 사용량 로깅
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            total_tokens = usage.get("totalTokens", 0)
            logger.info(f"📊 Token Usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            # 전체 응답 반환 (stopReason 포함)
            return response
            
        except ClientError as e:
            logger.error(f"Bedrock ClientError:")
            logger.error(f"   Error Code: {e.response['Error']['Code']}")
            logger.error(f"   Error Message: {e.response['Error']['Message']}")
            logger.error(f"   HTTP Status: {e.response['ResponseMetadata']['HTTPStatusCode']}")
            raise RuntimeError(f"Bedrock API error: {e}")
        except Exception as e:
            logger.error(f"예상치 못한 오류:")
            logger.error(f"   에러 타입: {type(e).__name__}")
            logger.error(f"   에러 메시지: {str(e)}")
            import traceback
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
            raise


class LLMHelper:
    """LLM 사용을 위한 헬퍼 함수들"""
    
    @staticmethod
    def invoke(
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        간단한 LLM 호출 (Bedrock)
        
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
        
        response = LLMManager._call_bedrock_converse(
            messages=messages,
            model_id=config["model_id"],
            region=config["region"],
            timeout=config["timeout"],
            temperature=kwargs.get("temperature", config["temperature"]),
            top_p=kwargs.get("top_p", config["top_p"]),
            max_tokens=kwargs.get("max_tokens", config["max_tokens"])
        )
        
        # 텍스트만 추출
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        
        if content_blocks:
            for block in content_blocks:
                if "text" in block:
                    return block["text"]
        return ""

    
    @staticmethod
    def invoke_with_history(
        history: List[Dict[str, str]],
        tool_config: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
        return_full_response: bool = False,
        **kwargs
    ) -> Union[str, Dict]:
        """
        대화 히스토리를 포함한 LLM 호출 (Bedrock)
        
        Args:
            history: 대화 히스토리 [{"role": "user/assistant/system", "content": "..."}]
            tool_config: Bedrock toolConfig (선택)
            tool_choice: Bedrock toolChoice (선택, 예: {"any": {}}, {"auto": {}}, {"tool": {"name": "tool_name"}})
            return_full_response: True면 전체 응답, False면 텍스트만
            **kwargs: LLM 설정
            
        Returns:
            str 또는 Dict: return_full_response에 따라
        """
        config = LLMManager.merge_config(**kwargs)
        
        response = LLMManager._call_bedrock_converse(
            messages=history,
            model_id=config["model_id"],
            region=config["region"],
            timeout=config["timeout"],
            tool_config=tool_config,
            tool_choice=tool_choice,
            temperature=kwargs.get("temperature", config["temperature"]),
            top_p=kwargs.get("top_p", config["top_p"]),
            max_tokens=kwargs.get("max_tokens", config["max_tokens"])
        )
        
        # return_full_response에 따라 처리
        if return_full_response:
            return response  # 전체 응답 (stopReason 포함)
        else:
            # 텍스트만 추출
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            
            if content_blocks:
                for block in content_blocks:
                    if "text" in block:
                        return block["text"]
            return ""

    
    @staticmethod
    def stream_invoke(
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        스트리밍 LLM 호출 (현재 미지원)
        
        Bedrock Converse 스트리밍은 converse_stream() API 사용 필요
        현재 구현되지 않음
        """
        raise NotImplementedError("Streaming is not yet implemented for Bedrock Converse API")