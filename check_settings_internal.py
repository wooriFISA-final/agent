import sys
import os

# 현재 디렉토리(agent)를 sys.path에 추가
sys.path.append(os.getcwd())

from core.config.setting import settings

print(f"MCP_URL: {settings.MCP_URL}")
print(f"API_PORT: {settings.API_PORT}")
