"""
SQL Agent 主程序

基于 deepagents 框架的 Text-to-SQL 智能体。
通过环境变量配置数据库连接。

"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from config import DatasourceConfig, load_config_from_env
from tools import SQL_TOOLS, set_datasource
from langchain_deepseek import ChatDeepSeek

load_dotenv()

logger = logging.getLogger(__name__)

# 当前脚本所在目录
CURRENT_DIR = Path(__file__).parent.resolve()
my_backend = LocalShellBackend(root_dir=str(CURRENT_DIR), virtual_mode=True)
#my_backend = FilesystemBackend(root_dir=str(CURRENT_DIR), virtual_mode=True)

# ==================== LLM 配置 ====================
# 1. 初始化 LLM
# Instantiate the model with thinking disabled
model = ChatDeepSeek(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),  # Or your preferred model
    temperature=0,
    # This extra_body dictionary disables the reasoning/thinking mode
    extra_body={"thinking": {"type": "disabled"}} 
)

#model = ChatOpenAI(
#    api_key=os.getenv("OPENAI_API_KEY"),
#    base_url=os.getenv("OPENAI_BASE_URL"),
#    model=os.getenv("OPENAI_MODEL_NAME"),
#    temperature=0,
#)
# ==================== Agent 创建 ====================

# 2. 初始化数据源
config = load_config_from_env()
set_datasource(config)

# 3. 创建 Agent
agent = create_deep_agent(
    model=model,
    skills=["/skills"],
    tools=SQL_TOOLS,
    backend=my_backend,
)
