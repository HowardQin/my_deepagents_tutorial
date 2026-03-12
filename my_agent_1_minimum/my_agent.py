import os
from pathlib import Path
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from shell import ShellMiddleware
import requests
from langchain_core.tools import tool
from openai import OpenAI

load_dotenv()

# 这些信息以环境变量设置，这些环境变量可以写在.env中
agent_model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL_NAME"),
    temperature=0,
)

# deepagent 需要一个文件系统存放中间结果，
# 这里设置一个虚拟文件系统，
# 当前目录下的 fs 作为虚拟文件系统的根目录“/”
CURRENT_DIR = Path(__file__).parent.resolve()
FS_ROOT = CURRENT_DIR / "fs"
my_backend = FilesystemBackend(root_dir=str(FS_ROOT), virtual_mode=True)

shell_middleware = ShellMiddleware(
    workspace_root=FS_ROOT,
    timeout=1000.0,
    max_output_bytes=100000,
)

agent = create_deep_agent(
    model=agent_model,
    backend=my_backend,
    middleware=[shell_middleware],
    debug=True
)