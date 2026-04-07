初始化：
uv add -r requirements.txt

进入子目录：
cd my_agent_1_minimum

启动:
uv run langgraph dev --allow-blocking --host 0.0.0.0 --port 2024

使用langsmith chat 或 Agent-Chat-UI 连接。

生成的结果文件在fs目录下。

如果需要langsmith调试，在.env 里修改你自己的LANGSMITH_API_KEY。
