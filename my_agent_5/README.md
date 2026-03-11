初始化：
uv add -r requirements.txt

启动，默认监听端口2024，使用langsmith chat连接：
uv run langgraph dev --allow-blocking --host 0.0.0.0

生成的结果文件在fs目录下。

如果需要langsmith调试，在.env 里修改你自己的LANGSMITH_API_KEY。