"""
SQL Agent - 基于 DeepAgents 框架的独立 Text-to-SQL 智能体

从 Aix-DB 的 my_agent.py 和 agent/deepagent 中提取的核心 SQL 工具，
使用 deepagents 框架驱动，可独立于 Aix-DB 主服务运行。

模块结构:
    config  - 数据源连接配置 (DatasourceConfig, load_config_from_env)
    db      - SQLAlchemy 数据库操作 (execute_query, get_tables, get_fields, get_foreign_keys)
    tools   - LangChain @tool 工具定义 (sql_db_list_tables, sql_db_schema, sql_db_query, ...)
    agent   - deepagents 主程序 (create_sql_agent, run_query, arun_query)
"""

from sql_agent.config import DatasourceConfig, load_config_from_env

__all__ = [
    "DatasourceConfig",
    "load_config_from_env",
]
