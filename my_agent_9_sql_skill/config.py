"""
数据库连接配置模块

从环境变量或 .env 文件加载数据库连接参数，
替代原 Aix-DB 中 DatasourceConfigUtil 加解密 + Datasource ORM 的方案。
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# 默认端口映射
DEFAULT_PORTS = {
    "mysql": 3306,
    "postgresql": 5432,
    "oracle": 1521,
    "sqlserver": 1433,
    "clickhouse": 8123,
}


@dataclass
class DatasourceConfig:
    """数据源连接配置"""

    db_type: str  # mysql / postgresql / oracle / sqlserver / clickhouse
    host: str = "127.0.0.1"
    port: int = 3306
    username: str = ""
    password: str = ""
    database: str = ""
    schema: Optional[str] = None  # PostgreSQL/Oracle/SQL Server 需要
    extra_jdbc: str = ""  # 额外 JDBC 参数，如 charset=utf8mb4

    @property
    def effective_schema(self) -> str:
        """获取有效的 schema 名称"""
        return self.schema or self.database


def load_config_from_env() -> DatasourceConfig:
    """
    从环境变量加载数据库配置。

    环境变量前缀为 DB_，支持：
        DB_TYPE       - 数据库类型 (mysql/postgresql/oracle/sqlserver/clickhouse)
        DB_HOST       - 主机地址
        DB_PORT       - 端口号（可选，默认按类型自动选择）
        DB_USERNAME   - 用户名
        DB_PASSWORD   - 密码
        DB_DATABASE   - 数据库名
        DB_SCHEMA     - Schema 名（PostgreSQL/Oracle/SQL Server 可能需要）
        DB_EXTRA_JDBC - 额外连接参数

    Returns:
        DatasourceConfig 实例

    Raises:
        ValueError: 缺少必需的配置项
    """
    db_type = os.getenv("DB_TYPE", "").lower()
    if not db_type:
        raise ValueError("环境变量 DB_TYPE 未设置，可选值: mysql, postgresql, oracle, sqlserver, clickhouse")

    supported = {"mysql", "postgresql", "oracle", "sqlserver", "clickhouse"}
    if db_type not in supported:
        raise ValueError(f"不支持的数据库类型: {db_type}，可选值: {', '.join(sorted(supported))}")

    host = os.getenv("DB_HOST", "127.0.0.1")
    port_str = os.getenv("DB_PORT", "")
    port = int(port_str) if port_str else DEFAULT_PORTS.get(db_type, 3306)

    username = os.getenv("DB_USERNAME", "")
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_DATABASE", "")
    schema = os.getenv("DB_SCHEMA") or None
    extra_jdbc = os.getenv("DB_EXTRA_JDBC", "")

    if not username:
        raise ValueError("环境变量 DB_USERNAME 未设置")
    if not database:
        raise ValueError("环境变量 DB_DATABASE 未设置")

    return DatasourceConfig(
        db_type=db_type,
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        schema=schema,
        extra_jdbc=extra_jdbc,
    )
