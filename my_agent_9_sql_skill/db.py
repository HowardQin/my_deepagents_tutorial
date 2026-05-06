"""
数据库连接与查询执行模块

使用 SQLAlchemy Core 层直接操作数据库，不依赖 Aix-DB 的 ORM 模型。
从 common/datasource_util.py 提取核心逻辑，简化为独立可用的版本。
"""

import logging
import os
import urllib.parse
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from config import DatasourceConfig

logger = logging.getLogger(__name__)


# ==================== 连接 URI 构建 ====================

def build_connection_uri(config: DatasourceConfig) -> str:
    """
    根据 DatasourceConfig 构建 SQLAlchemy 连接 URI。

    支持的数据库类型及驱动：
    - mysql:        mysql+pymysql
    - postgresql:   postgresql+psycopg2
    - oracle:       oracle+oracledb
    - sqlserver:    mssql+pymssql
    - clickhouse:   clickhouse+http
    """
    username = urllib.parse.quote(config.username)
    password = urllib.parse.quote(config.password)
    host = config.host
    port = config.port
    database = config.database
    extra = f"?{config.extra_jdbc}" if config.extra_jdbc else ""

    if config.db_type == "mysql":
        return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}{extra}"

    elif config.db_type == "postgresql":
        return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}{extra}"

    elif config.db_type == "oracle":
        # Oracle 支持 service_name 和 SID 两种连接模式
        mode = os.environ.get("DB_ORACLE_MODE", "service_name")
        if mode == "service_name":
            return f"oracle+oracledb://{username}:{password}@{host}:{port}?service_name={database}{extra.replace('?', '&') if extra else ''}"
        return f"oracle+oracledb://{username}:{password}@{host}:{port}/{database}{extra}"

    elif config.db_type == "sqlserver":
        return f"mssql+pymssql://{username}:{password}@{host}:{port}/{database}{extra}"

    elif config.db_type == "clickhouse":
        return f"clickhouse+http://{username}:{password}@{host}:{port}/{database}{extra}"

    raise ValueError(f"不支持的数据库类型: {config.db_type}")


# ==================== 引擎创建 ====================

def create_engine_from_config(config: DatasourceConfig) -> Engine:
    """
    根据配置创建 SQLAlchemy Engine。
    """
    uri = build_connection_uri(config)
    timeout = int(os.environ.get("DB_TIMEOUT", "30"))

    if config.db_type == "oracle":
        engine = create_engine(uri, pool_pre_ping=True)
    elif config.db_type == "sqlserver":
        engine = create_engine(
            uri,
            pool_pre_ping=True,
            connect_args={"timeout": timeout, "login_timeout": timeout, "encryption": "off"},
        )
    else:
        engine = create_engine(
            uri,
            pool_pre_ping=True,
            connect_args={"connect_timeout": timeout},
        )
    return engine


# ==================== 值处理 ====================

def _process_row_value(value: Any) -> Any:
    """处理行数据中的特殊类型"""
    if isinstance(value, Decimal):
        return float(value)
    elif hasattr(value, "isoformat"):
        return value.isoformat()
    elif hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def _decode_bytes(value: Any) -> str:
    """处理 SQL Server 可能返回的 bytes 类型"""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return value.decode("latin-1")
            except Exception:
                return str(value)
    return value or ""


# ==================== 表列表查询 ====================

def _get_table_sql(db_type: str, database: str, schema: str) -> Tuple[str, str]:
    """获取查询表列表的 SQL 和参数名"""
    if db_type == "mysql":
        return (
            "SELECT TABLE_NAME, TABLE_COMMENT FROM information_schema.TABLES WHERE TABLE_SCHEMA = :param",
            database,
        )
    elif db_type == "postgresql":
        return (
            """
            SELECT c.relname AS TABLE_NAME,
                   COALESCE(d.description, obj_description(c.oid)) AS TABLE_COMMENT
            FROM pg_class c
            LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
            LEFT JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = 0
            WHERE n.nspname = :param
                AND c.relkind IN ('r', 'v', 'p', 'm')
                AND c.relname NOT LIKE 'pg_%'
                AND c.relname NOT LIKE 'sql_%'
            ORDER BY c.relname
            """,
            schema,
        )
    elif db_type == "oracle":
        return (
            """
            SELECT DISTINCT
                t.TABLE_NAME AS "TABLE_NAME",
                NVL(c.COMMENTS, '') AS "TABLE_COMMENT"
            FROM (
                SELECT TABLE_NAME, 'TABLE' AS OBJECT_TYPE FROM ALL_TABLES WHERE OWNER = :param
                UNION ALL
                SELECT VIEW_NAME AS TABLE_NAME, 'VIEW' AS OBJECT_TYPE FROM ALL_VIEWS WHERE OWNER = :param
            ) t
            LEFT JOIN ALL_TAB_COMMENTS c ON t.TABLE_NAME = c.TABLE_NAME AND c.OWNER = :param
            ORDER BY t.TABLE_NAME
            """,
            schema.upper(),
        )
    elif db_type == "sqlserver":
        return (
            """
            SELECT TABLE_NAME AS [TABLE_NAME],
                   ISNULL(ep.value, '') AS [TABLE_COMMENT]
            FROM INFORMATION_SCHEMA.TABLES t
            LEFT JOIN sys.extended_properties ep
                ON ep.major_id = OBJECT_ID(t.TABLE_SCHEMA + '.' + t.TABLE_NAME)
                AND ep.minor_id = 0 AND ep.name = 'MS_Description'
            WHERE t.TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                AND t.TABLE_SCHEMA = :param
            """,
            schema,
        )
    elif db_type == "clickhouse":
        return (
            """
            SELECT name, comment
            FROM system.tables
            WHERE database = :param AND engine NOT IN ('Dictionary')
            ORDER BY name
            """,
            database,
        )
    raise ValueError(f"不支持的数据库类型: {db_type}")


def get_tables(engine: Engine, config: DatasourceConfig) -> List[Dict[str, Any]]:
    """获取数据库中的表列表"""
    sql, param = _get_table_sql(config.db_type, config.database, config.effective_schema)
    tables = []

    with engine.connect() as conn:
        result = conn.execute(text(sql), {"param": param})
        for row in result.fetchall():
            table_name = _decode_bytes(row[0])
            table_comment = _decode_bytes(row[1])
            tables.append({"tableName": table_name, "tableComment": table_comment})

    return tables


# ==================== 字段查询 ====================

def _get_field_sql(db_type: str, database: str, schema: str, table_name: str) -> Tuple[str, str, str]:
    """获取查询字段列表的 SQL"""
    if db_type == "mysql":
        sql = """
            SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :param1
        """
        if table_name:
            sql += " AND TABLE_NAME = :param2"
        return sql, database, table_name

    elif db_type == "postgresql":
        sql = """
            SELECT a.attname AS COLUMN_NAME,
                   pg_catalog.format_type(a.atttypid, a.atttypmod) AS DATA_TYPE,
                   col_description(c.oid, a.attnum) AS COLUMN_COMMENT
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :param1
                AND a.attnum > 0
                AND NOT a.attisdropped
        """
        if table_name:
            sql += " AND c.relname = :param2"
        return sql, schema, table_name

    elif db_type == "oracle":
        sql = """
            SELECT col.COLUMN_NAME AS "COLUMN_NAME",
                   col.DATA_TYPE AS "DATA_TYPE",
                   NVL(com.COMMENTS, '') AS "COLUMN_COMMENT"
            FROM ALL_TAB_COLUMNS col
            LEFT JOIN ALL_COL_COMMENTS com
                ON col.OWNER = com.OWNER
                AND col.TABLE_NAME = com.TABLE_NAME
                AND col.COLUMN_NAME = com.COLUMN_NAME
            WHERE col.OWNER = :param1
        """
        if table_name:
            sql += " AND col.TABLE_NAME = :param2"
        return sql, schema.upper(), table_name.upper()

    elif db_type == "sqlserver":
        sql = """
            SELECT COLUMN_NAME AS [COLUMN_NAME],
                   DATA_TYPE AS [DATA_TYPE],
                   ISNULL(EP.value, '') AS [COLUMN_COMMENT]
            FROM INFORMATION_SCHEMA.COLUMNS C
            LEFT JOIN sys.extended_properties EP
                ON EP.major_id = OBJECT_ID(C.TABLE_SCHEMA + '.' + C.TABLE_NAME)
                AND EP.minor_id = C.ORDINAL_POSITION
                AND EP.name = 'MS_Description'
            WHERE C.TABLE_SCHEMA = :param1
        """
        if table_name:
            sql += " AND C.TABLE_NAME = :param2"
        return sql, schema, table_name

    elif db_type == "clickhouse":
        sql = """
            SELECT name AS COLUMN_NAME, type AS DATA_TYPE, comment AS COLUMN_COMMENT
            FROM system.columns
            WHERE database = :param1
        """
        if table_name:
            sql += " AND table = :param2"
        return sql, database, table_name

    raise ValueError(f"不支持的数据库类型: {db_type}")


def get_fields(
    engine: Engine, config: DatasourceConfig, table_name: str = ""
) -> List[Dict[str, Any]]:
    """获取指定表的字段列表"""
    sql, p1, p2 = _get_field_sql(config.db_type, config.database, config.effective_schema, table_name)
    fields = []

    with engine.connect() as conn:
        result = conn.execute(text(sql), {"param1": p1, "param2": p2})
        for idx, row in enumerate(result.fetchall()):
            fields.append({
                "fieldName": _decode_bytes(row[0]),
                "fieldType": _decode_bytes(row[1]),
                "fieldComment": _decode_bytes(row[2]),
                "fieldIndex": idx,
            })

    return fields


# ==================== 查询执行 ====================

def execute_query(engine: Engine, sql: str) -> List[Dict[str, Any]]:
    """
    执行 SQL SELECT 查询并返回结果（字典列表）。
    """
    while sql.endswith(";"):
        sql = sql[:-1]

    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        columns = list(result.keys())
        data = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                row_dict[col] = _process_row_value(row[i])
            data.append(row_dict)
        return data


# ==================== 外键关系查询 ====================

def get_foreign_keys(engine: Engine, config: DatasourceConfig) -> List[Dict[str, str]]:
    """
    从数据库外键约束中获取表关系信息。
    返回格式: [{"from_table": ..., "from_column": ..., "to_table": ..., "to_column": ...}]
    """
    inspector = inspect(engine)
    relationships = []

    try:
        # 获取所有表名
        table_names = inspector.get_table_names(schema=config.effective_schema)

        for table_name in table_names:
            fks = inspector.get_foreign_keys(table_name, schema=config.effective_schema)
            for fk in fks:
                referred_table = fk.get("referred_table", "")
                referred_schema = fk.get("referred_schema") or config.effective_schema
                constrained_columns = fk.get("constrained_columns", [])
                referred_columns = fk.get("referred_columns", [])

                for i, col in enumerate(constrained_columns):
                    ref_col = referred_columns[i] if i < len(referred_columns) else ""
                    if referred_table and ref_col:
                        relationships.append({
                            "from_table": table_name,
                            "from_column": col,
                            "to_table": referred_table,
                            "to_column": ref_col,
                        })
    except Exception as e:
        logger.warning(f"获取外键关系失败（某些数据库不支持）: {e}")

    return relationships
