"""
SQL Agent 工具模块

提供 LangChain @tool 装饰的 SQL 数据库操作工具，
供 deepagents 框架调用。不依赖 Aix-DB 的 ORM 层，
直接使用 sql_agent/db.py 的 SQLAlchemy Core 函数。
"""

import logging
import re
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import Engine

from config import DatasourceConfig
from db import (
    create_engine_from_config,
    execute_query,
    get_fields,
    get_foreign_keys,
    get_tables,
)

logger = logging.getLogger(__name__)

# ==================== 模块级状态管理 ====================

# 当前活跃的 Engine 和 Config，供工具函数直接使用
_current_engine: Optional[Engine] = None
_current_config: Optional[DatasourceConfig] = None


def set_datasource(config: DatasourceConfig) -> None:
    """
    设置当前数据源，创建 Engine 并保存到模块状态。

    在创建 Agent 之前调用一次即可，后续工具会自动使用此 Engine。

    Args:
        config: 数据源连接配置
    """
    global _current_engine, _current_config
    _current_config = config
    _current_engine = create_engine_from_config(config)
    logger.info(f"数据源已设置: {config.db_type}://{config.host}:{config.port}/{config.database}")


def get_current_engine() -> Engine:
    """获取当前 Engine，未设置时抛出异常"""
    if _current_engine is None:
        raise RuntimeError("数据源未设置，请先调用 set_datasource()")
    return _current_engine


def get_current_config() -> DatasourceConfig:
    """获取当前 Config，未设置时抛出异常"""
    if _current_config is None:
        raise RuntimeError("数据源未设置，请先调用 set_datasource()")
    return _current_config


# ==================== 工具函数 ====================


@tool
def sql_db_list_tables() -> str:
    """列出数据库中的所有表名及表注释。"""
    try:
        engine = get_current_engine()
        config = get_current_config()
        tables = get_tables(engine, config)

        if not tables:
            logger.info("列出表完成: 数据库中没有表")
            return "数据库中没有表"

        result_lines = ["数据库中有以下表：\n"]
        for t in tables:
            name = t.get("tableName", "")
            comment = t.get("tableComment", "")
            if comment:
                result_lines.append(f"- {name}: {comment}")
            else:
                result_lines.append(f"- {name}")

        result_lines.append("\n表列表已获取完成。如需查看表结构，请使用 sql_db_schema 工具。")
        result = "\n".join(result_lines)
        logger.info(f"列出表完成，共 {len(tables)} 张表:\n{result[:500]}")
        return result

    except Exception as e:
        logger.error(f"列出表失败: {e}", exc_info=True)
        return f"列出表失败: {str(e)[:200]}"


@tool
def sql_db_schema(table_names: str) -> str:
    """获取指定表的架构信息（列名、类型、注释）。

    Args:
        table_names: 表名，可以是单个表名或多个表名（用逗号分隔）
    """
    try:
        engine = get_current_engine()
        config = get_current_config()

        if isinstance(table_names, str):
            table_list = [t.strip() for t in table_names.split(",")]
        else:
            table_list = [table_names]

        schema_parts = []
        for table_name in table_list:
            fields = get_fields(engine, config, table_name)

            if not fields:
                schema_parts.append(f"表 '{table_name}' 不存在或没有字段")
                continue

            schema_text = f"\n表 '{table_name}':"
            schema_text += "\n列:"
            for field in fields:
                col_name = field.get("fieldName", "")
                col_type = field.get("fieldType", "")
                col_comment = field.get("fieldComment", "")
                schema_text += f"\n  - {col_name} ({col_type})"
                if col_comment:
                    schema_text += f" - {col_comment}"

            schema_parts.append(schema_text)

        result = "\n".join(schema_parts) if schema_parts else "未找到表信息"
        result += "\n\n表架构已获取完成。请基于此信息编写 SQL 查询，无需重复获取架构。"
        logger.info(f"获取表架构完成，表: {table_names}:\n{result[:500]}")
        return result

    except Exception as e:
        logger.error(f"获取表架构失败: {e}", exc_info=True)
        return f"获取表架构失败: {str(e)[:200]}"


@tool
def sql_db_query(query: str) -> str:
    """执行 SQL SELECT 查询并返回结果。
    只允许执行 SELECT 查询，不允许执行 INSERT、UPDATE、DELETE、DROP 等操作。

    Args:
        query: 要执行的 SQL 查询语句
    """
    # 安全检查：只允许 SELECT 查询
    query_upper = query.strip().upper()
    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
    for keyword in forbidden_keywords:
        if keyword in query_upper:
            return f"错误: 不允许执行 {keyword} 操作，只允许 SELECT 查询"

    if not query_upper.startswith("SELECT"):
        return "错误: 只允许执行 SELECT 查询"

    logger.info(f"执行 SQL 查询:\n{query[:500]}")

    try:
        engine = get_current_engine()
        result_data = execute_query(engine, query)

        if not result_data:
            return "查询成功执行，但没有返回数据。"

        # 格式化结果（限制返回行数）
        max_rows = 50
        result_rows = result_data[:max_rows]

        if len(result_data) > max_rows:
            result_str = f"查询成功，返回 {len(result_data)} 行数据（显示前 {max_rows} 行）:\n\n"
        else:
            result_str = f"查询成功，返回 {len(result_data)} 行数据:\n\n"

        if result_rows:
            columns = list(result_rows[0].keys())

            col_widths = {}
            for col in columns:
                col_widths[col] = min(
                    max(
                        len(str(col)),
                        max(len(str(row.get(col, ""))[:50]) for row in result_rows),
                    ),
                    50,
                )

            header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
            separator = "-" * min(len(header), 200)
            result_str += header + "\n" + separator + "\n"

            for row in result_rows:
                row_str = " | ".join(
                    str(row.get(col, ""))[:50].ljust(col_widths[col])
                    for col in columns
                )
                result_str += row_str + "\n"

        result_str += "\n查询已完成。请基于以上结果进行分析，无需重复执行相同查询。"
        logger.info(f"SQL 查询完成，返回 {len(result_data)} 行数据:\n{result_str[:500]}")
        return result_str

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."

        logger.error(f"SQL 查询失败: {error_msg}")
        return (
            f"SQL 执行失败: {error_msg}\n\n"
            "请检查 SQL 语法和表结构是否正确。"
            "如果之前已获取表架构，请直接使用已有信息，无需重复查询。"
        )


@tool
def sql_db_query_checker(query: str) -> str:
    """检查 SQL 查询的语法是否正确。
    注意：这只是一个基本的检查，不会实际执行查询。

    Args:
        query: 要检查的 SQL 查询语句
    """
    query_upper = query.strip().upper()

    if not query_upper:
        return "错误: SQL 查询为空"

    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
    for keyword in forbidden_keywords:
        if query_upper.startswith(keyword):
            return f"错误: 不允许执行 {keyword} 操作，只允许 SELECT 查询"

    if not query_upper.startswith("SELECT"):
        return "错误: 只允许执行 SELECT 查询"

    issues = []
    if "FROM" not in query_upper:
        issues.append("缺少 FROM 子句")

    if query.count("(") != query.count(")"):
        issues.append("括号不匹配")

    if query.count("'") % 2 != 0:
        issues.append("单引号不匹配")

    if issues:
        result = f"SQL 语法警告: {', '.join(issues)}"
        logger.info(f"SQL 语法检查结果: {result}")
        return result

    logger.info("SQL 语法检查通过")
    return "SQL 查询语法检查通过，可以执行。"


@tool
def sql_db_table_relationship(table_names: str = "") -> str:
    """获取指定表之间的外键关系信息。

    通过数据库外键约束自动发现表关联，帮助编写正确的 JOIN 查询。

    Args:
        table_names: 逗号分隔的表名列表，如 "t_orders, t_customers"。
                    如果为空，则返回所有表的外键关系。
    """
    try:
        engine = get_current_engine()
        config = get_current_config()
        relationships = get_foreign_keys(engine, config)

        if not relationships:
            return (
                "未发现数据库外键约束。\n\n"
                "可能的原因：\n"
                "1. 数据库中确实没有定义外键约束\n"
                "2. 可以尝试通过列名推断关系（如 customer_id 可能关联 customers.id）\n\n"
                "提示：可以查看表架构中的列名来推断可能的关联关系。"
            )

        # 根据传入的表名过滤
        if table_names and table_names.strip():
            filter_tables = {t.strip() for t in table_names.split(",")}
            relationships = [
                r
                for r in relationships
                if r.get("from_table") in filter_tables
                or r.get("to_table") in filter_tables
            ]

        if not relationships:
            return f"表 {table_names} 之间没有外键关系。"

        result_lines = ["表之间的外键关系如下：\n"]
        for rel in relationships:
            from_table = rel.get("from_table", "")
            from_col = rel.get("from_column", "")
            to_table = rel.get("to_table", "")
            to_col = rel.get("to_column", "")
            result_lines.append(f"  - {from_table}.{from_col} -> {to_table}.{to_col}")

        result_lines.append("\n表关系已获取完成。")
        result_lines.append("请使用以上关系信息编写正确的 JOIN 语句。")
        result = "\n".join(result_lines)
        logger.info(f"获取表关系完成，共 {len(relationships)} 条外键关系:\n{result[:500]}")
        return result

    except Exception as e:
        logger.error(f"获取表关系失败: {e}", exc_info=True)
        return f"获取表关系失败: {str(e)[:200]}"


# ==================== 工具列表 ====================

SQL_TOOLS = [
    sql_db_list_tables,
    sql_db_schema,
    sql_db_query,
    sql_db_query_checker,
    sql_db_table_relationship,
]
