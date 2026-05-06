# SQL Agent - 独立 Text-to-SQL 智能体

基于 DeepAgents 框架的 Text-to-SQL 智能体，从 `my_agent.py` 中提取工具和模型逻辑，去除了对 Aix-DB 内部 ORM 模块的依赖，可直接通过环境变量配置数据库连接后独立运行。

## 目录结构

```
sql_agent/
├── __init__.py      # 模块入口，导出 DatasourceConfig 和 load_config_from_env
├── config.py        # 数据源连接配置（环境变量加载）
├── db.py            # SQLAlchemy Core 数据库操作
├── tools.py         # LangChain @tool 工具定义
├── agent.py         # DeepAgents 主程序
├── AGENTS.md        # Agent 指令（自动加载为 memory）
└── skills/          # 可选的 skills 目录
```

## 快速开始

### 1. 安装依赖

```bash
pip install sqlalchemy pymysql psycopg2-binary langchain-openai deepagents python-dotenv
```

按需安装数据库驱动：

| 数据库 | 驱动包 | 安装命令 |
|--------|--------|---------|
| MySQL | pymysql | `pip install pymysql` |
| PostgreSQL | psycopg2 | `pip install psycopg2-binary` |
| Oracle | oracledb | `pip install oracledb` |
| SQL Server | pymssql | `pip install pymssql` |
| ClickHouse | clickhouse-sqlalchemy | `pip install clickhouse-sqlalchemy` |

### 2. 配置环境变量

创建 `.env` 文件或直接导出环境变量：

#### MySQL 配置示例

```bash
# 数据库连接
DB_TYPE=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USERNAME=root
DB_PASSWORD=your_password
DB_DATABASE=your_database
# DB_SCHEMA=              # MySQL 不需要
DB_EXTRA_JDBC=charset=utf8mb4

# LLM 配置
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o
```

#### PostgreSQL 配置示例

```bash
# 数据库连接
DB_TYPE=postgresql
DB_HOST=127.0.0.1
DB_PORT=5432
DB_USERNAME=postgres
DB_PASSWORD=your_password
DB_DATABASE=your_database
DB_SCHEMA=public           # PostgreSQL 通常需要指定 schema
# DB_EXTRA_JDBC=

# LLM 配置
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o
```

#### SQL Server 配置示例

```bash
DB_TYPE=sqlserver
DB_HOST=127.0.0.1
DB_PORT=1433
DB_USERNAME=sa
DB_PASSWORD=your_password
DB_DATABASE=your_database
DB_SCHEMA=dbo
```

#### Oracle 配置示例

```bash
DB_TYPE=oracle
DB_HOST=127.0.0.1
DB_PORT=1521
DB_USERNAME=system
DB_PASSWORD=your_password
DB_DATABASE=ORCL           # service_name
DB_SCHEMA=SYSTEM
```

#### ClickHouse 配置示例

```bash
DB_TYPE=clickhouse
DB_HOST=127.0.0.1
DB_PORT=8123
DB_USERNAME=default
DB_PASSWORD=your_password
DB_DATABASE=your_database
```

### 3. 运行

```bash
# 命令行交互模式
python -m sql_agent.agent

# 或在 Python 代码中使用
from sql_agent.agent import create_sql_agent, run_query

agent = create_sql_agent()
answer = run_query(agent, "查询销售额最高的5个产品")
print(answer)
```

### 4. 代码中使用（自定义配置）

```python
from sql_agent.config import DatasourceConfig
from sql_agent.agent import create_sql_agent, run_query

# 直接构造配置（不从环境变量加载）
config = DatasourceConfig(
    db_type="mysql",
    host="127.0.0.1",
    port=3306,
    username="root",
    password="your_password",
    database="your_database",
    extra_jdbc="charset=utf8mb4",
)

agent = create_sql_agent(config=config)
answer = run_query(agent, "统计每个部门的员工数量")
```

### 5. 异步流式输出

```python
import asyncio
from sql_agent.agent import create_sql_agent, arun_query

async def main():
    agent = create_sql_agent()
    async for token in arun_query(agent, "查询今天的订单量"):
        print(token, end="", flush=True)

asyncio.run(main())
```

## 环境变量参考

### 数据库配置（前缀 `DB_`）

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `DB_TYPE` | 是 | 数据库类型：mysql / postgresql / oracle / sqlserver / clickhouse | - |
| `DB_HOST` | 否 | 主机地址 | 127.0.0.1 |
| `DB_PORT` | 否 | 端口号 | 按类型自动选择 |
| `DB_USERNAME` | 是 | 用户名 | - |
| `DB_PASSWORD` | 否 | 密码 | - |
| `DB_DATABASE` | 是 | 数据库名 | - |
| `DB_SCHEMA` | 否 | Schema 名（PostgreSQL/Oracle/SQL Server 建议设置） | 同 DB_DATABASE |
| `DB_EXTRA_JDBC` | 否 | 额外连接参数 | - |

默认端口映射：MySQL=3306, PostgreSQL=5432, Oracle=1521, SQL Server=1433, ClickHouse=8123

### LLM 配置（前缀 `OPENAI_`）

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | API Key |
| `OPENAI_BASE_URL` | API Base URL |
| `OPENAI_MODEL_NAME` | 模型名称 |

## 支持的数据库连接 URI

| 数据库类型 | URI 格式 | 驱动 |
|-----------|---------|------|
| MySQL | `mysql+pymysql://user:pass@host:port/db` | pymysql |
| PostgreSQL | `postgresql+psycopg2://user:pass@host:port/db` | psycopg2 |
| Oracle | `oracle+oracledb://user:pass@host:port?service_name=db` | oracledb |
| SQL Server | `mssql+pymssql://user:pass@host:port/db` | pymssql |
| ClickHouse | `clickhouse+http://user:pass@host:port/db` | clickhouse-sqlalchemy |

## 工具列表

| 工具名 | 功能 |
|--------|------|
| `sql_db_list_tables` | 列出数据库中所有表名及注释 |
| `sql_db_schema` | 获取指定表的字段信息（列名/类型/注释） |
| `sql_db_query` | 执行 SELECT 查询并返回结果（限制50行） |
| `sql_db_query_checker` | 检查 SQL 语法（不实际执行） |
| `sql_db_table_relationship` | 获取表之间的外键关系 |

## 与 my_agent.py 的主要区别

| 特性 | my_agent.py | sql_agent/ |
|------|-------------|-----------|
| 数据库元信息 | 依赖 Aix-DB ORM (DatasourceTable/Field) | 直接查询数据库 catalog |
| 配置来源 | Aix-DB 数据源管理 + AES 加密 | 环境变量或 DatasourceConfig |
| 表关系 | 从 Datasource.table_relation JSON 获取 | 从数据库外键约束自动发现 |
| 依赖 | model/db_connection_pool, model/datasource_models, common/datasource_util | 仅 sqlalchemy |
