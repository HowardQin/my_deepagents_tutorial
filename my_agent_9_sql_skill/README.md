# NL2SQL Agent

基于 [DeepAgents] 框架的 Text-to-SQL 智能体，能将自然语言问题自动转换为 SQL 并执行查询。

## 架构概览

```
main.py / agent.py          → Agent 入口，加载模型、工具和技能
config.py                   → 数据源配置（从 .env 加载）
db.py                       → SQLAlchemy 数据库操作层
tools.py                    → LangChain @tool 工具定义
skills/nl2sql/SKILL.md      → NL2SQL 技能流程规范
skills/nl2sql/TABLES/       → 业务表说明文档
skills/nl2sql/FEWSHOTS/     → SQL 写法示例
```

## 支持的数据库

| 类型 | 驱动 |
|------|------|
| MySQL | pymysql |
| PostgreSQL | psycopg2 |
| Oracle | oracledb |
| SQL Server | pymssql |
| ClickHouse | http |

## 快速开始

### 1. 配置环境

复制 `.env.template` 为 `.env`，填入数据库和 LLM 连接信息：

```bash
cp .env.template .env
```

关键环境变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `DB_TYPE` | 数据库类型 | `oracle` / `mysql` / `postgresql` |
| `DB_HOST` | 数据库地址 | `172.32.148.108` |
| `DB_PORT` | 端口 | `1521` |
| `DB_USERNAME` | 用户名 | `paastest` |
| `DB_PASSWORD` | 密码 | `****` |
| `DB_DATABASE` | 数据库名 | `orcl12c` |
| `DB_SCHEMA` | Schema（PG/Oracle/SQLServer） | `paastest` |
| `OPENAI_API_KEY` | LLM API Key | `sk-xxx` |
| `OPENAI_BASE_URL` | LLM API 地址 | `https://api.deepseek.com` |
| `OPENAI_MODEL_NAME` | 模型名 | `deepseek-v4-flash` |

### 2. 启动

```bash
uv run langgraph dev --allow-blocking --host 0.0.0.0 --port 2024
```

Agent 会自动加载 `skills/` 目录下的 `nl2sql` 技能。

## 可用工具

Agent 提供 5 个 SQL 数据库工具，由 LangChain `@tool` 装饰器定义（`tools.py:66-313`）：

| 工具 | 功能 |
|------|------|
| `sql_db_list_tables` | 列出数据库中所有表及注释 |
| `sql_db_schema` | 获取指定表的字段名、类型、注释 |
| `sql_db_table_relationship` | 获取表间外键关系 |
| `sql_db_query_checker` | 预检查 SQL 语法 |
| `sql_db_query` | 执行 SELECT 查询（仅允许 SELECT） |

所有工具共享模块级 Engine 状态，通过 `set_datasource()` 初始化（`tools.py:34`）。

## NL2SQL 技能流程

技能定义在 `skills/nl2sql/SKILL.md`，Agent 遇到数据库查询问题时强制按以下流程执行：

```
第一步：读取 TABLES 目录获取表名 → 验证表是否存在 → 获取 Schema 和外键关系
第二步：理解用户意图 → 明确性检查（不确定时追问用户）
第三步：编写并执行 SQL（只允许 SELECT）
```

核心安全约束：
- **只允许 SELECT**，禁止 INSERT/UPDATE/DELETE/DROP 等写操作
- 表名必须原样来自 TABLES 文档，严禁自行修正
- 表名验证失败时立即终止，不得自行映射到相似表

## 扩展方法

### 添加新业务表

在 `skills/nl2sql/TABLES/` 下新建 `.md` 文件，格式参照现有文件：

```markdown
## Table Name
<name>YOUR_TABLE_NAME</name>

## 表作用
简要说明表的用途。

## 字段说明
| 字段名 | 类型 | 含义 |
|--------|------|------|
| col_1 | VARCHAR2 | 列含义 |
| col_2 | NUMBER | 列含义 |
```

### 添加 SQL 示例

在 `skills/nl2sql/FEWSHOTS/` 下新建 `.md` 文件，提供 SQL 写法参考。示例中的表名和列名仅用于展示写法模式，不会直接出现在最终 SQL 中——Agent 会替换为已验证的实际表名和列名。

### 添加新数据库类型

1. `config.py` — `DEFAULT_PORTS` 加入默认端口
2. `db.py` — 实现 `_get_table_sql`、`_get_field_sql`、`build_connection_uri` 对应分支
3. `tools.py` — `sql_db_query_checker` 中补充对应数据库的引号规则

### 添加新工具

在 `tools.py` 中用 `@tool` 装饰器定义函数，并将其加入 `SQL_TOOLS` 列表（`tools.py:318`）。
