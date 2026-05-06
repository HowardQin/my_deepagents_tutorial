## Table Name
<name>CHATBI_RESULT_DAY_1</name>
又称为"经分会表1"

## 表作用
日粒度指标汇总表，存储各层级（省/地市/区县/网格）的 ICT 业务相关指标数据，包含指标值及其同比、环比增长率。

## 字段说明

| 字段名 | 类型 | 含义 |
|--------|------|------|
| PARTITION_KEY | NUMBER | 分区键 |
| STAT_DATE | NUMBER | 统计日期，格式 YYYYMMDD（如 20250430） |
| DIM_VALUE_1 | VARCHAR2 | 维度值，地域/组织的中文名称（如"南京市"、"江宁分公司"、"句容后白网格"） |
| DIM_LEVEL_NAME | VARCHAR2 | 维层名称，取值：`省级`、`地市级`、`区县级`、`网格级` |
| PARENT_DIM_NAME | VARCHAR2 | 父级维度名称（如"江宁分公司"的父级是"南京市"；"句容后白网格"的父级是"句容分公司"） |
| IND_VALUE_1 | NUMBER | ICT签约额（主指标） |
| IND_TB_1 | NUMBER | ICT签约额同比增长率（%） |
| IND_HB_1 | NUMBER | ICT签约额环比增长率（%） |
| IND_VALUE_1 | NUMBER | ICT签约额 |
| IND_TB_1 | NUMBER | ICT签约额同比增长率（%） |
| IND_HB_1 | NUMBER | ICT签约额环比增长率（%） |
| IND_VALUE_2 | NUMBER | IT集成服务部分签约额 |
| IND_TB_2 | NUMBER | IT集成服务部分签约额同比（%） |
| IND_HB_2 | NUMBER | IT集成服务部分签约额环比（%） |
| IND_VALUE_3 | NUMBER | ICT集成收入 |
| IND_TB_3 | NUMBER | ICT集成收入同比（%） |
| IND_HB_3 | NUMBER | ICT集成收入环比（%） |
| IND_VALUE_4 | NUMBER | 物联网产品收入 |
| IND_TB_4 | NUMBER | 物联网产品收入同比（%） |
| IND_HB_4 | NUMBER | 物联网产品收入环比（%） |
| IND_VALUE_5 | NUMBER | 物联网在网用户数 |
| IND_TB_5 | NUMBER | 物联网在网用户数同比（%） |
| IND_HB_5 | NUMBER | 物联网在网用户数环比（%） |
| IND_VALUE_6 | NUMBER | 物联网通信用户数 |
| IND_TB_6 | NUMBER | 物联网通信用户数同比（%） |
| IND_HB_6 | NUMBER | 物联网通信用户数环比（%） |
| IND_VALUE_7 | NUMBER | 物联网出账用户数 |
| IND_TB_7 | NUMBER | 物联网出账用户数同比（%） |
| IND_HB_7 | NUMBER | 物联网出账用户数环比（%） |
| IND_VALUE_8 | NUMBER | 待确认（与IND_VALUE_4物联网产品收入走势相近，但数值更大，可能是累计合同数或项目数） |
| IND_TB_8 | NUMBER | 对应指标同比（%） |
| IND_HB_8 | NUMBER | 对应指标环比（%） |
| IND_VALUE_9 | NUMBER | 待确认（与IND_VALUE_1比例稳定在约3%，可能是管理费或服务收入类指标） |
| IND_TB_9 | NUMBER | 对应指标同比（%） |
| IND_HB_9 | NUMBER | 对应指标环比（%） |
| IND_VALUE_10~30 | NUMBER | 表中全部为NULL，当前未使用 |

## 关键说明

1. **层级关系**：`PARENT_DIM_NAME` 指向上一级组织，例如：
   - 地市级 → 父级为省级
   - 区县级 → 父级为地市级
   - 网格级 → 父级为区县级

2. **DIM_VALUE_1 直接存储中文名称**，非编码值，无需关联维度字典表解码。

3. **IND_TB_N / IND_HB_N** 为预计算的同比/环比百分比值，无需手动计算。

4. **数据范围**：STAT_DATE 从 20250131 至 20250910（日粒度）。

5. **指标映射来源**：IND_VALUE_1~7 的含义通过与 `CHATBI_RESULT_1_NEW`（集团网格相关月指标1）的同名字段列注释比对确认。该表有完整的列注释可作为参考。

6. **IND_VALUE_8~10 说明**：在 `CHATBI_RESULT_1_NEW` 中同样无注释，且 `CHATBI_RESULT_DAY_1` 中仅月末（如每月最后一天）有值，非每日更新。

## 使用示例

```sql
-- 查询2025年4月30日南京市区县级ICT签约额
SELECT STAT_DATE, DIM_VALUE_1, IND_VALUE_1 AS ict_签约额, IND_HB_1 AS 环比
FROM CHATBI_RESULT_DAY_1
WHERE PARENT_DIM_NAME = '南京市'
  AND DIM_LEVEL_NAME = '区县级'
  AND STAT_DATE = 20250430
ORDER BY IND_VALUE_1 DESC;

-- 查询南京全市及下辖区县的同期对比
SELECT STAT_DATE, DIM_VALUE_1, DIM_LEVEL_NAME, IND_VALUE_1, IND_HB_1
FROM CHATBI_RESULT_DAY_1
WHERE (DIM_VALUE_1 = '南京市' OR PARENT_DIM_NAME = '南京市')
  AND STAT_DATE = 20250430
ORDER BY DIM_LEVEL_NAME, DIM_VALUE_1;
```
