## 示例：基础查询
**用户问题：** "查询所有来自上海的用户的姓名和年龄。"
```sql
SELECT DISTINCT `column_1`, `column_2` FROM `table_users` WHERE `column_3` = '上海'
```

## 示例：聚合统计
**用户问题：** "统计2023年每个月的订单总额。"
```sql
SELECT `month_field` AS `month`, SUM(`amount_field`) AS `total_sales` FROM `table_orders` WHERE `year_field` = '2023' GROUP BY `month_field` ORDER BY `month_field`
```

## 示例：比较类问题
**用户问题：** "比较2024年和2025年的销售总额。"
```sql
SELECT `year_field` AS `year`, SUM(`sales_amount_field`) AS `total_sales` FROM `table_sales` WHERE `year_field` IN ('2024', '2025') GROUP BY `year_field` ORDER BY `year_field`
```

## 示例：区间类问题
**用户问题：** "统计2024年上半年（1月至6月）的新增客户数量。"
```sql
SELECT COUNT(`customer_id_field`) AS `new_customers` FROM `table_customers` WHERE `register_month_field` BETWEEN 1 AND 6 AND `register_year_field` = '2024'
```

## 示例：比率/占比类问题
**用户问题：** "计算2024年男性客户占总客户数的比例。"
```sql
SELECT (COUNT(CASE WHEN `gender_field` = '男' THEN 1 END) * 1.0 / COUNT(*)) AS `male_ratio` FROM `table_customers` WHERE `register_year_field` = '2024'
```

## 示例：排序/TOP类问题
**用户问题：** "查询销售额最高的前10个客户。"
```sql
SELECT `customer_id_field`, SUM(`sales_amount_field`) AS `total_sales` FROM `table_sales` GROUP BY `customer_id_field` ORDER BY `total_sales` DESC FETCH FIRST 10 ROWS ONLY
```

## 示例：复杂条件类问题
**用户问题：** "查询2024年在北京注册、且消费金额大于1000的用户数量。"
```sql
SELECT COUNT(`user_id_field`) AS `qualified_users` FROM `table_users` WHERE `register_city_field` = '北京' AND `register_year_field` = '2024' AND `total_spent_field` > 1000
```

## 示例：同比/环比类问题
**用户问题：** "计算2025年1月相对于2024年12月的订单增长率。"
```sql
SELECT ( (SUM(CASE WHEN `month_field` = '2025-01' THEN `order_count_field` END) - SUM(CASE WHEN `month_field` = '2024-12' THEN `order_count_field` END)) * 1.0 / SUM(CASE WHEN `month_field` = '2024-12' THEN `order_count_field` END) ) AS `growth_rate` FROM `table_orders` WHERE `month_field` IN ('2024-12', '2025-01')
```

## 示例：区间统计 + 分组
**用户问题：** "统计每个城市2025年Q1的订单数量。"
```sql
SELECT `city_field`, COUNT(`order_id_field`) AS `order_count` FROM `table_orders` WHERE `order_month_field` BETWEEN 1 AND 3 AND `order_year_field` = '2025' GROUP BY `city_field` ORDER BY `order_count` DESC
```

## 示例：数值比较 + 聚合
**用户问题：** "统计销售额超过平均值的商品数量。"
```sql
SELECT COUNT(*) AS `above_avg_products` FROM `table_products` WHERE `sales_amount_field` > (SELECT AVG(`sales_amount_field`) FROM `table_products`)
```

## 示例：时间趋势类问题
**用户问题：** "计算最近30天每天的新增客户数量。"
```sql
SELECT `date_field` AS `day`, COUNT(`customer_id_field`) AS `new_customers` FROM `table_customers` WHERE `date_field` >= CURRENT_DATE - INTERVAL '30' DAY GROUP BY `date_field` ORDER BY `date_field`
```

## 示例：分类占比 + 条件聚合
**用户问题：** 统计2025年各城市VIP用户占比。
```sql
SELECT `city_field`, COUNT(CASE WHEN `vip_flag_field` = 1 THEN 1 END) * 1.0 / COUNT(*) AS `vip_ratio` FROM `table_users` WHERE `register_year_field` = '2025' GROUP BY `city_field`
```

## 示例：复杂拆解示例
**用户问题：** "我们公司有多少新入职的应届生？他们的平均工资是多少？"
*** 查询1：新入职的应届生人数 ***
```sql
SELECT COUNT(`employee_id_field`) AS `new_grads` FROM `table_employees` WHERE `is_new_grad_field` = 1 AND `join_year_field` = '2025'
```
*** 查询2：新入职应届生的平均工资 ***
```sql
SELECT AVG(`salary_field`) AS `avg_salary` FROM `table_employees` WHERE `is_new_grad_field` = 1 AND `join_year_field` = '2025'
```

## 示例：按国家统计收入

```sql
SELECT
    c.Country,
    ROUND(SUM(i.Total), 2) as TotalRevenue
FROM Invoice i
INNER JOIN Customer c ON i.CustomerId = c.CustomerId
GROUP BY c.Country
ORDER BY TotalRevenue DESC
LIMIT 10;
```

## 示例：多维度销售分析（多条 SQL）

**用户问题：** "分析今年的月度销售趋势，为什么8月特别高？"

**查询 1：月度趋势**
```sql
SELECT
    DATE_FORMAT(order_date, '%Y-%m') as month,
    ROUND(SUM(amount), 2) as total_sales,
    COUNT(*) as order_count
FROM orders
WHERE YEAR(order_date) = 2024
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month ASC;
```

**查询 2：8月品类分解归因**
```sql
SELECT
    p.category,
    ROUND(SUM(o.amount), 2) as sales,
    COUNT(*) as order_count
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE DATE_FORMAT(o.order_date, '%Y-%m') = '2024-08'
GROUP BY p.category
ORDER BY sales DESC;
```

**查询 3：8月 vs 7月品类对比**
```sql
SELECT
    p.category,
    ROUND(SUM(CASE WHEN DATE_FORMAT(o.order_date, '%Y-%m') = '2024-08' THEN o.amount ELSE 0 END), 2) as aug_sales,
    ROUND(SUM(CASE WHEN DATE_FORMAT(o.order_date, '%Y-%m') = '2024-07' THEN o.amount ELSE 0 END), 2) as jul_sales,
    ROUND(SUM(CASE WHEN DATE_FORMAT(o.order_date, '%Y-%m') = '2024-08' THEN o.amount ELSE 0 END) -
          SUM(CASE WHEN DATE_FORMAT(o.order_date, '%Y-%m') = '2024-07' THEN o.amount ELSE 0 END), 2) as diff
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE DATE_FORMAT(o.order_date, '%Y-%m') IN ('2024-07', '2024-08')
GROUP BY p.category
ORDER BY diff DESC;
```

## 示例：Oracle 数据库 — 探索式查询（真实案例）

**用户问题：** "2025年4月1日泰州市及其下所有区县的ICT签约额"

**场景特点：** Oracle 数据库，表名和列名无直接中文注释，需通过 `user_tab_comments` 和 `user_col_comments` 探索元数据。

**步骤 1：浏览表列表，寻找相关表**
```sql
-- 查看所有表及其注释
SELECT table_name, comments FROM user_tab_comments WHERE ROWNUM <= 50;

-- 按关键词过滤
SELECT table_name, comments FROM user_tab_comments
WHERE comments LIKE '%ICT%' OR comments LIKE '%签约%' OR comments LIKE '%集团%';
```

**步骤 2：查看相关表的列注释**
```sql
SELECT column_name, comments FROM user_col_comments WHERE table_name = 'CHATBI_RESULT_1_NEW';
```
发现关键列：`IND_VALUE_1` = ICT签约额，`DIM_VALUE_1` = 集团网格组织机构，`DIM_LEVEL_NAME` = 维层名称，`STAT_MONTH` = 统计月。

**步骤 3：查看维度字典表，建立编码与名称的映射**
```sql
SELECT * FROM ST_DIM_VALUE_DICT_REL_ALL WHERE DIM_VALUE_DESC LIKE '%泰州%';
```
找到泰州市编码 = `1000523`。

**步骤 4：查询目标数据**
```sql
SELECT STAT_MONTH, DIM_VALUE_1, DIM_LEVEL_NAME,
       IND_VALUE_1 AS ICT签约额,
       IND_VALUE_2 AS IT集成服务部分签约额
FROM CHATBI_RESULT_1_NEW
WHERE DIM_VALUE_1 = '泰州市'
ORDER BY STAT_MONTH;
```

**步骤 5：检查数据范围，确认是否有目标日期数据**
```sql
SELECT MIN(STAT_MONTH), MAX(STAT_MONTH) FROM CHATBI_RESULT_1_NEW;
```

**关键经验：**
1. **Oracle 特有查询**：使用 `user_tab_comments` / `user_col_comments` 查看元数据，而非 `information_schema`
2. **列注释可能为空**：即使表注释存在，列注释可能缺失（如 `CHATBI_RESULT_DAY_1`），此时需通过采样数据推断含义
3. **维度表先行**：先查 `ST_DIM_VALUE_DICT_REL_ALL` 这类字典表，理解编码含义后再查事实表
4. **数据范围检查**：查询前先确认 `MIN/MAX` 日期，避免无效查询
5. **无数据时如实告知**：若目标日期无数据或不存在区县级粒度，如实说明而非编造

## 示例：日粒度层级查询 — 地市及下辖区县ICT签约额

**用户问题：** "2025年4月1日泰州市及其下所有区县的ICT签约额"

**场景特点：** 日粒度表 `CHATBI_RESULT_DAY_1`，`DIM_VALUE_1` 直接存储中文名称，`PARENT_DIM_NAME` 指向父级组织，无需关联维度字典表。

**SQL：**
```sql
SELECT STAT_DATE, DIM_VALUE_1, DIM_LEVEL_NAME,
       IND_VALUE_1 AS ICT签约额,
       IND_TB_1 AS 同比增长率,
       IND_HB_1 AS 环比增长率
FROM CHATBI_RESULT_DAY_1
WHERE (DIM_VALUE_1 = '泰州市' OR PARENT_DIM_NAME = '泰州市')
  AND STAT_DATE = 20250401
ORDER BY DIM_LEVEL_NAME, DIM_VALUE_1
```

**查询结果：**

| DIM_VALUE_1 | DIM_LEVEL_NAME | ICT签约额 | 同比增长率 | 环比增长率 |
|------------|---------------|----------|-----------|-----------|
| 泰州市 | 地市级 | 132,334 | 14.25% | 1.38% |
| 高港营销中心 | 区县级 | 16,262 | 15.26% | 1.16% |
| 海陵营销中心 | 区县级 | 19,258 | 13.33% | 2.10% |
| 姜堰分公司 | 区县级 | 26,115 | 12.32% | 0.88% |
| 靖江分公司 | 区县级 | 19,403 | 20.77% | 1.78% |
| 泰兴分公司 | 区县级 | 24,932 | 13.93% | 1.73% |
| 兴化分公司 | 区县级 | 26,364 | 12.07% | 0.85% |

**关键经验：**
1. **层级查询模式**：`DIM_VALUE_1 = '泰州市'` 查本级，`PARENT_DIM_NAME = '泰州市'` 查下级，用 `OR` 合并
2. **DIM_VALUE_1 直接存中文名**：无需关联维度字典表，直接按名称过滤
3. **日粒度日期格式**：`STAT_DATE` 为 NUMBER 类型，格式 `YYYYMMDD`，如 `20250401`
4. **预计算同比/环比**：`IND_TB_1` / `IND_HB_1` 已预计算好百分比值，无需手动计算
5. **空值处理**：部分区县可能无数据（如"泰州未知县分公司"返回 NULL），需如实展示
