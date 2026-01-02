# 快速开始指南 / Quick Start Guide

## 🚀 三步开始使用交互式报表

### 步骤 1️⃣: 安装依赖

```bash
pip install -r requirements.txt
```

这将安装所有必需的库：
- plotly (交互式图表)
- dash (Web仪表板)
- pandas, numpy (数据处理)
- 其他分析库

### 步骤 2️⃣: 生成报表

**选项 A: 完整分析 + 交互式报表**
```bash
python main.py
```
- ✅ 执行所有数据分析
- ✅ 生成静态PNG图表
- ✅ 自动生成5个交互式HTML报表
- ⏱️ 需要约2-3分钟

**选项 B: 仅生成交互式HTML报表**
```bash
python generate_html_reports.py
```
- ✅ 快速生成5个HTML报表
- ✅ 无需重新分析数据
- ⏱️ 需要约30秒

### 步骤 3️⃣: 查看报表

**方式 A: 打开HTML文件**
```bash
# 方法1: 双击文件
双击 analysis_results/ 目录中的任意 interactive_*.html 文件

# 方法2: 命令行打开
# Windows:
start analysis_results/interactive_overview.html

# Mac:
open analysis_results/interactive_overview.html

# Linux:
xdg-open analysis_results/interactive_overview.html
```

**方式 B: 启动实时仪表板**
```bash
python run_interactive_dashboard.py

# 然后在浏览器中访问:
# http://127.0.0.1:8050/
```

---

## 📊 5个交互式报表说明

### 1. interactive_overview.html
**总览仪表板 - 了解整体情况**
- 6个KPI指标卡片
- 订单状态、时间分布、车型分析

**适用场景**: 快速了解业务概况、管理层汇报

### 2. interactive_customer_analysis.html
**客户分析 - 深入了解客户**
- 客户留存分析
- 评分对比
- 高价值客户识别
- RFM客户细分

**适用场景**: 客户运营策略、营销活动设计

### 3. interactive_location_analysis.html
**位置与时间分析 - 优化运营布局**
- 热门上车/下车地点
- 时空热力图
- 行程距离分布

**适用场景**: 运力调度、区域扩张决策

### 4. interactive_revenue_analysis.html
**收入分析与预测 - 财务规划**
- 月度收入趋势
- 车型收入贡献
- 2025年Q1预测

**适用场景**: 财务预测、收入分析、目标设定

### 5. interactive_comprehensive_dashboard.html
**综合仪表板 - 全面视图**
- 12个关键图表
- 一页展示所有维度

**适用场景**: 全面汇报、打印报表、团队分享

---

## 🎯 交互功能使用技巧

### 基础操作

1. **查看详细数据**
   ```
   → 将鼠标悬停在任何图表元素上
   → 自动显示该点的详细信息
   ```

2. **缩放查看**
   ```
   → 在图表上拖动鼠标框选区域
   → 图表自动放大到选中区域
   → 双击图表恢复原始视图
   ```

3. **筛选数据**
   ```
   → 点击图例中的项目
   → 隐藏/显示对应的数据系列
   → 方便对比分析
   ```

4. **导出图表**
   ```
   → 将鼠标悬停在图表上
   → 点击右上角的相机图标
   → 图表将保存为PNG图片
   ```

### 高级技巧

1. **时间序列分析**
   ```
   在日期轴上：
   → 框选特定时间段
   → 查看该时段的详细趋势
   → 对比不同时期的数据
   ```

2. **多维度探索**
   ```
   → 同时打开多个报表
   → 在不同维度间切换
   → 发现数据关联性
   ```

3. **组合筛选**
   ```
   → 点击多个图例项
   → 创建自定义数据视图
   → 专注于特定分析目标
   ```

---

## 💡 常见问题解答

### Q1: HTML文件太大，加载慢？
**A**: 
- 这是正常的，文件包含了完整的交互库
- 首次打开需要几秒钟加载
- 后续操作会非常流畅
- 建议使用Chrome或Edge浏览器

### Q2: 如何分享报表给同事？
**A**: 
```
方法1: 发送HTML文件
→ 直接通过邮件发送 interactive_*.html 文件
→ 接收者只需用浏览器打开
→ 无需安装任何软件

方法2: 部署Web仪表板
→ 在服务器上运行 run_interactive_dashboard.py
→ 团队成员通过浏览器访问
→ 实时查看最新数据
```

### Q3: 可以修改图表样式吗？
**A**: 
可以！修改以下文件：
- `modules/interactive_dashboard.py` - Web仪表板样式
- `modules/html_reports.py` - HTML报表样式
- 修改颜色、布局、图表类型等

### Q4: 如何添加新的图表？
**A**: 
1. 在 `modules/html_reports.py` 中添加新的图表生成函数
2. 使用 Plotly Express 或 Graph Objects 创建图表
3. 将图表添加到子图布局中
4. 重新运行 `python generate_html_reports.py`

### Q5: 交互式仪表板无法访问？
**A**: 
检查以下几点：
```bash
# 1. 确认端口8050未被占用
netstat -an | grep 8050

# 2. 检查防火墙设置
# 允许端口8050的访问

# 3. 尝试其他端口
# 修改 run_interactive_dashboard.py 中的端口号
```

---

## 🎨 自定义配置

### 修改颜色主题

编辑 `modules/interactive_dashboard.py`：

```python
# 找到 colors 字典
colors = {
    'primary': '#2C3E50',    # 主色调
    'secondary': '#3498DB',  # 次要色调
    'success': '#2ECC71',    # 成功/收入色
    'warning': '#F39C12',    # 警告/关注色
    'danger': '#E74C3C',     # 危险/问题色
    'info': '#1ABC9C'        # 信息色
}

# 修改为您喜欢的颜色
colors = {
    'primary': '#你的颜色',
    # ... 其他颜色
}
```

### 修改端口号

编辑 `run_interactive_dashboard.py`：

```python
# 找到这行
launch_dashboard(df, port=8050, debug=False)

# 修改为其他端口
launch_dashboard(df, port=8888, debug=False)
```

### 添加公司LOGO

编辑 `modules/interactive_dashboard.py`，在头部添加：

```python
# 在 app.layout 中添加
html.Img(src='您的LOGO路径', height='50px')
```

---

## 📞 获取帮助

遇到问题？查看以下资源：

1. **详细文档**
   - [INTERACTIVE_VISUALIZATION_GUIDE.md](INTERACTIVE_VISUALIZATION_GUIDE.md)
   - [PROJECT_ENHANCEMENT_SUMMARY.md](PROJECT_ENHANCEMENT_SUMMARY.md)

2. **示例说明**
   - [INTERACTIVE_REPORTS_DEMO.md](INTERACTIVE_REPORTS_DEMO.md)

3. **项目README**
   - [README.md](README.md)

4. **GitHub Issues**
   - 在项目仓库创建Issue
   - 描述您遇到的问题
   - 我们会尽快回复

---

## 🎉 开始探索您的数据！

```bash
# 立即开始
python generate_html_reports.py

# 然后打开任意HTML文件
# 享受交互式数据探索的乐趣！
```

**祝您使用愉快！** 📊✨

---

*快速开始指南 v1.0 - 2026年1月*
