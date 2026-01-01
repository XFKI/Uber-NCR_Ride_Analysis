# 交互式可视化报表使用指南 / Interactive Visualization Guide

[中文](#中文版) | [English](#english-version)

---

## 中文版

### 📊 项目概述

本项目现已增强为包含**现代化交互式可视化报表**，提供类似 Power BI 和 Tableau 的交互体验。交互式报表支持两种形式：

1. **实时Web仪表板** - 基于 Plotly Dash 的动态交互式仪表板
2. **独立HTML报表** - 可直接在浏览器中打开的自包含交互式图表

### 🚀 快速开始

#### 方式一：生成独立 HTML 报表（推荐）

最简单的方式是生成可以直接在浏览器中打开的 HTML 文件：

```bash
# 安装依赖
pip install plotly dash dash-bootstrap-components pandas numpy

# 生成交互式 HTML 报表
python generate_html_reports.py
```

生成的报表文件位于 `analysis_results/` 目录：
- `interactive_overview.html` - 总览仪表板
- `interactive_customer_analysis.html` - 客户分析报表
- `interactive_location_analysis.html` - 位置与时间分析
- `interactive_revenue_analysis.html` - 收入分析与预测
- `interactive_comprehensive_dashboard.html` - 综合仪表板

**使用方法**：直接双击 HTML 文件或在浏览器中打开即可查看交互式图表。

#### 方式二：启动实时 Web 仪表板

运行实时 Web 应用，获得更强大的交互体验：

```bash
# 启动交互式仪表板服务器
python run_interactive_dashboard.py
```

然后在浏览器中访问：`http://127.0.0.1:8050/`

**提示**：按 `Ctrl+C` 停止服务器

#### 方式三：完整分析 + HTML 报表

运行完整分析并自动生成所有报表（包括静态图和交互式报表）：

```bash
python main.py
```

此命令将：
1. 执行所有数据分析
2. 生成静态 PNG 图表
3. 生成交互式 HTML 报表
4. 输出 CSV 数据文件

### 📈 交互式功能特性

#### 1. 总览仪表板 (`interactive_overview.html`)

**核心指标卡片**：
- 总订单数
- 总收入
- 总客户数
- 平均评分
- 完成率
- 平均车费

**交互式图表**：
- 订单状态分布（饼图）- 悬停查看详细数据
- 24小时订单分布（柱状图）- 点击选择时间段
- 每日趋势（折线图）- 缩放、平移查看详细趋势
- 车型分布（柱状图）- 交互式筛选

**交互操作**：
- 悬停显示详细信息
- 点击图例筛选数据
- 缩放查看特定时间段
- 导出图表为PNG

#### 2. 客户分析报表 (`interactive_customer_analysis.html`)

**客户细分**：
- 客户留存分析（1次、2次、3次购买）
- 评分分布对比（客户评分 vs 司机评分）
- Top 10 高消费客户排行
- 消费金额分布

**RFM 分析**（如果已生成）：
- 客户分群饼图
- RFM 指标对比（Recency、Frequency、Monetary）
- 交互式客群筛选

**交互功能**：
- 动态筛选客户群
- 对比不同评分
- 悬停查看客户详情

#### 3. 位置与时间分析 (`interactive_location_analysis.html`)

**位置热力**：
- Top 10 上车地点
- Top 10 下车地点
- 时空热力图（星期 × 小时）

**行程分析**：
- 距离分布直方图
- 支付方式分布

**交互功能**：
- 缩放热力图查看特定时段
- 悬停查看地点订单数
- 交互式距离区间选择

#### 4. 收入分析与预测 (`interactive_revenue_analysis.html`)

**收入趋势**：
- 月度收入和订单趋势（双轴图）
- 各车型收入贡献

**收入预测**（如果已生成）：
- 2025年Q1收入预测
- 历史数据 vs 预测对比
- 支付方式收入分析

**交互功能**：
- 缩放查看特定月份
- 对比历史与预测数据
- 动态切换不同指标

#### 5. 综合仪表板 (`interactive_comprehensive_dashboard.html`)

包含所有核心图表的单页综合视图：
- 12个交互式图表
- 全面覆盖订单、客户、位置、收入维度
- 适合打印或分享的完整报表

### 🎨 实时 Web 仪表板特性

运行 `python run_interactive_dashboard.py` 后的 Web 仪表板提供：

**现代化 UI 设计**：
- Bootstrap 响应式布局
- 专业的配色方案
- Font Awesome 图标

**多标签页导航**：
- 📊 Overview（总览）
- 👥 Customer Analysis（客户分析）
- 📍 Location & Time（位置与时间）
- 💰 Revenue Forecast（收入预测）

**实时交互**：
- 无需刷新页面
- 流畅的图表动画
- 响应式设计适配各种屏幕

**高级功能**：
- 动态过滤器（即将推出）
- 数据导出功能
- 自定义日期范围选择

### 💡 使用技巧

1. **缩放图表**：在任意图表上拖动鼠标框选区域即可放大
2. **重置视图**：双击图表恢复原始视图
3. **隐藏/显示数据系列**：点击图例项
4. **导出图表**：将鼠标悬停在图表上，点击相机图标
5. **查看数值**：将鼠标悬停在数据点上
6. **平移图表**：缩放后可拖动图表查看其他部分

### 🔧 技术栈

- **Plotly** - 强大的交互式图表库
- **Dash** - Python Web 应用框架
- **Dash Bootstrap Components** - 现代化 UI 组件
- **Pandas** - 数据处理
- **NumPy** - 数值计算

### 📦 依赖安装

```bash
pip install plotly dash dash-bootstrap-components pandas numpy matplotlib seaborn scikit-learn statsmodels prophet
```

### 🐛 故障排除

**问题**：无法访问 Web 仪表板
- **解决**：确保端口 8050 未被占用，或在 `run_interactive_dashboard.py` 中修改端口号

**问题**：HTML 文件打开为空白
- **解决**：确保使用现代浏览器（Chrome、Firefox、Edge、Safari）

**问题**：缺少模块错误
- **解决**：运行 `pip install -r requirements.txt`（如果存在）或手动安装依赖

### 📧 支持

如有问题或建议，请在 GitHub 项目中创建 Issue。

---

## English Version

### 📊 Project Overview

This project now features **modern interactive visualization dashboards** providing Power BI and Tableau-like interactive experiences. Interactive reports are available in two forms:

1. **Live Web Dashboard** - Dynamic interactive dashboard based on Plotly Dash
2. **Standalone HTML Reports** - Self-contained interactive charts that can be opened directly in browsers

### 🚀 Quick Start

#### Method 1: Generate Standalone HTML Reports (Recommended)

The easiest way is to generate HTML files that can be opened directly in a browser:

```bash
# Install dependencies
pip install plotly dash dash-bootstrap-components pandas numpy

# Generate interactive HTML reports
python generate_html_reports.py
```

Generated report files are located in the `analysis_results/` directory:
- `interactive_overview.html` - Overview dashboard
- `interactive_customer_analysis.html` - Customer analysis report
- `interactive_location_analysis.html` - Location & time analysis
- `interactive_revenue_analysis.html` - Revenue analysis & forecast
- `interactive_comprehensive_dashboard.html` - Comprehensive dashboard

**Usage**: Simply double-click the HTML file or open it in your browser to view interactive charts.

#### Method 2: Launch Live Web Dashboard

Run the live web application for a more powerful interactive experience:

```bash
# Start the interactive dashboard server
python run_interactive_dashboard.py
```

Then visit in your browser: `http://127.0.0.1:8050/`

**Tip**: Press `Ctrl+C` to stop the server

#### Method 3: Complete Analysis + HTML Reports

Run complete analysis and automatically generate all reports (including static charts and interactive reports):

```bash
python main.py
```

This command will:
1. Execute all data analyses
2. Generate static PNG charts
3. Generate interactive HTML reports
4. Output CSV data files

### 📈 Interactive Features

#### 1. Overview Dashboard (`interactive_overview.html`)

**Key Metric Cards**:
- Total Rides
- Total Revenue
- Total Customers
- Average Rating
- Completion Rate
- Average Fare

**Interactive Charts**:
- Booking status distribution (pie chart) - hover for details
- 24-hour ride distribution (bar chart) - click to select time ranges
- Daily trends (line chart) - zoom and pan for detailed trends
- Vehicle type distribution (bar chart) - interactive filtering

**Interactive Operations**:
- Hover to display detailed information
- Click legend to filter data
- Zoom to view specific time periods
- Export charts as PNG

#### 2. Customer Analysis Report (`interactive_customer_analysis.html`)

**Customer Segmentation**:
- Customer retention analysis (1, 2, 3 rides)
- Rating distribution comparison (customer vs driver ratings)
- Top 10 high-spending customers
- Spending amount distribution

**RFM Analysis** (if generated):
- Customer segment pie chart
- RFM metrics comparison (Recency, Frequency, Monetary)
- Interactive segment filtering

**Interactive Features**:
- Dynamic customer group filtering
- Compare different ratings
- Hover to view customer details

#### 3. Location & Time Analysis (`interactive_location_analysis.html`)

**Location Hotspots**:
- Top 10 pickup locations
- Top 10 drop locations
- Spatiotemporal heatmap (day × hour)

**Trip Analysis**:
- Distance distribution histogram
- Payment method distribution

**Interactive Features**:
- Zoom heatmap to view specific periods
- Hover to view location ride counts
- Interactive distance range selection

#### 4. Revenue Analysis & Forecast (`interactive_revenue_analysis.html`)

**Revenue Trends**:
- Monthly revenue and ride trends (dual-axis chart)
- Revenue contribution by vehicle type

**Revenue Forecast** (if generated):
- 2025 Q1 revenue forecast
- Historical data vs forecast comparison
- Revenue analysis by payment method

**Interactive Features**:
- Zoom to view specific months
- Compare historical and forecast data
- Dynamically switch between metrics

#### 5. Comprehensive Dashboard (`interactive_comprehensive_dashboard.html`)

Single-page comprehensive view containing all core charts:
- 12 interactive charts
- Comprehensive coverage of order, customer, location, revenue dimensions
- Suitable for printing or sharing complete reports

### 🎨 Live Web Dashboard Features

The web dashboard from running `python run_interactive_dashboard.py` provides:

**Modern UI Design**:
- Bootstrap responsive layout
- Professional color scheme
- Font Awesome icons

**Multi-Tab Navigation**:
- 📊 Overview
- 👥 Customer Analysis
- 📍 Location & Time
- 💰 Revenue Forecast

**Real-Time Interaction**:
- No page refresh needed
- Smooth chart animations
- Responsive design for all screen sizes

**Advanced Features**:
- Dynamic filters (coming soon)
- Data export functionality
- Custom date range selection

### 💡 Usage Tips

1. **Zoom Charts**: Drag mouse to select an area on any chart to zoom in
2. **Reset View**: Double-click chart to restore original view
3. **Hide/Show Data Series**: Click legend items
4. **Export Chart**: Hover over chart and click camera icon
5. **View Values**: Hover mouse over data points
6. **Pan Chart**: After zooming, drag chart to view other parts

### 🔧 Tech Stack

- **Plotly** - Powerful interactive charting library
- **Dash** - Python web application framework
- **Dash Bootstrap Components** - Modern UI components
- **Pandas** - Data processing
- **NumPy** - Numerical computing

### 📦 Dependencies Installation

```bash
pip install plotly dash dash-bootstrap-components pandas numpy matplotlib seaborn scikit-learn statsmodels prophet
```

### 🐛 Troubleshooting

**Issue**: Cannot access web dashboard
- **Solution**: Ensure port 8050 is not in use, or modify the port number in `run_interactive_dashboard.py`

**Issue**: HTML file opens as blank
- **Solution**: Ensure using a modern browser (Chrome, Firefox, Edge, Safari)

**Issue**: Missing module error
- **Solution**: Run `pip install -r requirements.txt` (if exists) or manually install dependencies

### 📧 Support

For questions or suggestions, please create an Issue in the GitHub project.
