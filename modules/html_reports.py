"""
Standalone Interactive HTML Reports Module
Generate self-contained HTML files with interactive Plotly charts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from .config import OUTPUT_DIR


def generate_interactive_html_reports(df):
    """
    Generate standalone interactive HTML reports
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ride booking dataframe
    """
    print("\n" + "="*80)
    print("GENERATING INTERACTIVE HTML REPORTS")
    print("="*80)
    
    # Load RFM data if available
    try:
        rfm_data = pd.read_csv(f"{OUTPUT_DIR}/rfm_ml_result.csv")
        rfm_summary = pd.read_csv(f"{OUTPUT_DIR}/rfm_cluster_summary.csv")
    except:
        rfm_data = None
        rfm_summary = None
    
    # Load revenue forecast if available
    try:
        revenue_forecast = pd.read_csv(f"{OUTPUT_DIR}/revenue_forecast_2025Q1.csv")
    except:
        revenue_forecast = None
    
    completed_rides = df[df['Booking Status'] == 'Completed'].copy()
    
    # Generate individual reports
    generate_overview_report(df, completed_rides)
    generate_customer_analysis_report(df, completed_rides, rfm_data, rfm_summary)
    generate_location_analysis_report(df, completed_rides)
    generate_revenue_analysis_report(df, revenue_forecast)
    generate_comprehensive_report(df, completed_rides, rfm_data, rfm_summary, revenue_forecast)
    
    print("\n✅ All interactive HTML reports generated!")
    print("="*80)


def generate_overview_report(df, completed_rides):
    """Generate overview interactive report"""
    
    # Create subplots figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Booking Status Distribution',
            'Hourly Ride Distribution',
            'Daily Rides Trend',
            'Daily Revenue Trend',
            'Vehicle Type Distribution',
            'Payment Method Distribution'
        ),
        specs=[
            [{'type': 'pie'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'pie'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. Booking Status Distribution (Pie)
    status_counts = df['Booking Status'].value_counts()
    fig.add_trace(
        go.Pie(labels=status_counts.index, values=status_counts.values, 
               hole=0.4, name='Status'),
        row=1, col=1
    )
    
    # 2. Hourly Distribution (Bar)
    hourly_data = df.groupby('Hour').size().reset_index(name='count')
    fig.add_trace(
        go.Bar(x=hourly_data['Hour'], y=hourly_data['count'], 
               marker_color='steelblue', name='Hourly'),
        row=1, col=2
    )
    
    # 3-4. Daily Trend - separate traces for rides and revenue
    df['Date_Only'] = pd.to_datetime(df['Date']).dt.date
    daily_trend = df.groupby('Date_Only').agg({
        'Booking ID': 'count',
        'Booking Value': 'sum'
    }).reset_index()
    daily_trend.columns = ['Date', 'Rides', 'Revenue']
    
    fig.add_trace(
        go.Scatter(x=daily_trend['Date'], y=daily_trend['Rides'],
                  mode='lines', name='Daily Rides', line=dict(color='#2C3E50', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=daily_trend['Date'], y=daily_trend['Revenue'],
                  mode='lines', name='Daily Revenue', line=dict(color='#2ECC71', width=2)),
        row=2, col=2
    )
    
    # 5. Vehicle Type (Bar)
    vehicle_counts = completed_rides['Vehicle Type'].value_counts()
    fig.add_trace(
        go.Bar(y=vehicle_counts.index, x=vehicle_counts.values,
               orientation='h', marker_color='teal', name='Vehicle'),
        row=3, col=1
    )
    
    # 6. Payment Method (Pie)
    payment_counts = df['Payment Method'].value_counts()
    fig.add_trace(
        go.Pie(labels=payment_counts.index, values=payment_counts.values,
               hole=0.4, name='Payment'),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="<b>NCR Ride Analysis - Overview Dashboard</b>",
        title_font_size=24,
        height=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    # Save
    output_file = f"{OUTPUT_DIR}/interactive_overview.html"
    fig.write_html(output_file)
    print(f"✓ Overview report: {output_file}")


def generate_customer_analysis_report(df, completed_rides, rfm_data, rfm_summary):
    """Generate customer analysis interactive report"""
    
    # Calculate retention
    cust_freq = df['Customer ID'].value_counts()
    freq_dist = cust_freq.value_counts().sort_index()
    
    # Create figure with subplots
    if rfm_summary is not None:
        rows, cols = 3, 2
    else:
        rows, cols = 2, 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=(
            'Customer Retention by Purchase Frequency',
            'Rating Distribution',
            'Top 10 Customers by Total Spending',
            'Customer Spending Distribution',
            'RFM Customer Segmentation' if rfm_summary is not None else '',
            'RFM Metrics Comparison' if rfm_summary is not None else ''
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'histogram'}],
            [{'type': 'bar'}, {'type': 'histogram'}],
            [{'type': 'pie'}, {'type': 'bar'}] if rfm_summary is not None else []
        ][:rows],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. Customer Retention
    retention_labels = ['One-time (1)', 'Returning (2)', 'Loyal (3)']
    fig.add_trace(
        go.Bar(x=retention_labels[:len(freq_dist)], y=freq_dist.values,
               marker_color=['#E74C3C', '#F39C12', '#2ECC71'],
               name='Retention'),
        row=1, col=1
    )
    
    # 2. Rating Distribution
    fig.add_trace(
        go.Histogram(x=completed_rides['Customer Rating'].dropna(),
                    name='Customer Rating', marker_color='#3498DB', opacity=0.7),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=completed_rides['Driver Ratings'].dropna(),
                    name='Driver Rating', marker_color='#F39C12', opacity=0.7),
        row=1, col=2
    )
    
    # 3. Top Spenders
    top_spenders = completed_rides.groupby('Customer ID')['Booking Value'].sum().sort_values(ascending=False).head(10)
    fig.add_trace(
        go.Bar(y=[f"Customer {i+1}" for i in range(len(top_spenders))],
               x=top_spenders.values, orientation='h',
               marker_color=px.colors.sequential.Reds[::-1],
               name='Top Spenders'),
        row=2, col=1
    )
    
    # 4. Spending Distribution
    spending_dist = completed_rides.groupby('Customer ID')['Booking Value'].sum()
    fig.add_trace(
        go.Histogram(x=spending_dist, nbinsx=50,
                    marker_color='#1ABC9C', name='Spending'),
        row=2, col=2
    )
    
    # RFM sections if available
    if rfm_summary is not None:
        # 5. RFM Segmentation
        fig.add_trace(
            go.Pie(labels=rfm_summary['Business_Label'], values=rfm_summary['Size'],
                   hole=0.4, name='RFM Segments'),
            row=3, col=1
        )
        
        # 6. RFM Metrics
        metrics = ['Recency', 'Frequency', 'Monetary']
        for i, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(x=rfm_summary['Business_Label'], y=rfm_summary[metric],
                       name=metric, offsetgroup=i),
                row=3, col=2
            )
    
    fig.update_layout(
        title_text="<b>NCR Ride Analysis - Customer Analysis</b>",
        title_font_size=24,
        height=1200 if rfm_summary is not None else 800,
        showlegend=True,
        barmode='overlay',
        template='plotly_white'
    )
    
    output_file = f"{OUTPUT_DIR}/interactive_customer_analysis.html"
    fig.write_html(output_file)
    print(f"✓ Customer analysis report: {output_file}")


def generate_location_analysis_report(df, completed_rides):
    """Generate location and time analysis interactive report"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Top 10 Pickup Locations',
            'Top 10 Drop Locations',
            'Ride Heatmap: Hour vs Day of Week',
            'Distance Distribution'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'heatmap'}, {'type': 'histogram'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Top Pickups
    top_pickups = df['Pickup Location'].value_counts().head(10)
    fig.add_trace(
        go.Bar(y=top_pickups.index, x=top_pickups.values, orientation='h',
               marker_color='lightgreen', name='Pickups'),
        row=1, col=1
    )
    
    # 2. Top Drops
    top_drops = df['Drop Location'].value_counts().head(10)
    fig.add_trace(
        go.Bar(y=top_drops.index, x=top_drops.values, orientation='h',
               marker_color='coral', name='Drops'),
        row=1, col=2
    )
    
    # 3. Heatmap
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values='count').reindex(day_order)
    
    fig.add_trace(
        go.Heatmap(z=heatmap_pivot.values, x=heatmap_pivot.columns, y=heatmap_pivot.index,
                   colorscale='YlOrRd', name='Heatmap'),
        row=2, col=1
    )
    
    # 4. Distance Distribution
    fig.add_trace(
        go.Histogram(x=completed_rides['Ride Distance'].dropna(), nbinsx=50,
                    marker_color='teal', name='Distance'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="<b>NCR Ride Analysis - Location & Time Analysis</b>",
        title_font_size=24,
        height=900,
        showlegend=True,
        template='plotly_white'
    )
    
    output_file = f"{OUTPUT_DIR}/interactive_location_analysis.html"
    fig.write_html(output_file)
    print(f"✓ Location analysis report: {output_file}")


def generate_revenue_analysis_report(df, revenue_forecast):
    """Generate revenue analysis and forecast report"""
    
    # Monthly revenue
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date_dt'].dt.to_period('M').astype(str)
    monthly_revenue = df.groupby('Month').agg({
        'Booking Value': 'sum',
        'Booking ID': 'count'
    }).reset_index()
    monthly_revenue.columns = ['Month', 'Revenue', 'Rides']
    
    # Create figure
    rows = 2 if revenue_forecast is not None else 1
    
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=(
            'Monthly Revenue Trend',
            'Revenue by Vehicle Type',
            'Revenue Forecast - 2025 Q1' if revenue_forecast is not None else '',
            'Revenue by Payment Method' if revenue_forecast is not None else ''
        ),
        specs=[
            [{'type': 'xy'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'bar'}] if revenue_forecast is not None else []
        ][:rows],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Monthly Revenue (bar chart for revenue, line for rides)
    fig.add_trace(
        go.Bar(x=monthly_revenue['Month'], y=monthly_revenue['Revenue'],
               name='Revenue', marker_color='#2ECC71'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=monthly_revenue['Month'], y=monthly_revenue['Rides'],
                  name='Rides', mode='lines+markers', 
                  line=dict(color='#2C3E50', width=3), yaxis='y2'),
        row=1, col=1
    )
    
    # 2. Revenue by Vehicle Type
    vehicle_revenue = df.groupby('Vehicle Type')['Booking Value'].sum().sort_values(ascending=False)
    fig.add_trace(
        go.Bar(x=vehicle_revenue.index, y=vehicle_revenue.values,
               marker_color='teal', name='Vehicle Revenue'),
        row=1, col=2
    )
    
    # Forecast section
    if revenue_forecast is not None:
        # 3. Revenue Forecast
        fig.add_trace(
            go.Scatter(x=monthly_revenue['Month'], y=monthly_revenue['Revenue'],
                      mode='lines+markers', name='Historical',
                      line=dict(color='#2C3E50', width=2)),
            row=2, col=1
        )
        
        if 'Date' in revenue_forecast.columns and 'Ensemble_Forecast' in revenue_forecast.columns:
            revenue_forecast['Month'] = pd.to_datetime(revenue_forecast['Date']).dt.to_period('M').astype(str)
            forecast_monthly = revenue_forecast.groupby('Month')['Ensemble_Forecast'].sum().reset_index()
            
            fig.add_trace(
                go.Scatter(x=forecast_monthly['Month'], y=forecast_monthly['Ensemble_Forecast'],
                          mode='lines+markers', name='Forecast',
                          line=dict(color='#E74C3C', width=2, dash='dash')),
                row=2, col=1
            )
        
        # 4. Revenue by Payment Method
        payment_revenue = df.groupby('Payment Method')['Booking Value'].sum()
        fig.add_trace(
            go.Bar(x=payment_revenue.index, y=payment_revenue.values,
                   marker_color='purple', name='Payment Revenue'),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="<b>NCR Ride Analysis - Revenue Analysis & Forecast</b>",
        title_font_size=24,
        height=900 if revenue_forecast is not None else 500,
        showlegend=True,
        template='plotly_white'
    )
    
    output_file = f"{OUTPUT_DIR}/interactive_revenue_analysis.html"
    fig.write_html(output_file)
    print(f"✓ Revenue analysis report: {output_file}")


def generate_comprehensive_report(df, completed_rides, rfm_data, rfm_summary, revenue_forecast):
    """Generate a comprehensive all-in-one interactive report"""
    
    # Create a complex dashboard with many visualizations
    fig = go.Figure()
    
    # We'll create a comprehensive single-page report with key visualizations
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            'Booking Status', 'Hourly Distribution', 'Vehicle Types',
            'Top Pickup Locations', 'Top Drop Locations', 'Payment Methods',
            'Customer Retention', 'Rating Distribution', 'Distance Distribution',
            'Monthly Revenue Trend', 'RFM Segmentation' if rfm_summary is not None else 'Revenue by Vehicle',
            'Day-Hour Heatmap'
        ),
        specs=[
            [{'type': 'pie'}, {'type': 'bar'}, {'type': 'pie'}],
            [{'type': 'bar'}, {'type': 'bar'}, {'type': 'pie'}],
            [{'type': 'bar'}, {'type': 'histogram'}, {'type': 'histogram'}],
            [{'type': 'scatter'}, {'type': 'pie'}, {'type': 'heatmap'}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.12
    )
    
    # Row 1
    status_counts = df['Booking Status'].value_counts()
    fig.add_trace(go.Pie(labels=status_counts.index, values=status_counts.values, hole=0.3), row=1, col=1)
    
    hourly_data = df.groupby('Hour').size()
    fig.add_trace(go.Bar(x=hourly_data.index, y=hourly_data.values, marker_color='steelblue'), row=1, col=2)
    
    vehicle_counts = completed_rides['Vehicle Type'].value_counts()
    fig.add_trace(go.Pie(labels=vehicle_counts.index, values=vehicle_counts.values, hole=0.3), row=1, col=3)
    
    # Row 2
    top_pickups = df['Pickup Location'].value_counts().head(8)
    fig.add_trace(go.Bar(y=top_pickups.index, x=top_pickups.values, orientation='h', marker_color='lightgreen'), row=2, col=1)
    
    top_drops = df['Drop Location'].value_counts().head(8)
    fig.add_trace(go.Bar(y=top_drops.index, x=top_drops.values, orientation='h', marker_color='coral'), row=2, col=2)
    
    payment_counts = df['Payment Method'].value_counts()
    fig.add_trace(go.Pie(labels=payment_counts.index, values=payment_counts.values, hole=0.3), row=2, col=3)
    
    # Row 3
    cust_freq = df['Customer ID'].value_counts()
    freq_dist = cust_freq.value_counts().sort_index()
    retention_labels = ['1 ride', '2 rides', '3 rides']
    fig.add_trace(go.Bar(x=retention_labels[:len(freq_dist)], y=freq_dist.values, 
                        marker_color=['#E74C3C', '#F39C12', '#2ECC71']), row=3, col=1)
    
    fig.add_trace(go.Histogram(x=completed_rides['Customer Rating'].dropna(), marker_color='#3498DB', opacity=0.7), row=3, col=2)
    
    fig.add_trace(go.Histogram(x=completed_rides['Ride Distance'].dropna(), nbinsx=50, marker_color='teal'), row=3, col=3)
    
    # Row 4
    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M').astype(str)
    monthly_revenue = df.groupby('Month')['Booking Value'].sum()
    fig.add_trace(go.Scatter(x=monthly_revenue.index, y=monthly_revenue.values, 
                            mode='lines+markers', line=dict(color='#2ECC71', width=3)), row=4, col=1)
    
    if rfm_summary is not None:
        fig.add_trace(go.Pie(labels=rfm_summary['Business_Label'], values=rfm_summary['Size'], hole=0.3), row=4, col=2)
    else:
        vehicle_revenue = df.groupby('Vehicle Type')['Booking Value'].sum()
        fig.add_trace(go.Pie(labels=vehicle_revenue.index, values=vehicle_revenue.values, hole=0.3), row=4, col=2)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values='count').reindex(day_order)
    fig.add_trace(go.Heatmap(z=heatmap_pivot.values, x=heatmap_pivot.columns, y=heatmap_pivot.index, 
                            colorscale='YlOrRd'), row=4, col=3)
    
    fig.update_layout(
        title_text="<b>NCR Ride Analysis - Comprehensive Dashboard</b>",
        title_font_size=28,
        height=1600,
        showlegend=False,
        template='plotly_white'
    )
    
    output_file = f"{OUTPUT_DIR}/interactive_comprehensive_dashboard.html"
    fig.write_html(output_file)
    print(f"✓ Comprehensive dashboard: {output_file}")
