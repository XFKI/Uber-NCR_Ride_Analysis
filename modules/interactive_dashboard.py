"""
Interactive Dashboard Module
Modern, interactive dashboard using Plotly Dash (similar to Power BI/Tableau)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime
from .config import OUTPUT_DIR


def create_interactive_dashboard(df):
    """
    Create an interactive Dash application for ride analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ride booking dataframe
    """
    
    # Initialize Dash app with Bootstrap theme for modern UI
    app = dash.Dash(__name__, 
                    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                    suppress_callback_exceptions=True)
    
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
    
    # Define color palette
    colors = {
        'primary': '#2C3E50',
        'secondary': '#3498DB',
        'success': '#2ECC71',
        'warning': '#F39C12',
        'danger': '#E74C3C',
        'info': '#1ABC9C'
    }
    
    # Prepare data
    completed_rides = df[df['Booking Status'] == 'Completed'].copy()
    
    # App layout
    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🚗 NCR Ride Analysis - Interactive Dashboard", 
                       className="text-center mb-4 mt-3",
                       style={'color': colors['primary'], 'fontWeight': 'bold'})
            ])
        ]),
        
        # Tabs for different analysis sections
        dbc.Tabs([
            dbc.Tab(label="📊 Overview", tab_id="overview"),
            dbc.Tab(label="👥 Customer Analysis", tab_id="customer"),
            dbc.Tab(label="📍 Location & Time", tab_id="location"),
            dbc.Tab(label="💰 Revenue Forecast", tab_id="revenue"),
        ], id="tabs", active_tab="overview"),
        
        html.Div(id="tab-content", className="mt-4")
    ], fluid=True, style={'backgroundColor': '#F8F9FA'})
    
    # Callback for tab content
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab")
    )
    def render_tab_content(active_tab):
        if active_tab == "overview":
            return create_overview_tab(df, completed_rides, colors)
        elif active_tab == "customer":
            return create_customer_tab(df, completed_rides, rfm_data, rfm_summary, colors)
        elif active_tab == "location":
            return create_location_tab(df, completed_rides, colors)
        elif active_tab == "revenue":
            return create_revenue_tab(df, revenue_forecast, colors)
        return html.Div("Tab not found")
    
    return app


def create_kpi_card(title, value, icon, color, subtitle=""):
    """Create a KPI card component"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=icon, style={'fontSize': '2rem', 'color': color}),
                html.H3(value, className="mt-2 mb-0", style={'color': color, 'fontWeight': 'bold'}),
                html.P(title, className="mb-0 text-muted"),
                html.Small(subtitle, className="text-muted")
            ], className="text-center")
        ])
    ], className="shadow-sm h-100")


def create_overview_tab(df, completed_rides, colors):
    """Create overview tab with KPIs and key metrics"""
    
    # Calculate KPIs
    total_rides = len(df)
    total_revenue = df['Booking Value'].sum()
    total_customers = df['Customer ID'].nunique()
    avg_rating = completed_rides['Customer Rating'].mean()
    completion_rate = (len(completed_rides) / total_rides * 100) if total_rides > 0 else 0
    avg_fare = completed_rides['Booking Value'].mean()
    
    # KPI Cards Row
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card("Total Rides", f"{total_rides:,}", 
                          "fas fa-car", colors['primary'])
        ], md=2),
        dbc.Col([
            create_kpi_card("Total Revenue", f"Rs. {total_revenue:,.0f}", 
                          "fas fa-rupee-sign", colors['success'])
        ], md=2),
        dbc.Col([
            create_kpi_card("Total Customers", f"{total_customers:,}", 
                          "fas fa-users", colors['info'])
        ], md=2),
        dbc.Col([
            create_kpi_card("Avg Rating", f"{avg_rating:.2f}/5.0", 
                          "fas fa-star", colors['warning'])
        ], md=2),
        dbc.Col([
            create_kpi_card("Completion Rate", f"{completion_rate:.1f}%", 
                          "fas fa-check-circle", colors['success'])
        ], md=2),
        dbc.Col([
            create_kpi_card("Avg Fare", f"Rs. {avg_fare:.0f}", 
                          "fas fa-money-bill-wave", colors['secondary'])
        ], md=2),
    ], className="mb-4")
    
    # Booking Status Distribution
    status_counts = df['Booking Status'].value_counts()
    status_fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Booking Status Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    status_fig.update_layout(height=400)
    
    # Hourly Distribution
    hourly_data = df.groupby('Hour').size().reset_index(name='count')
    hourly_fig = px.bar(
        hourly_data, x='Hour', y='count',
        title="Hourly Ride Distribution",
        labels={'count': 'Number of Rides', 'Hour': 'Hour of Day'},
        color='count',
        color_continuous_scale='Viridis'
    )
    hourly_fig.update_layout(height=400)
    
    # Daily Trend
    df['Date_Only'] = pd.to_datetime(df['Date']).dt.date
    daily_trend = df.groupby('Date_Only').agg({
        'Booking ID': 'count',
        'Booking Value': 'sum'
    }).reset_index()
    daily_trend.columns = ['Date', 'Rides', 'Revenue']
    
    daily_fig = make_subplots(specs=[[{"secondary_y": True}]])
    daily_fig.add_trace(
        go.Scatter(x=daily_trend['Date'], y=daily_trend['Rides'], 
                  name="Rides", line=dict(color=colors['primary'], width=2)),
        secondary_y=False
    )
    daily_fig.add_trace(
        go.Scatter(x=daily_trend['Date'], y=daily_trend['Revenue'], 
                  name="Revenue", line=dict(color=colors['success'], width=2)),
        secondary_y=True
    )
    daily_fig.update_xaxes(title_text="Date")
    daily_fig.update_yaxes(title_text="Number of Rides", secondary_y=False)
    daily_fig.update_yaxes(title_text="Revenue (Rs.)", secondary_y=True)
    daily_fig.update_layout(title="Daily Rides and Revenue Trend", height=400)
    
    # Vehicle Type Distribution
    vehicle_counts = completed_rides['Vehicle Type'].value_counts()
    vehicle_fig = px.bar(
        x=vehicle_counts.values,
        y=vehicle_counts.index,
        orientation='h',
        title="Vehicle Type Distribution",
        labels={'x': 'Number of Rides', 'y': 'Vehicle Type'},
        color=vehicle_counts.values,
        color_continuous_scale='Blues'
    )
    vehicle_fig.update_layout(height=400, showlegend=False)
    
    return html.Div([
        kpi_row,
        dbc.Row([
            dbc.Col([dcc.Graph(figure=status_fig)], md=6),
            dbc.Col([dcc.Graph(figure=hourly_fig)], md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=daily_fig)], md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=vehicle_fig)], md=12),
        ], className="mb-4"),
    ])


def create_customer_tab(df, completed_rides, rfm_data, rfm_summary, colors):
    """Create customer analysis tab with RFM segmentation"""
    
    # Customer retention analysis
    cust_freq = df['Customer ID'].value_counts()
    freq_dist = cust_freq.value_counts().sort_index()
    
    retention_labels = ['One-time (1 ride)', 'Returning (2 rides)', 'Loyal (3 rides)']
    retention_fig = px.bar(
        x=retention_labels[:len(freq_dist)],
        y=freq_dist.values,
        title="Customer Retention by Purchase Frequency",
        labels={'x': 'Customer Type', 'y': 'Number of Customers'},
        color=freq_dist.values,
        color_continuous_scale='RdYlGn'
    )
    retention_fig.update_layout(height=400, showlegend=False)
    
    # Rating distribution
    rating_fig = go.Figure()
    rating_fig.add_trace(go.Histogram(
        x=completed_rides['Customer Rating'].dropna(),
        name='Customer Rating',
        marker_color=colors['secondary'],
        opacity=0.7,
        nbinsx=20
    ))
    rating_fig.add_trace(go.Histogram(
        x=completed_rides['Driver Ratings'].dropna(),
        name='Driver Rating',
        marker_color=colors['warning'],
        opacity=0.7,
        nbinsx=20
    ))
    rating_fig.update_layout(
        title="Rating Distribution (Customer vs Driver)",
        xaxis_title="Rating",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    # Top spenders
    top_spenders = completed_rides.groupby('Customer ID')['Booking Value'].sum().sort_values(ascending=False).head(10)
    spenders_fig = px.bar(
        x=top_spenders.values,
        y=[f"Customer {i+1}" for i in range(len(top_spenders))],
        orientation='h',
        title="Top 10 Customers by Total Spending",
        labels={'x': 'Total Spending (Rs.)', 'y': 'Customer'},
        color=top_spenders.values,
        color_continuous_scale='Reds'
    )
    spenders_fig.update_layout(height=400, showlegend=False)
    
    # RFM Analysis if available
    rfm_section = []
    if rfm_summary is not None:
        # RFM Segment Distribution
        rfm_pie = px.pie(
            rfm_summary,
            values='Size',
            names='Business_Label',
            title="RFM Customer Segmentation",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        rfm_pie.update_layout(height=400)
        
        # RFM Metrics Comparison
        rfm_bar = go.Figure()
        rfm_bar.add_trace(go.Bar(
            x=rfm_summary['Business_Label'],
            y=rfm_summary['Recency'],
            name='Recency (days)',
            marker_color=colors['danger']
        ))
        rfm_bar.add_trace(go.Bar(
            x=rfm_summary['Business_Label'],
            y=rfm_summary['Frequency'],
            name='Frequency (orders)',
            marker_color=colors['success']
        ))
        rfm_bar.add_trace(go.Bar(
            x=rfm_summary['Business_Label'],
            y=rfm_summary['Monetary'],
            name='Monetary (Rs.)',
            marker_color=colors['warning']
        ))
        rfm_bar.update_layout(
            title="RFM Metrics by Customer Segment",
            barmode='group',
            height=400,
            xaxis_title="Customer Segment",
            yaxis_title="Value"
        )
        
        rfm_section = [
            dbc.Row([
                dbc.Col([html.H4("RFM Customer Segmentation", className="mt-4 mb-3")], md=12)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=rfm_pie)], md=6),
                dbc.Col([dcc.Graph(figure=rfm_bar)], md=6),
            ], className="mb-4")
        ]
    
    return html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=retention_fig)], md=6),
            dbc.Col([dcc.Graph(figure=rating_fig)], md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=spenders_fig)], md=12),
        ], className="mb-4"),
        *rfm_section
    ])


def create_location_tab(df, completed_rides, colors):
    """Create location and time analysis tab"""
    
    # Top pickup locations
    top_pickups = df['Pickup Location'].value_counts().head(10)
    pickup_fig = px.bar(
        x=top_pickups.values,
        y=top_pickups.index,
        orientation='h',
        title="Top 10 Pickup Locations",
        labels={'x': 'Number of Rides', 'y': 'Location'},
        color=top_pickups.values,
        color_continuous_scale='Greens'
    )
    pickup_fig.update_layout(height=500, showlegend=False)
    
    # Top drop locations
    top_drops = df['Drop Location'].value_counts().head(10)
    drop_fig = px.bar(
        x=top_drops.values,
        y=top_drops.index,
        orientation='h',
        title="Top 10 Drop Locations",
        labels={'x': 'Number of Rides', 'y': 'Location'},
        color=top_drops.values,
        color_continuous_scale='Oranges'
    )
    drop_fig.update_layout(height=500, showlegend=False)
    
    # Heatmap: Hour vs Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values='count').reindex(day_order)
    
    heatmap_fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Rides"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale="YlOrRd",
        title="Ride Heatmap: Day of Week vs Hour of Day"
    )
    heatmap_fig.update_layout(height=400)
    
    # Distance distribution
    distance_fig = px.histogram(
        completed_rides,
        x='Ride Distance',
        nbins=50,
        title="Ride Distance Distribution",
        labels={'Ride Distance': 'Distance (km)', 'count': 'Frequency'},
        color_discrete_sequence=[colors['info']]
    )
    distance_fig.update_layout(height=400)
    
    # Payment method distribution
    payment_counts = df['Payment Method'].value_counts()
    payment_fig = px.pie(
        values=payment_counts.values,
        names=payment_counts.index,
        title="Payment Method Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    payment_fig.update_layout(height=400)
    
    return html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=pickup_fig)], md=6),
            dbc.Col([dcc.Graph(figure=drop_fig)], md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=heatmap_fig)], md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=distance_fig)], md=6),
            dbc.Col([dcc.Graph(figure=payment_fig)], md=6),
        ], className="mb-4"),
    ])


def create_revenue_tab(df, revenue_forecast, colors):
    """Create revenue analysis and forecast tab"""
    
    # Monthly revenue trend
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date_dt'].dt.to_period('M').astype(str)
    monthly_revenue = df.groupby('Month').agg({
        'Booking Value': 'sum',
        'Booking ID': 'count'
    }).reset_index()
    monthly_revenue.columns = ['Month', 'Revenue', 'Rides']
    
    monthly_fig = make_subplots(specs=[[{"secondary_y": True}]])
    monthly_fig.add_trace(
        go.Bar(x=monthly_revenue['Month'], y=monthly_revenue['Revenue'], 
              name="Revenue", marker_color=colors['success']),
        secondary_y=False
    )
    monthly_fig.add_trace(
        go.Scatter(x=monthly_revenue['Month'], y=monthly_revenue['Rides'], 
                  name="Rides", mode='lines+markers', 
                  line=dict(color=colors['primary'], width=3)),
        secondary_y=True
    )
    monthly_fig.update_xaxes(title_text="Month")
    monthly_fig.update_yaxes(title_text="Revenue (Rs.)", secondary_y=False)
    monthly_fig.update_yaxes(title_text="Number of Rides", secondary_y=True)
    monthly_fig.update_layout(title="Monthly Revenue and Rides Trend", height=500)
    
    # Revenue by vehicle type
    vehicle_revenue = df.groupby('Vehicle Type')['Booking Value'].sum().sort_values(ascending=False)
    vehicle_rev_fig = px.bar(
        x=vehicle_revenue.index,
        y=vehicle_revenue.values,
        title="Total Revenue by Vehicle Type",
        labels={'x': 'Vehicle Type', 'y': 'Total Revenue (Rs.)'},
        color=vehicle_revenue.values,
        color_continuous_scale='Tealgrn'
    )
    vehicle_rev_fig.update_layout(height=400, showlegend=False)
    
    # Revenue forecast section
    forecast_section = []
    if revenue_forecast is not None:
        forecast_fig = go.Figure()
        
        # Historical data
        forecast_fig.add_trace(go.Scatter(
            x=monthly_revenue['Month'],
            y=monthly_revenue['Revenue'],
            mode='lines+markers',
            name='Historical Revenue',
            line=dict(color=colors['primary'], width=2)
        ))
        
        # Add forecast if columns exist
        if 'Date' in revenue_forecast.columns and 'Ensemble_Forecast' in revenue_forecast.columns:
            revenue_forecast['Month'] = pd.to_datetime(revenue_forecast['Date']).dt.to_period('M').astype(str)
            forecast_monthly = revenue_forecast.groupby('Month')['Ensemble_Forecast'].sum().reset_index()
            
            forecast_fig.add_trace(go.Scatter(
                x=forecast_monthly['Month'],
                y=forecast_monthly['Ensemble_Forecast'],
                mode='lines+markers',
                name='Forecast (2025 Q1)',
                line=dict(color=colors['danger'], width=2, dash='dash')
            ))
        
        forecast_fig.update_layout(
            title="Revenue Forecast - 2025 Q1",
            xaxis_title="Month",
            yaxis_title="Revenue (Rs.)",
            height=500
        )
        
        forecast_section = [
            dbc.Row([
                dbc.Col([html.H4("Revenue Forecast", className="mt-4 mb-3")], md=12)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=forecast_fig)], md=12),
            ], className="mb-4")
        ]
    
    return html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=monthly_fig)], md=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=vehicle_rev_fig)], md=12),
        ], className="mb-4"),
        *forecast_section
    ])


def launch_dashboard(df, port=8050, debug=False):
    """
    Launch the interactive dashboard
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ride booking dataframe
    port : int
        Port number for the dashboard
    debug : bool
        Enable debug mode
    """
    app = create_interactive_dashboard(df)
    print("\n" + "="*80)
    print("🚀 LAUNCHING INTERACTIVE DASHBOARD")
    print("="*80)
    print(f"\n📊 Dashboard URL: http://127.0.0.1:{port}/")
    print("\n💡 Press CTRL+C to stop the server")
    print("="*80 + "\n")
    
    app.run_server(debug=debug, port=port, host='0.0.0.0')
