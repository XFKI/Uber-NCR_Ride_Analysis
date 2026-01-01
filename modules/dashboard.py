"""
Dashboard Module
Comprehensive dashboard combining all three analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .config import OUTPUT_DIR


def create_comprehensive_dashboard(df):
    """
    Create a comprehensive dashboard combining all three analyses
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ride booking dataframe
    """
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE ANALYSIS DASHBOARD")
    print("="*80)
    
    # Read cluster summary if exists
    try:
        cluster_summary = pd.read_csv(f"{OUTPUT_DIR}/rfm_cluster_summary.csv")
    except:
        print("Warning: Cluster summary not found. Dashboard may be incomplete.")
        return
    
    # Create large dashboard figure
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('NCR Ride Bookings - Comprehensive Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Color mapping for segments (4 actual segments from RFM)
    segment_colors = {
        'High Potential': '#FF8C00',      # 橙色
        'New Customers': '#32CD32',       # 绿色
        'Loyal Customers': '#FFD700',     # 金色
        'Hibernating': '#808080'          # 灰色
    }
    
    # Filter completed rides once for reuse
    completed_rides = df[df['Booking Status'] == 'Completed']
    
    # ========================================
    # PANEL 1: Key Metrics Summary (Top Row)
    # ========================================
    
    # 1.1 - Key Business Metrics
    ax_metrics = fig.add_subplot(3, 4, 1)
    ax_metrics.axis('off')
    
    total_customers = df['Customer ID'].nunique()
    total_rides = len(df)
    total_revenue = df['Booking Value'].sum()
    avg_rating = completed_rides['Customer Rating'].mean()
    avg_distance = completed_rides['Ride Distance'].mean()
    
    metrics_text = (
        f"KEY BUSINESS METRICS\n"
        f"{'='*30}\n\n"
        f"Total Customers: {total_customers:,}\n\n"
        f"Total Rides: {total_rides:,}\n\n"
        f"Total Revenue: Rs.{total_revenue:,.0f}\n\n"
        f"Avg Rating: {avg_rating:.2f} / 5.0\n\n"
        f"Avg Distance: {avg_distance:.1f} km"
    )
    ax_metrics.text(0.1, 0.95, metrics_text,
                    transform=ax_metrics.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax_metrics.set_title('Business Overview', fontsize=14, fontweight='bold', pad=10)
    
    # 1.2 - Customer Segment Distribution (Pie)
    ax_pie = fig.add_subplot(3, 4, 2)
    labels = cluster_summary['Business_Label'].tolist()
    sizes = cluster_summary['Size'].tolist()
    colors = [segment_colors.get(l, '#888888') for l in labels]
    
    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=None, autopct='%1.1f%%',
        startangle=90, colors=colors, 
        explode=[0.02]*len(sizes),
        textprops={'fontsize': 9}
    )
    ax_pie.set_title('Customer Segmentation', fontsize=14, fontweight='bold')
    ax_pie.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # 1.3 - Hourly Ride Distribution
    ax_hourly = fig.add_subplot(3, 4, 3)
    hourly_counts = df.groupby('Hour').size()
    ax_hourly.bar(hourly_counts.index, hourly_counts.values, color='steelblue', edgecolor='black', alpha=0.8)
    ax_hourly.set_xlabel('Hour of Day', fontsize=10)
    ax_hourly.set_ylabel('Number of Rides', fontsize=10)
    ax_hourly.set_title('Hourly Ride Distribution', fontsize=14, fontweight='bold')
    ax_hourly.set_xticks(range(0, 24, 3))
    ax_hourly.grid(axis='y', alpha=0.3)
    
    # 1.4 - Day of Week Distribution
    ax_dow = fig.add_subplot(3, 4, 4)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = df['DayOfWeek'].value_counts().reindex(day_order)
    colors_dow = ['#4169E1'] * 5 + ['#32CD32'] * 2
    ax_dow.bar(range(7), dow_counts.values, color=colors_dow, edgecolor='black', alpha=0.8)
    ax_dow.set_xticks(range(7))
    ax_dow.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=9)
    ax_dow.set_ylabel('Number of Rides', fontsize=10)
    ax_dow.set_title('Weekly Pattern', fontsize=14, fontweight='bold')
    ax_dow.grid(axis='y', alpha=0.3)
    
    # ========================================
    # PANEL 2: Location & Revenue Analysis (Middle Row)
    # ========================================
    
    # 2.1 - Top 10 Pickup Locations
    ax_pickup = fig.add_subplot(3, 4, 5)
    top_pickups = df['Pickup Location'].value_counts().head(10)
    ax_pickup.barh(range(len(top_pickups)), top_pickups.values, color='coral', edgecolor='black')
    ax_pickup.set_yticks(range(len(top_pickups)))
    ax_pickup.set_yticklabels([name[:20]+'...' if len(str(name)) > 20 else name 
                               for name in top_pickups.index], fontsize=8)
    ax_pickup.set_xlabel('Ride Count', fontsize=10)
    ax_pickup.set_title('Top 10 Pickup Locations', fontsize=14, fontweight='bold')
    ax_pickup.invert_yaxis()
    
    # 2.2 - Rating Distribution
    ax_rating = fig.add_subplot(3, 4, 6)
    rating_counts = completed_rides['Customer Rating'].value_counts().sort_index()
    rating_colors = ['#8B0000', '#FF6347', '#FFD700', '#90EE90', '#228B22']
    colors_to_use = [rating_colors[min(int(r)-1, 4)] for r in rating_counts.index]
    ax_rating.bar(rating_counts.index, rating_counts.values, color=colors_to_use, edgecolor='black', alpha=0.8)
    ax_rating.set_xlabel('Rating', fontsize=10)
    ax_rating.set_ylabel('Count', fontsize=10)
    ax_rating.set_title('Customer Rating Distribution', fontsize=14, fontweight='bold')
    ax_rating.grid(axis='y', alpha=0.3)
    
    # 2.3 - Fare Distribution by Vehicle Type
    ax_vehicle = fig.add_subplot(3, 4, 7)
    vehicle_fare = completed_rides.groupby('Vehicle Type')['Booking Value'].mean().sort_values(ascending=False)
    colors_vehicle = plt.cm.viridis(np.linspace(0.2, 0.9, len(vehicle_fare)))
    ax_vehicle.barh(range(len(vehicle_fare)), vehicle_fare.values, color=colors_vehicle, edgecolor='black', alpha=0.8)
    ax_vehicle.set_yticks(range(len(vehicle_fare)))
    ax_vehicle.set_yticklabels(vehicle_fare.index, fontsize=9)
    ax_vehicle.set_xlabel('Average Fare (Rs.)', fontsize=10)
    ax_vehicle.set_title('Avg Fare by Vehicle Type', fontsize=14, fontweight='bold')
    ax_vehicle.invert_yaxis()
    ax_vehicle.grid(axis='x', alpha=0.3)
    
    # 2.4 - Distance Distribution
    ax_dist = fig.add_subplot(3, 4, 8)
    ax_dist.hist(completed_rides['Ride Distance'].dropna(), bins=30, color='teal', edgecolor='black', alpha=0.7)
    avg_dist = completed_rides['Ride Distance'].mean()
    ax_dist.axvline(avg_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_dist:.1f} km')
    ax_dist.set_xlabel('Distance (km)', fontsize=10)
    ax_dist.set_ylabel('Frequency', fontsize=10)
    ax_dist.set_title('Ride Distance Distribution', fontsize=14, fontweight='bold')
    ax_dist.legend(fontsize=9)
    ax_dist.grid(axis='y', alpha=0.3)
    
    # ========================================
    # PANEL 3: RFM Analysis Summary (Bottom Row)
    # ========================================
    
    # 3.1 - RFM Segment Comparison (Bar)
    ax_rfm_bar = fig.add_subplot(3, 4, 9)
    x = range(len(labels))
    width = 0.6
    bars = ax_rfm_bar.bar(x, cluster_summary['Size'], width, color=colors, edgecolor='black', alpha=0.8)
    ax_rfm_bar.set_xticks(x)
    ax_rfm_bar.set_xticklabels([l[:15] for l in labels], rotation=25, ha='right', fontsize=9)
    ax_rfm_bar.set_ylabel('Customer Count', fontsize=10)
    ax_rfm_bar.set_title('Segment Size Comparison', fontsize=14, fontweight='bold')
    ax_rfm_bar.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax_rfm_bar.annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    # 3.2 - Average Recency by Segment
    ax_recency = fig.add_subplot(3, 4, 10)
    ax_recency.bar(x, cluster_summary['Recency'], width, color=colors, edgecolor='black', alpha=0.8)
    ax_recency.set_xticks(x)
    ax_recency.set_xticklabels([l[:15] for l in labels], rotation=25, ha='right', fontsize=9)
    ax_recency.set_ylabel('Days', fontsize=10)
    ax_recency.set_title('Avg Recency (Lower=Better)', fontsize=14, fontweight='bold')
    ax_recency.grid(axis='y', alpha=0.3)
    
    # 3.3 - Average Frequency by Segment
    ax_freq = fig.add_subplot(3, 4, 11)
    ax_freq.bar(x, cluster_summary['Frequency'], width, color=colors, edgecolor='black', alpha=0.8)
    ax_freq.set_xticks(x)
    ax_freq.set_xticklabels([l[:15] for l in labels], rotation=25, ha='right', fontsize=9)
    ax_freq.set_ylabel('Order Count', fontsize=10)
    ax_freq.set_title('Avg Frequency (Higher=Better)', fontsize=14, fontweight='bold')
    ax_freq.grid(axis='y', alpha=0.3)
    
    # 3.4 - Average Monetary by Segment
    ax_monetary = fig.add_subplot(3, 4, 12)
    ax_monetary.bar(x, cluster_summary['Monetary'], width, color=colors, edgecolor='black', alpha=0.8)
    ax_monetary.set_xticks(x)
    ax_monetary.set_xticklabels([l[:15] for l in labels], rotation=25, ha='right', fontsize=9)
    ax_monetary.set_ylabel('Amount (Rs.)', fontsize=10)
    ax_monetary.set_title('Avg Monetary (Higher=Better)', fontsize=14, fontweight='bold')
    ax_monetary.grid(axis='y', alpha=0.3)
    
    # Add timestamp
    fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
             ha='right', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    save_path = f"{OUTPUT_DIR}/comprehensive_dashboard.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[Dashboard] Saved to: {save_path}")
    print("="*80)
