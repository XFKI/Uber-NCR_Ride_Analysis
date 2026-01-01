"""
Analysis Module 1: Customer Patterns & Ratings
Analyzes customer behavior, ratings, retention, and spending patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .config import OUTPUT_DIR


def analyze_patterns_and_ratings_enhanced(df):
    """
    Analysis 1: Customer Patterns, Ratings, Retention, and Spending Analysis (Enhanced)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ride booking dataframe
    """
    print("\n" + "="*50)
    print("Analysis 1: Customer Patterns & Ratings (Enhanced)")
    print("="*50)

    completed_rides = df[df['Booking Status'] == 'Completed']
    
    # Setup figure with expanded grid layout (4 rows x 2 columns for more insights)
    fig = plt.figure(figsize=(20, 22))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    # ========================================
    # 1. Booking Status Distribution
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    status_counts = df['Booking Status'].value_counts()
    sns.barplot(x=status_counts.values, y=status_counts.index, ax=ax1, palette='viridis')
    ax1.set_title('Booking Status Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Count')
    ax1.set_ylabel('')
    
    # Add value labels
    for i, v in enumerate(status_counts.values):
        ax1.text(v + status_counts.max()*0.01, i, f'{v:,}', 
                va='center', fontsize=10, fontweight='bold')
    
    # ========================================
    # 2. Top 10 Customers by Spending (with Details)
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    top_spenders = completed_rides.groupby('Customer ID')['Booking Value'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_spenders.values, y=top_spenders.index, ax=ax2, palette='magma')
    ax2.set_title('Top 10 Customers by Total Spending', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Booking Value (Rs.)')
    ax2.set_ylabel('Customer ID')
    
    # Add value labels
    for i, v in enumerate(top_spenders.values):
        ax2.text(v + top_spenders.max()*0.01, i, f'Rs.{v:,.0f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # ========================================
    # 3. Customer Retention (Actual Purchase Frequency - 1, 2, 3 times only)
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate actual customer frequency (data shows only 1, 2, 3 times)
    cust_freq = df['Customer ID'].value_counts()
    freq_dist = cust_freq.value_counts().sort_index()
    
    # Create bar chart for actual frequencies (1, 2, 3)
    colors_retention = ['#e74c3c', '#f39c12', '#2ecc71']
    labels = ['One-time\nCustomers\n(1 ride)', 'Returning\nCustomers\n(2 rides)', 'Loyal\nCustomers\n(3 rides)']
    
    bars = ax3.bar(range(len(freq_dist)), freq_dist.values, 
                   color=colors_retention, edgecolor='black', linewidth=1.5, width=0.6)
    ax3.set_xticks(range(len(freq_dist)))
    ax3.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax3.set_ylabel('Number of Customers', fontsize=11, fontweight='bold')
    ax3.set_title('Customer Retention by Purchase Frequency', fontsize=14, fontweight='bold')
    
    # Add count and percentage labels
    total_customers = freq_dist.sum()
    for i, (bar, value) in enumerate(zip(bars, freq_dist.values)):
        height = bar.get_height()
        percentage = (value / total_customers) * 100
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, max(freq_dist.values) * 1.15)
    
    # ========================================
    # 4. Ratings Distribution
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create dual KDE plot
    sns.kdeplot(data=completed_rides.dropna(subset=['Driver Ratings']), 
                x='Driver Ratings', label='Driver Ratings', 
                fill=True, ax=ax4, color='blue', alpha=0.3)
    sns.kdeplot(data=completed_rides.dropna(subset=['Customer Rating']), 
                x='Customer Rating', label='Customer Ratings', 
                fill=True, ax=ax4, color='orange', alpha=0.3)
    
    ax4.set_title('Distribution of Ratings', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Rating Score')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=10)
    
    # Add average lines
    avg_driver = completed_rides['Driver Ratings'].mean()
    avg_customer = completed_rides['Customer Rating'].mean()
    ax4.axvline(avg_driver, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axvline(avg_customer, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.text(avg_driver, ax4.get_ylim()[1]*0.95, f'Avg: {avg_driver:.2f}', 
            color='blue', fontsize=9, ha='center', fontweight='bold')
    ax4.text(avg_customer, ax4.get_ylim()[1]*0.85, f'Avg: {avg_customer:.2f}', 
            color='orange', fontsize=9, ha='center', fontweight='bold')
    
    # ========================================
    # 5. Top 10 Customers - Detailed Analysis Table
    # ========================================
    ax5 = fig.add_subplot(gs[2, :])  # Spans both columns
    ax5.axis('off')
    
    # Calculate detailed metrics for top 10 customers
    top_customer_ids = top_spenders.index.tolist()
    top_customer_data = []
    
    for cid in top_customer_ids:
        cust_data = completed_rides[completed_rides['Customer ID'] == cid]
        total_spend = cust_data['Booking Value'].sum()
        num_rides = len(cust_data)
        avg_spend = cust_data['Booking Value'].mean()
        avg_distance = cust_data['Ride Distance'].mean()
        avg_rating = cust_data['Customer Rating'].mean()
        
        # Determine behavior pattern based on frequency and spending
        if num_rides >= 10:
            pattern = 'VIP (High Frequency)'
        elif num_rides >= 5:
            pattern = 'Loyal (Repeat)'
        elif avg_spend > 1000:
            pattern = 'Premium (High Value)'
        else:
            pattern = 'Regular'
        
        top_customer_data.append({
            'Customer ID': cid[:15] + '...' if len(cid) > 15 else cid,
            'Total Spend (Rs.)': f'{total_spend:,.0f}',
            'Rides': num_rides,
            'Avg/Ride (Rs.)': f'{avg_spend:.0f}',
            'Avg Dist (km)': f'{avg_distance:.1f}',
            'Avg Rating': f'{avg_rating:.2f}',
            'Pattern': pattern
        })
    
    # Create DataFrame for table
    table_df = pd.DataFrame(top_customer_data)
    
    # Create table
    table = ax5.table(cellText=table_df.values,
                     colLabels=table_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(table_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
        cell.set_edgecolor('white')
    
    # Style rows with alternating colors
    for i in range(1, len(table_df) + 1):
        for j in range(len(table_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('#bdc3c7')
            
            # Highlight key metrics
            if j == 1:  # Total Spend
                cell.set_text_props(weight='bold', color='#e74c3c')
            elif j == 2:  # Rides
                cell.set_text_props(weight='bold', color='#3498db')
            elif j == 6:  # Pattern
                cell.set_text_props(weight='bold', color='#27ae60', style='italic')
    
    ax5.set_title('Top 10 Customers - Detailed Behavior Analysis\n(Spending, Frequency, Patterns)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # ========================================
    # 6. Customer Time Preference Analysis (Peak Hours)
    # ========================================
    ax6 = fig.add_subplot(gs[3, 0])
    
    # Analyze customer booking time preferences
    hour_dist = completed_rides['Hour'].value_counts().sort_index()
    
    # Define time segments
    time_colors = []
    for hour in hour_dist.index:
        if 6 <= hour < 10:
            time_colors.append('#e74c3c')  # Morning rush
        elif 10 <= hour < 17:
            time_colors.append('#3498db')  # Daytime
        elif 17 <= hour < 21:
            time_colors.append('#f39c12')  # Evening rush
        else:
            time_colors.append('#95a5a6')  # Night/Early morning
    
    bars = ax6.bar(hour_dist.index, hour_dist.values, color=time_colors, 
                   edgecolor='black', linewidth=0.8, alpha=0.8)
    ax6.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Number of Rides', fontsize=11, fontweight='bold')
    ax6.set_title('Customer Booking Time Preference\n(Hourly Distribution)', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(0, 24, 2))
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend for time segments
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Morning Rush (6-10)'),
        Patch(facecolor='#3498db', label='Daytime (10-17)'),
        Patch(facecolor='#f39c12', label='Evening Rush (17-21)'),
        Patch(facecolor='#95a5a6', label='Night/Early (21-6)')
    ]
    ax6.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    # ========================================
    # 7. Customer Distance Preference & Vehicle Type
    # ========================================
    ax7 = fig.add_subplot(gs[3, 1])
    
    # Analyze ride distance distribution
    distance_bins = [0, 5, 10, 20, 50, float('inf')]
    distance_labels = ['Short\n(0-5km)', 'Medium\n(5-10km)', 'Long\n(10-20km)', 'Very Long\n(20-50km)', 'Ultra\n(50km+)']
    
    completed_rides['Distance_Category'] = pd.cut(completed_rides['Ride Distance'], 
                                                   bins=distance_bins, 
                                                   labels=distance_labels)
    
    distance_dist = completed_rides['Distance_Category'].value_counts().sort_index()
    
    # Create bar chart
    colors_dist = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
    bars = ax7.bar(range(len(distance_dist)), distance_dist.values, 
                   color=colors_dist, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax7.set_xticks(range(len(distance_dist)))
    ax7.set_xticklabels(distance_dist.index, fontsize=10, fontweight='bold')
    ax7.set_ylabel('Number of Rides', fontsize=11, fontweight='bold')
    ax7.set_title('Customer Distance Preference\n(Trip Length Distribution)', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    total_rides = distance_dist.sum()
    for bar, value in zip(bars, distance_dist.values):
        height = bar.get_height()
        percentage = (value / total_rides) * 100
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax7.grid(axis='y', alpha=0.3, linestyle='--')
    ax7.set_ylim(0, max(distance_dist.values) * 1.15)
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/1_customer_patterns_enhanced.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart saved to: {save_path}")
