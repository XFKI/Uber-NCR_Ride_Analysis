"""
RFM Visualization Module
All visualization functions for RFM clustering results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .config import OUTPUT_DIR, get_segment_color, get_segment_strategy


def visualize_3d_clusters(rfm, cluster_summary):
    """
    Create 3D scatter plot of RFM clusters
    
    Parameters:
    -----------
    rfm : pd.DataFrame
        RFM data with cluster assignments
    cluster_summary : pd.DataFrame
        Cluster summary statistics
    """
    print("\n" + "="*50)
    print("Creating 3D Cluster Visualization...")
    print("="*50)
    
    # For large datasets, sample for visualization
    n_customers = len(rfm)
    if n_customers > 5000:
        print(f"   Sampling {min(5000, n_customers)} customers for visualization...")
        rfm_viz = rfm.sample(n=min(5000, n_customers), random_state=42)
    else:
        rfm_viz = rfm
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each cluster
    unique_labels = rfm_viz['Business_Label'].unique()
    
    for label in unique_labels:
        cluster_data = rfm_viz[rfm_viz['Business_Label'] == label]
        color = get_segment_color(label)
        strategy = get_segment_strategy(label)
        ax.scatter(
            cluster_data['Recency'],
            cluster_data['Frequency'],
            cluster_data['Monetary'],
            c=color,
            label=f"{label} [{strategy}]",
            s=40,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.3
        )
    
    ax.set_xlabel('\n\nRecency (Days since last order)', fontsize=11, labelpad=15)
    ax.set_ylabel('\n\nFrequency (Order count)', fontsize=11, labelpad=15)
    ax.set_zlabel('\n\nMonetary (Total spend Rs.)', fontsize=11, labelpad=15)
    ax.set_title('RFM Customer Segmentation - 3D Visualization\n(K-Means Clustering with Marketing Strategies)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.view_init(elev=25, azim=135)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, 
              title='Segments & Strategies', title_fontsize=10, framealpha=0.9)
    
    # Add insights
    total_customers = len(rfm)
    loyal_pct = cluster_summary[cluster_summary['Business_Label'].str.contains('Loyal|VIP|Champions', na=False)]['Size'].sum() / total_customers * 100
    
    insight_text = (f"Analysis Summary:\n"
                    f"- Total Customers: {total_customers:,}\n"
                    f"- Clusters: {len(unique_labels)}\n"
                    f"- High-Value: {loyal_pct:.1f}%\n\n"
                    f"Axis Guide:\n"
                    f"- X: Lower = Recent\n"
                    f"- Y: Higher = Frequent\n"
                    f"- Z: Higher = Valuable")
    
    fig.text(0.02, 0.02, insight_text, fontsize=9, 
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.72, bottom=0.12)
    
    save_path = f"{OUTPUT_DIR}/3_rfm_ml_clusters.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart saved to: {save_path}")


def create_pie_chart(cluster_summary):
    """Create pie charts for segment distribution"""
    print("\n" + "="*50)
    print("Creating Customer Segment Pie Chart...")
    print("="*50)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    labels = cluster_summary['Business_Label'].tolist()
    sizes = cluster_summary['Size'].tolist()
    colors = [get_segment_color(label) for label in labels]
    total = sum(sizes)
    
    legend_labels = [f"{label}\n[{get_segment_strategy(label)}]" for label in labels]
    
    # Pie 1: Count
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=None, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=[0.03]*len(sizes), shadow=True,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax1.set_title('Customer Segment Distribution\n(by Count)', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(wedges, legend_labels, title="Segments & Strategy", loc="center left", 
               bbox_to_anchor=(1.05, 0.5), fontsize=9, title_fontsize=10)
    
    # Pie 2: Monetary
    ax2 = axes[1]
    monetary_values = cluster_summary['Monetary'].tolist()
    total_monetary = [s * m for s, m in zip(sizes, monetary_values)]
    total_m = sum(total_monetary)
    
    wedges2, texts2, autotexts2 = ax2.pie(
        total_monetary, labels=None, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=[0.03]*len(sizes), shadow=True,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    ax2.set_title('Revenue Contribution by Segment\n(Total Monetary Value)', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(wedges2, legend_labels, title="Segments & Strategy", loc="center left", 
               bbox_to_anchor=(1.05, 0.5), fontsize=9, title_fontsize=10)
    
    fig.text(0.5, 0.02, f"Total Customers: {total:,} | Total Revenue: Rs.{total_m:,.0f}", 
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, wspace=0.4)
    
    save_path = f"{OUTPUT_DIR}/3_rfm_pie_charts.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart saved to: {save_path}")


def create_bar_comparison_chart(cluster_summary):
    """Create bar chart comparing RFM metrics"""
    print("\n" + "="*50)
    print("Creating RFM Metrics Comparison Bar Chart...")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    
    labels = cluster_summary['Business_Label'].tolist()
    colors = [get_segment_color(label) for label in labels]
    x_labels = [f"{l}\n[{get_segment_strategy(l)}]" for l in labels]
    x_pos = range(len(labels))
    
    # 1. Customer Count
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, cluster_summary['Size'], color=colors, edgecolor='black', width=0.7)
    ax1.set_title('Customer Count by Segment', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Customers', fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=8, ha='center')
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Average Recency
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, cluster_summary['Recency'], color=colors, edgecolor='black', width=0.7)
    ax2.set_title('Average Recency by Segment\n(Days since last order - Lower is better)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Days', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=8, ha='center')
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}d', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Average Frequency
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, cluster_summary['Frequency'], color=colors, edgecolor='black', width=0.7)
    ax3.set_title('Average Order Frequency by Segment\n(Orders per customer)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Order Count', fontsize=11)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, fontsize=8, ha='center')
    for bar in bars3:
        height = bar.get_height()
        ax3.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Average Monetary
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, cluster_summary['Monetary'], color=colors, edgecolor='black', width=0.7)
    ax4.set_title('Average Monetary Value by Segment\n(Total spend per customer)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Amount (Rs.)', fontsize=11)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x_labels, fontsize=8, ha='center')
    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(f'Rs.{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fig.suptitle('RFM Metrics Comparison Across Customer Segments\n(with Recommended Strategies)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/3_rfm_bar_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart saved to: {save_path}")


def create_radar_chart(cluster_summary):
    """Create radar chart with average baseline"""
    print("\n" + "="*50)
    print("Creating Cluster Comparison Radar Chart...")
    print("="*50)
    
    categories = ['Recency\n(Days)', 'Frequency\n(Orders)', 'Monetary\n(Rs.)']
    N = len(categories)
    
    recency_values = cluster_summary['Recency'].values
    frequency_values = cluster_summary['Frequency'].values
    monetary_values = cluster_summary['Monetary'].values
    
    # Improved normalization
    max_recency = 365
    r_scores = np.clip(1 - (recency_values / max_recency), 0.05, 1.0)
    f_scores = np.clip(frequency_values / frequency_values.max(), 0.05, 1.0)
    m_scores = np.clip(monetary_values / monetary_values.max(), 0.05, 1.0)
    
    cluster_summary_plot = cluster_summary.copy()
    cluster_summary_plot['R_Plot'] = r_scores
    cluster_summary_plot['F_Plot'] = f_scores
    cluster_summary_plot['M_Plot'] = m_scores
    
    # Calculate weighted average
    total_customers = cluster_summary_plot['Size'].sum()
    avg_r = (cluster_summary_plot['R_Plot'] * cluster_summary_plot['Size']).sum() / total_customers
    avg_f = (cluster_summary_plot['F_Plot'] * cluster_summary_plot['Size']).sum() / total_customers
    avg_m = (cluster_summary_plot['M_Plot'] * cluster_summary_plot['Size']).sum() / total_customers
    avg_values = [avg_r, avg_f, avg_m]
    avg_values_closed = avg_values + avg_values[:1]
    
    avg_recency_actual = (cluster_summary['Recency'] * cluster_summary['Size']).sum() / total_customers
    avg_frequency_actual = (cluster_summary['Frequency'] * cluster_summary['Size']).sum() / total_customers
    avg_monetary_actual = (cluster_summary['Monetary'] * cluster_summary['Size']).sum() / total_customers
    
    # Angles for radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles_closed = angles + angles[:1]
    
    # Create figure
    num_clusters = len(cluster_summary_plot)
    fig, axes = plt.subplots(1, num_clusters + 1, figsize=(5*(num_clusters + 1), 7), 
                             subplot_kw=dict(projection='polar'))
    
    # Average baseline subplot
    ax_avg = axes[0]
    ax_avg.plot(angles_closed, avg_values_closed, 'o-', linewidth=3, color='#2F4F4F', label='All Customers Average')
    ax_avg.fill(angles_closed, avg_values_closed, alpha=0.3, color='#2F4F4F')
    ax_avg.set_xticks(angles)
    ax_avg.set_xticklabels(categories, size=9)
    ax_avg.set_ylim(0, 1.1)
    ax_avg.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_avg.set_yticklabels(['25%', '50%', '75%', '100%'], size=7)
    ax_avg.grid(True, alpha=0.4)
    ax_avg.set_title(f"AVERAGE BASELINE\n({total_customers:,} customers)\n[Reference]", 
                     size=11, fontweight='bold', pad=20, color='#2F4F4F')
    
    textstr = f"R: {avg_recency_actual:.0f} days\nF: {avg_frequency_actual:.2f} orders\nM: Rs.{avg_monetary_actual:.0f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#2F4F4F')
    ax_avg.text(0.5, -0.15, textstr, transform=ax_avg.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='center', bbox=props)
    
    # Cluster subplots
    for idx, (_, row) in enumerate(cluster_summary_plot.iterrows()):
        ax = axes[idx + 1]
        
        values = [row['R_Plot'], row['F_Plot'], row['M_Plot']]
        values_closed = values + values[:1]
        
        segment_color = get_segment_color(row['Business_Label'])
        strategy = get_segment_strategy(row['Business_Label'])
        
        ax.plot(angles_closed, avg_values_closed, '--', linewidth=1.5, color='#2F4F4F', 
                alpha=0.6, label='Average')
        ax.fill(angles_closed, avg_values_closed, alpha=0.1, color='#2F4F4F')
        
        ax.plot(angles_closed, values_closed, 'o-', linewidth=2.5, color=segment_color, 
                label=row['Business_Label'])
        ax.fill(angles_closed, values_closed, alpha=0.35, color=segment_color)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=7)
        ax.grid(True, alpha=0.4)
        
        ax.set_title(f"{row['Business_Label']}\n({int(row['Size']):,} customers)\n[{strategy}]", 
                     size=10, fontweight='bold', pad=20, color=segment_color)
        
        textstr = f"R: {row['Recency']:.0f} days\nF: {row['Frequency']:.2f} orders\nM: Rs.{row['Monetary']:.0f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=segment_color)
        ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='center', bbox=props)
        
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    
    fig.suptitle('RFM Profile Comparison - Radar Chart with Average Baseline\n(Dashed line = Overall Average | Solid = Segment | Lower Recency = Better)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/3_rfm_cluster_profiles.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart saved to: {save_path}")
