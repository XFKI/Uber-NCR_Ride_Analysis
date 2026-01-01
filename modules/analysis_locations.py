"""
Analysis Module 2: Location & Temporal Analysis
Analyzes pickup/drop locations, peak times, and routes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import OUTPUT_DIR


def analyze_locations_enhanced(df):
    """
    Analysis 2: Advanced Location Analysis (Heatmaps, Routes, Peak Times)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ride booking dataframe
    """
    print("\n" + "="*50)
    print("Analysis 2: Advanced Location & Temporal Analysis")
    print("="*50)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # 1. Top 10 Pickup Locations
    ax1 = fig.add_subplot(gs[0, 0])
    top_pickups = df['Pickup Location'].value_counts().head(10)
    sns.barplot(x=top_pickups.values, y=top_pickups.index, ax=ax1, palette='Blues_r')
    ax1.set_title('Top 10 Pickup Locations', fontsize=14, fontweight='bold')

    # 2. Top 10 Drop Locations
    ax2 = fig.add_subplot(gs[0, 1])
    top_drops = df['Drop Location'].value_counts().head(10)
    sns.barplot(x=top_drops.values, y=top_drops.index, ax=ax2, palette='Greens_r')
    ax2.set_title('Top 10 Drop Locations', fontsize=14, fontweight='bold')

    # 3. Peak Time Heatmap (Day of Week vs Hour)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Pivot table for heatmap
    time_pivot = df.pivot_table(index='DayOfWeek', columns='Hour', values='Booking ID', aggfunc='count')
    sns.heatmap(time_pivot, cmap='YlOrRd', ax=ax3, linewidths=.5, annot=False)
    ax3.set_title('Peak Time Heatmap (Day vs Hour)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Hour of Day (0-23)')
    ax3.set_ylabel('Day of Week')

    # 4. Route Matrix Heatmap (Top Pickup vs Top Drop)
    ax4 = fig.add_subplot(gs[1, 1])
    
    top_p_list = top_pickups.index.tolist()
    top_d_list = top_drops.index.tolist()
    
    subset = df[df['Pickup Location'].isin(top_p_list) & df['Drop Location'].isin(top_d_list)]
    
    if not subset.empty:
        route_matrix = pd.crosstab(subset['Pickup Location'], subset['Drop Location'])
        sns.heatmap(route_matrix, cmap='viridis', ax=ax4, annot=True, fmt='d', linewidths=.5, cbar_kws={'shrink': 0.8})
        ax4.set_title('Route Heatmap (Top Locations Interaction)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Drop Location', fontsize=11)
        ax4.set_ylabel('Pickup Location', fontsize=11)
        ax4.tick_params(labelsize=9)
    else:
        ax4.text(0.5, 0.5, 'Not enough data for Top Location intersection', ha='center')

    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/2_locations_temporal_enhanced.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart saved to: {save_path}")
