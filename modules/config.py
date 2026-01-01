"""
Configuration Module
Global settings, constants, and helper functions for RFM analysis
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Plot Style Configuration
# ==========================================
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# Output Directory
# ==========================================
OUTPUT_DIR = 'analysis_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# Global Color & Strategy Configuration
# Unified across all RFM charts
# ==========================================
SEGMENT_CONFIG = {
    'VIP Champions': {
        'color': '#FFD700',      # Gold
        'strategy': 'VIP Exclusive'
    },
    'Loyal Customers': {
        'color': '#4169E1',      # Royal Blue
        'strategy': 'Loyalty Rewards'
    },
    'Potential Loyalists': {
        'color': '#32CD32',      # Lime Green
        'strategy': 'Engagement Program'
    },
    'New Customers': {
        'color': '#00CED1',      # Dark Turquoise
        'strategy': 'First Order Discount'
    },
    'Hibernating': {
        'color': '#808080',      # Gray
        'strategy': 'Win-back Campaign'
    },
    'At Risk': {
        'color': '#FF6347',      # Tomato
        'strategy': 'Urgent Recall'
    },
    'Lost': {
        'color': '#8B0000',      # Dark Red
        'strategy': 'Last Attempt'
    },
    'High Potential': {
        'color': '#9370DB',      # Medium Purple
        'strategy': 'Premium Offers'
    }
}

def get_segment_color(label):
    """Get consistent color for a segment"""
    return SEGMENT_CONFIG.get(label, {}).get('color', '#888888')

def get_segment_strategy(label):
    """Get marketing strategy for a segment"""
    return SEGMENT_CONFIG.get(label, {}).get('strategy', 'Monitor')
