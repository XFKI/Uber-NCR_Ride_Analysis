# NCR Ride Bookings Analysis Modules
# This package contains modular components for the ride booking analysis

from .config import OUTPUT_DIR, SEGMENT_CONFIG, get_segment_color, get_segment_strategy
from .utils import load_and_clean_data
from .analysis_patterns import analyze_patterns_and_ratings_enhanced
from .analysis_locations import analyze_locations_enhanced
from .analysis_rfm import analyze_rfm_ml
from .dashboard import create_comprehensive_dashboard
from .analysis_revenue_prediction import run_revenue_prediction_analysis
from .interactive_dashboard import create_interactive_dashboard, launch_dashboard
from .html_reports import generate_interactive_html_reports

__all__ = [
    'OUTPUT_DIR',
    'SEGMENT_CONFIG',
    'get_segment_color',
    'get_segment_strategy',
    'load_and_clean_data',
    'analyze_patterns_and_ratings_enhanced',
    'analyze_locations_enhanced',
    'analyze_rfm_ml',
    'create_comprehensive_dashboard',
    'run_revenue_prediction_analysis',
    'create_interactive_dashboard',
    'launch_dashboard',
    'generate_interactive_html_reports'
]
