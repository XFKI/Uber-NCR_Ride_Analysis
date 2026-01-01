"""
NCR Ride Bookings - Advanced Analysis with Machine Learning
Main Entry Point (Modular Version)

This is the main program that orchestrates all analysis modules.
The code has been refactored into separate modules for better maintainability.
"""

from modules import (
    OUTPUT_DIR,
    load_and_clean_data,
    analyze_patterns_and_ratings_enhanced,
    analyze_locations_enhanced,
    analyze_rfm_ml,
    create_comprehensive_dashboard,
    run_revenue_prediction_analysis
)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("NCR RIDE BOOKINGS - ADVANCED ANALYSIS WITH MACHINE LEARNING")
    print("="*80)
    
    # File path
    file_path = 'ncr_ride_bookings.csv'
    
    # Load data
    df = load_and_clean_data(file_path)
    
    if df is not None:
        # Run all analyses
        analyze_patterns_and_ratings_enhanced(df)
        analyze_locations_enhanced(df)
        analyze_rfm_ml(df)
        run_revenue_prediction_analysis(df)
        
        # Print completion summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nAll results saved to '{OUTPUT_DIR}/' directory:")
        print("   - 1_customer_patterns_enhanced.png - Customer behavior analysis")
        print("   - 2_locations_temporal_enhanced.png - Location & time analysis")
        print("   - 3_rfm_elbow_silhouette.png - Optimal cluster selection")
        print("   - 3_rfm_ml_clusters.png - 3D cluster visualization")
        print("   - 3_rfm_pie_charts.png - Customer segment distribution")
        print("   - 3_rfm_bar_comparison.png - RFM metrics comparison")
        print("   - 3_rfm_cluster_profiles.png - Cluster radar charts")
        print("   - 4_revenue_prediction.png - Revenue forecasting analysis")
        print("   - rfm_ml_result.csv - Customer-level clustering results")
        print("   - rfm_cluster_summary.csv - Cluster summary statistics")
        print("   - revenue_forecast_2025Q1.csv - Q1 2025 revenue predictions")
        print("   - revenue_history_and_forecast.csv - Complete revenue timeline")
        print("\nMachine Learning Applied:")
        print("   • K-Means Clustering for RFM Segmentation")
        print("   • ARIMA/Exponential Smoothing for Revenue Forecasting")
        print("="*80 + "\n")
        
        # Generate comprehensive dashboard
        create_comprehensive_dashboard(df)
        
        print("\n" + "="*80)
        print("ALL TASKS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
