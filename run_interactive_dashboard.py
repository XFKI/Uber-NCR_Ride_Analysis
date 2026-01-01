"""
NCR Ride Bookings - Interactive Dashboard Launcher
Launch the interactive web-based dashboard (similar to Power BI/Tableau)
"""

from modules import load_and_clean_data, launch_dashboard


def main():
    """Launch the interactive dashboard"""
    print("\n" + "="*80)
    print("NCR RIDE BOOKINGS - INTERACTIVE DASHBOARD")
    print("="*80)
    print("\nLoading data...")
    
    # File path
    file_path = 'ncr_ride_bookings.csv'
    
    # Load data
    df = load_and_clean_data(file_path)
    
    if df is not None:
        print("\n✅ Data loaded successfully!")
        print(f"📊 Total records: {len(df):,}")
        print(f"👥 Total customers: {df['Customer ID'].nunique():,}")
        
        # Launch interactive dashboard
        launch_dashboard(df, port=8050, debug=False)
    else:
        print("\n❌ Failed to load data. Please check the data file.")


if __name__ == "__main__":
    main()
