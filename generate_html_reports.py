"""
NCR Ride Bookings - Generate Interactive HTML Reports
Generate standalone HTML files with interactive visualizations
These files can be opened directly in any web browser without running a server
"""

from modules import load_and_clean_data, generate_interactive_html_reports


def main():
    """Generate interactive HTML reports"""
    print("\n" + "="*80)
    print("NCR RIDE BOOKINGS - GENERATE INTERACTIVE HTML REPORTS")
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
        
        # Generate HTML reports
        generate_interactive_html_reports(df)
        
        print("\n" + "="*80)
        print("✅ ALL HTML REPORTS GENERATED SUCCESSFULLY!")
        print("="*80)
        print("\n📁 Reports are saved in the 'analysis_results/' directory:")
        print("   - interactive_overview.html")
        print("   - interactive_customer_analysis.html")
        print("   - interactive_location_analysis.html")
        print("   - interactive_revenue_analysis.html")
        print("   - interactive_comprehensive_dashboard.html")
        print("\n💡 Simply open these HTML files in your web browser to view the interactive charts!")
        print("="*80 + "\n")
    else:
        print("\n❌ Failed to load data. Please check the data file.")


if __name__ == "__main__":
    main()
