"""
Utility Module
Data loading and cleaning functions
"""

import pandas as pd


def load_and_clean_data(filepath):
    """
    Load and clean data, handling ID formats, datetime, and numeric columns.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame or None
        Cleaned dataframe or None if file not found
    """
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

    # 1. Clean ID columns (remove extra quotes like """CID...""")
    id_cols = ['Booking ID', 'Customer ID']
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.strip()

    # 2. Handle Date and Time
    try:
        # Combine Date and Time
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Extract additional time features
        df['Hour'] = df['Datetime'].dt.hour
        df['DayOfWeek'] = df['Datetime'].dt.day_name()
        # Order days for plotting
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=days_order, ordered=True)
        
    except Exception as e:
        print(f"Date conversion warning: {e}")

    # 3. Convert numeric columns
    numeric_cols = ['Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating', 'Avg VTAT', 'Avg CTAT']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Data loading and cleaning complete!")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
