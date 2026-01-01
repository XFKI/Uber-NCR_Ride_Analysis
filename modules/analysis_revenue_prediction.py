"""
Revenue Prediction Module with Advanced Model Comparison
Using multiple time series models with comprehensive evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .config import OUTPUT_DIR


def calculate_daily_revenue(df):
    """
    Calculate daily revenue for 2024
    
    Parameters:
    -----------
    df : DataFrame
        Order data
        
    Returns:
    --------
    DataFrame
        Daily revenue summary
    """
    # Only count completed orders
    completed = df[df['Booking Status'] == 'Completed'].copy()
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(completed['Date']):
        completed['Date'] = pd.to_datetime(completed['Date'])
    
    # Extract date only (no time)
    completed['DateOnly'] = completed['Date'].dt.date
    
    # Aggregate by day
    daily_revenue = completed.groupby('DateOnly').agg({
        'Booking Value': 'sum',
        'Booking ID': 'count'
    }).reset_index()
    
    daily_revenue.columns = ['Date', 'Revenue', 'Order_Count']
    
    # Convert to datetime
    daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
    
    # Fill missing dates with 0 revenue
    date_range = pd.date_range(start=daily_revenue['Date'].min(), 
                               end=daily_revenue['Date'].max(), 
                               freq='D')
    daily_revenue = daily_revenue.set_index('Date').reindex(date_range, fill_value=0).reset_index()
    daily_revenue.columns = ['Date', 'Revenue', 'Order_Count']
    
    return daily_revenue


def calculate_monthly_revenue(df, from_daily=None):
    """
    Calculate monthly revenue for 2024
    Can aggregate from daily data if provided
    
    Parameters:
    -----------
    df : DataFrame
        Order data
    from_daily : DataFrame, optional
        Daily revenue data to aggregate from
        
    Returns:
    --------
    DataFrame
        Monthly revenue summary
    """
    # If daily data provided, aggregate from it
    if from_daily is not None:
        daily = from_daily.copy()
        daily['YearMonth'] = daily['Date'].dt.to_period('M')
        
        monthly_revenue = daily.groupby('YearMonth').agg({
            'Revenue': 'sum',
            'Order_Count': 'sum'
        }).reset_index()
        
        monthly_revenue['Date'] = monthly_revenue['YearMonth'].dt.to_timestamp()
        return monthly_revenue
    
    # Otherwise calculate from raw order data
    # Only count completed orders
    completed = df[df['Booking Status'] == 'Completed'].copy()
    
    # Ensure Datetime field exists
    if 'Datetime' not in completed.columns:
        # Handle both datetime and string Date columns
        if pd.api.types.is_datetime64_any_dtype(completed['Date']):
            # Date is already datetime, just use it
            completed['Datetime'] = completed['Date']
        else:
            # Date is string, combine with Time
            completed['Datetime'] = pd.to_datetime(completed['Date'].astype(str) + ' ' + completed['Time'].astype(str))
    
    # Extract year-month
    completed['YearMonth'] = completed['Datetime'].dt.to_period('M')
    
    # Aggregate by month
    monthly_revenue = completed.groupby('YearMonth').agg({
        'Booking Value': 'sum',
        'Booking ID': 'count'
    }).reset_index()
    
    monthly_revenue.columns = ['YearMonth', 'Revenue', 'Order_Count']
    
    # Convert to datetime for plotting
    monthly_revenue['Date'] = monthly_revenue['YearMonth'].dt.to_timestamp()
    
    return monthly_revenue


def calculate_metrics(actual, predicted):
    """
    Calculate evaluation metrics
    
    Metrics Priority:
    - MAPE: Main metric for accuracy (lower is better)
    - MAE: Error magnitude (intuitive, Rs. units)
    - RMSE: Detects large deviations (sensitive to outliers)
    - MBE: Systematic over/under prediction (+ = over, - = under)
    - R²: Reference only (coefficient of determination)
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 1. MAPE - Main metric (Average Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # 2. MAE - Mean Absolute Error (intuitive, in Rs.)
    mae = np.mean(np.abs(actual - predicted))
    
    # 3. RMSE - Root Mean Squared Error (detects large deviations)
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    
    # 4. MBE - Mean Bias Error (systematic bias detection)
    # Positive = over-prediction, Negative = under-prediction
    mbe = np.mean(predicted - actual)
    
    # 5. R² - Coefficient of Determination (reference only)
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        'MAPE': mape,      # Main metric
        'MAE': mae,        # Intuitive
        'RMSE': rmse,      # Large deviations
        'MBE': mbe,        # Systematic bias
        'R2': r2           # Reference only
    }


def fit_arima_model(monthly_data, forecast_steps=3):
    """
    Fit ARIMA model with grid search
    
    Parameters:
    -----------
    monthly_data : Series
        Monthly revenue time series
    forecast_steps : int
        Number of periods to forecast
        
    Returns:
    --------
    tuple
        (forecast, lower_bound, upper_bound, best_order, fitted_model)
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        best_aic = float('inf')
        best_model = None
        best_order = None
        
        # Grid search for best parameters
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(monthly_data.values, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q)
                    except:
                        continue
        
        if best_model is not None:
            # Forecast
            forecast = best_model.forecast(steps=forecast_steps)
            forecast_obj = best_model.get_forecast(steps=forecast_steps)
            conf_int = forecast_obj.conf_int()
            
            # Handle different conf_int return types
            if isinstance(conf_int, np.ndarray):
                lower = conf_int[:, 0]
                upper = conf_int[:, 1]
            else:
                lower = conf_int.iloc[:, 0].values
                upper = conf_int.iloc[:, 1].values
            
            return (forecast, lower, upper, best_order, best_model)
        else:
            return None, None, None, None, None
            
    except ImportError:
        print("Warning: statsmodels not installed, ARIMA unavailable")
        return None, None, None, None, None


def fit_exponential_smoothing(monthly_data, forecast_steps=3):
    """
    Fit Exponential Smoothing (Holt-Winters)
    
    Parameters:
    -----------
    monthly_data : Series
        Monthly revenue time series
    forecast_steps : int
        Number of periods to forecast
        
    Returns:
    --------
    tuple
        (forecast, fitted_model)
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Try Holt-Winters with different configurations
        best_aic = float('inf')
        best_forecast = None
        best_model = None
        
        configs = [
            {'trend': 'add', 'seasonal': None},
            {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 4},
            {'trend': None, 'seasonal': None},
        ]
        
        for config in configs:
            try:
                model = ExponentialSmoothing(monthly_data.values, **config)
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_forecast = fitted.forecast(steps=forecast_steps)
                    best_model = fitted
            except:
                continue
        
        if best_forecast is not None:
            return best_forecast, best_model
        else:
            # Fallback to simple exponential smoothing
            alpha = 0.3
            forecast = []
            last_value = monthly_data.values[-1]
            for _ in range(forecast_steps):
                forecast.append(last_value)
                last_value = alpha * last_value + (1 - alpha) * monthly_data.mean()
            return np.array(forecast), None
        
    except ImportError:
        # Simple exponential smoothing fallback
        alpha = 0.3
        forecast = []
        last_value = monthly_data.values[-1]
        for _ in range(forecast_steps):
            forecast.append(last_value)
            last_value = alpha * last_value + (1 - alpha) * monthly_data.mean()
        return np.array(forecast), None


def fit_prophet_model(monthly_data, forecast_steps=3):
    """
    Fit Facebook Prophet model
    
    Parameters:
    -----------
    monthly_data : Series
        Monthly revenue time series with datetime index
    forecast_steps : int
        Number of periods to forecast
        
    Returns:
    --------
    tuple
        (forecast, fitted_model)
    """
    try:
        from prophet import Prophet
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': monthly_data.index,
            'y': monthly_data.values
        })
        
        # Fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_steps, freq='MS')
        forecast_df = model.predict(future)
        
        # Get forecast values and intervals
        forecast = forecast_df['yhat'].tail(forecast_steps).values
        lower = forecast_df['yhat_lower'].tail(forecast_steps).values
        upper = forecast_df['yhat_upper'].tail(forecast_steps).values
        
        return forecast, model, lower, upper
        
    except ImportError:
        print("Warning: prophet not installed, Prophet model unavailable")
        return None, None


def fit_linear_trend(monthly_data, forecast_steps=3):
    """
    Fit linear trend model
    
    Parameters:
    -----------
    monthly_data : Series
        Monthly revenue time series
    forecast_steps : int
        Number of periods to forecast
        
    Returns:
    --------
    tuple
        (forecast, fitted_model)
    """
    from sklearn.linear_model import LinearRegression
    
    X = np.arange(len(monthly_data)).reshape(-1, 1)
    y = monthly_data.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast
    X_future = np.arange(len(monthly_data), len(monthly_data) + forecast_steps).reshape(-1, 1)
    forecast = model.predict(X_future)
    
    return forecast, model


def fit_polynomial_trend(monthly_data, degree=2, forecast_steps=3):
    """
    Fit polynomial trend model
    
    Parameters:
    -----------
    monthly_data : Series
        Monthly revenue time series
    degree : int
        Polynomial degree
    forecast_steps : int
        Number of periods to forecast
        
    Returns:
    --------
    tuple
        (forecast, coefficients)
    """
    X = np.arange(len(monthly_data))
    y = monthly_data.values
    
    # Fit polynomial
    coeffs = np.polyfit(X, y, degree)
    poly = np.poly1d(coeffs)
    
    # Forecast
    X_future = np.arange(len(monthly_data), len(monthly_data) + forecast_steps)
    forecast = poly(X_future)
    
    return forecast, coeffs


def fit_moving_average(monthly_data, window=3, forecast_steps=3):
    """
    Moving average forecast
    
    Parameters:
    -----------
    monthly_data : Series
        Monthly revenue time series
    window : int
        Moving window size
    forecast_steps : int
        Number of periods to forecast
        
    Returns:
    --------
    array
        Forecast values
    """
    recent_avg = monthly_data.values[-window:].mean()
    forecast = np.array([recent_avg] * forecast_steps)
    
    return forecast


def cross_validate_model(monthly_data, model_func, n_splits=3):
    """
    Perform time series cross-validation
    
    Parameters:
    -----------
    monthly_data : Series
        Monthly revenue time series
    model_func : callable
        Model fitting function
    n_splits : int
        Number of CV splits
        
    Returns:
    --------
    list
        List of metrics for each split
    """
    n = len(monthly_data)
    min_train_size = 6  # Minimum 6 months for training
    
    if n < min_train_size + n_splits:
        return []
    
    metrics_list = []
    
    for i in range(n_splits):
        # Expanding window
        train_size = min_train_size + i
        test_size = min(3, n - train_size)
        
        if test_size <= 0:
            break
        
        train_data = monthly_data[:train_size]
        test_data = monthly_data[train_size:train_size + test_size]
        
        try:
            # Get forecast
            if 'arima' in model_func.__name__.lower():
                forecast, _, _, _, _ = model_func(train_data, forecast_steps=test_size)
            elif 'prophet' in model_func.__name__.lower():
                forecast, _ = model_func(train_data, forecast_steps=test_size)
            else:
                forecast, _ = model_func(train_data, forecast_steps=test_size)
            
            if forecast is not None:
                metrics = calculate_metrics(test_data.values, forecast)
                metrics_list.append(metrics)
        except:
            continue
    
    return metrics_list


# Old monthly-only prediction function removed - using daily-to-monthly approach instead

def run_revenue_prediction_analysis(df):
    """
    Main function for revenue prediction analysis
    Uses DAILY revenue prediction for better accuracy, then aggregates to monthly
    
    Parameters:
    -----------
    df : DataFrame
        Order data
    """
    print("\n" + "="*80)
    print("Analysis 4: Revenue Prediction with Daily-to-Monthly Approach")
    print("="*80)
    print("\n⚡ NEW APPROACH: Daily revenue prediction → Monthly aggregation")
    print("   Reason: 365 daily data points vs 12 monthly points")
    print("   Benefit: Better trend & seasonality detection")
    
    # Calculate DAILY revenue first
    print("\n📅 Step 1: Calculating 2024 daily revenue...")
    daily_revenue = calculate_daily_revenue(df)
    
    print(f"✓ {len(daily_revenue)} days of data")
    print(f"✓ Total revenue: Rs.{daily_revenue['Revenue'].sum():,.0f}")
    print(f"✓ Average daily revenue: Rs.{daily_revenue['Revenue'].mean():,.0f}")
    print(f"✓ Date range: {daily_revenue['Date'].min().date()} to {daily_revenue['Date'].max().date()}")
    
    # Aggregate to monthly for comparison
    print("\n📊 Step 2: Aggregating to monthly for visualization...")
    monthly_revenue = calculate_monthly_revenue(df, from_daily=daily_revenue)
    print(f"✓ {len(monthly_revenue)} months of data")
    print(f"✓ Average monthly revenue: Rs.{monthly_revenue['Revenue'].mean():,.0f}")
    
    # Train models on DAILY data and predict next 90 days
    print("\n🤖 Step 3: Training models on DAILY data...")
    print("   Predicting next 90 days (2025-01-01 to 2025-03-31)...")
    
    daily_series = daily_revenue.set_index('Date')['Revenue']
    
    # Fit models on daily data
    print("\nTraining core time series models on daily data...")
    print("Models: ARIMA, Exponential Smoothing, Prophet")
    
    daily_arima_forecast, daily_arima_lower, daily_arima_upper, arima_order, arima_model = fit_arima_model(daily_series, forecast_steps=90)
    daily_exp_forecast, exp_model = fit_exponential_smoothing(daily_series, forecast_steps=90)
    daily_prophet_forecast, prophet_model, daily_prophet_lower, daily_prophet_upper = fit_prophet_model(daily_series, forecast_steps=90)
    
    daily_models = {}
    if daily_arima_forecast is not None:
        daily_models['ARIMA'] = daily_arima_forecast
        print(f"  ✓ ARIMA{arima_order} trained on daily data")
    
    daily_models['Exp Smoothing'] = daily_exp_forecast
    print(f"  ✓ Exponential Smoothing trained on daily data")
    
    if daily_prophet_forecast is not None:
        daily_models['Prophet'] = daily_prophet_forecast
        print(f"  ✓ Prophet trained on daily data")
    
    # Calculate ensemble from daily forecasts
    daily_ensemble_forecast = np.mean(list(daily_models.values()), axis=0)
    print(f"  ✓ Ensemble forecast calculated ({len(daily_models)} models)")
    
    # Aggregate daily forecasts to monthly
    print("\n📊 Step 4: Aggregating daily forecasts to monthly (Q1 2025)...")
    
    # Create date range for next 90 days (starting from 2025-01-01)
    last_date = daily_revenue['Date'].max()
    # Ensure we start from 2025-01-01 to get full month predictions
    start_date = pd.Timestamp('2025-01-01')
    future_dates = pd.date_range(start=start_date, periods=90, freq='D')
    
    # Monthly aggregation of daily forecasts
    monthly_forecasts = {}
    for model_name, daily_forecast in daily_models.items():
        # Create DataFrame with dates and forecasts
        daily_forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Revenue': daily_forecast
        })
        daily_forecast_df['YearMonth'] = daily_forecast_df['Date'].dt.to_period('M')
        
        # Aggregate to monthly (take only first 3 months)
        monthly_agg = daily_forecast_df.groupby('YearMonth')['Revenue'].sum().values[:3]
        monthly_forecasts[model_name] = monthly_agg
    
    # Aggregate ensemble
    ensemble_df = pd.DataFrame({
        'Date': future_dates,
        'Revenue': daily_ensemble_forecast
    })
    ensemble_df['YearMonth'] = ensemble_df['Date'].dt.to_period('M')
    ensemble_forecast = ensemble_df.groupby('YearMonth')['Revenue'].sum().values[:3]
    
    future_months = ['2025-01', '2025-02', '2025-03']
    
    print(f"  ✓ Daily forecasts aggregated to 3 months")
    
    # Evaluate models on daily validation data (last 30 days)
    print("\n📈 Step 5: Evaluating models on daily validation data...")
    train_daily = daily_series[:-30]
    test_daily = daily_series[-30:]
    
    model_metrics = {}
    
    if daily_arima_forecast is not None:
        arima_val, _, _, _, _ = fit_arima_model(train_daily, forecast_steps=30)
        if arima_val is not None:
            model_metrics['ARIMA'] = calculate_metrics(test_daily.values, arima_val)
            print(f"  ✓ ARIMA validated: MAPE={model_metrics['ARIMA']['MAPE']:.2f}%")
    
    exp_val, _ = fit_exponential_smoothing(train_daily, forecast_steps=30)
    model_metrics['Exp Smoothing'] = calculate_metrics(test_daily.values, exp_val)
    print(f"  ✓ Exp Smoothing validated: MAPE={model_metrics['Exp Smoothing']['MAPE']:.2f}%")
    
    if daily_prophet_forecast is not None:
        prophet_val, _, _, _ = fit_prophet_model(train_daily, forecast_steps=30)
        if prophet_val is not None:
            model_metrics['Prophet'] = calculate_metrics(test_daily.values, prophet_val)
            print(f"  ✓ Prophet validated: MAPE={model_metrics['Prophet']['MAPE']:.2f}%")
    
    # Get best model
    mape_values = {k: v['MAPE'] for k, v in model_metrics.items()}
    sorted_models = sorted(mape_values.items(), key=lambda x: x[1])
    best_model_name = sorted_models[0][0]
    
    # Create visualization with monthly aggregated results
    print("\n🎨 Step 6: Creating visualization...")
    predictions = create_revenue_prediction_visualization_daily(
        monthly_revenue=monthly_revenue,
        daily_revenue=daily_revenue,
        monthly_forecasts=monthly_forecasts,
        ensemble_forecast=ensemble_forecast,
        future_months=future_months,
        model_metrics=model_metrics,
        best_model_name=best_model_name,
        arima_order=arima_order,
        df=df
    )
    
    # Save results
    print("\n💾 Step 7: Saving prediction results...")
    save_prediction_results_daily(monthly_revenue, predictions)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 2025 Q1 REVENUE FORECAST SUMMARY (Daily-to-Monthly):")
    print("="*80)
    for i, month in enumerate(future_months):
        print(f"  {month}: Rs.{ensemble_forecast[i]:,.0f}")
    print("="*80)
    print(f"  Q1 Total: Rs.{ensemble_forecast.sum():,.0f}")
    print(f"\n🏆 Best Model: {best_model_name} (MAPE: {sorted_models[0][1]:.2f}% on daily data)")
    
    if arima_order is not None:
        print(f"📈 Optimal ARIMA: {arima_order}")
    
    print(f"\n✓ Models trained on: {len(daily_revenue)} daily data points")
    print(f"✓ Models compared: {len(model_metrics)}")
    print("✓ Revenue prediction analysis complete!")
    print("="*80)


def create_revenue_prediction_visualization_daily(monthly_revenue, daily_revenue, monthly_forecasts, 
                                                   ensemble_forecast, future_months, model_metrics,
                                                   best_model_name, arima_order, df):
    """
    Create visualization for daily-to-monthly prediction approach
    
    Parameters:
    -----------
    monthly_revenue : DataFrame
        Historical monthly revenue
    daily_revenue : DataFrame
        Historical daily revenue  
    monthly_forecasts : dict
        Monthly forecasts from each model
    ensemble_forecast : array
        Ensemble monthly forecast
    future_months : list
        Future month labels
    model_metrics : dict
        Model evaluation metrics
    best_model_name : str
        Name of best performing model
    arima_order : tuple
        ARIMA order parameters
    df : DataFrame
        Original data
    """
    # Set style
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    
    # Create figure
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 3, hspace=0.50, wspace=0.40)
    
    months = monthly_revenue['YearMonth'].dt.strftime('%Y-%m').values
    model_colors = {'ARIMA': '#E74C3C', 'Exp Smoothing': '#27AE60', 'Prophet': '#8E44AD'}
    model_markers = {'ARIMA': 's', 'Exp Smoothing': '^', 'Prophet': 'D'}
    
    # ===== CHART 1: Historical Revenue Trend (Monthly) =====
    ax1 = fig.add_subplot(gs[0, :2])
    
    bars = ax1.bar(range(len(months)), monthly_revenue['Revenue'].values, 
                   color=plt.cm.Blues(np.linspace(0.4, 0.8, len(months))),
                   edgecolor='navy', linewidth=1.5, alpha=0.8)
    
    # Trend line
    z = np.polyfit(range(len(months)), monthly_revenue['Revenue'].values, 1)
    p = np.poly1d(z)
    ax1.plot(range(len(months)), p(range(len(months))), 
             "r--", linewidth=2.5, alpha=0.7, 
             label=f'Trend (slope: Rs.{z[0]:.0f}/month)')
    
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Revenue (Rs.)', fontsize=12, fontweight='bold')
    ax1.set_title('2024 Monthly Revenue Trend\n(Aggregated from Daily Data)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(range(len(months)))
    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height/1e6:.2f}M',
                ha='center', va='bottom', fontsize=9)
    
    # ===== CHART 2: Statistics & Approach Summary =====
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    total_revenue = monthly_revenue['Revenue'].sum()
    avg_daily = daily_revenue['Revenue'].mean()
    max_daily = daily_revenue['Revenue'].max()
    days_count = len(daily_revenue)
    
    stats_text = f"""
    DAILY-TO-MONTHLY PREDICTION
    {'='*48}
    
    📅 Training Data:
       Days: {days_count} (vs 12 months)
       Advantage: 30x more data points
    
    💰 2024 Revenue:
       Total:     Rs.{total_revenue:,.0f}
                  ({total_revenue/1e6:.2f} Million)
       Avg/Day:   Rs.{avg_daily:,.0f}
       Max/Day:   Rs.{max_daily:,.0f}
    
    🤖 Prediction Method:
       1. Train on 365 daily points
       2. Forecast 90 days ahead
       3. Aggregate to 3 months
    
    ✅ Benefits:
       • Better trend capture
       • Seasonality detection
       • Higher accuracy
    
    Total Orders: {monthly_revenue['Order_Count'].sum():,}
    """
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=9.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # ===== CHART 3: Multi-Model Forecast Comparison =====
    ax3 = fig.add_subplot(gs[1, :])
    
    x_hist = range(len(months))
    ax3.plot(x_hist, monthly_revenue['Revenue'].values, 
            'o-', color='navy', linewidth=3, markersize=8,
            label='Historical Revenue', zorder=5)
    
    x_future = range(len(months), len(months) + 3)
    
    # Plot forecasts
    for name, forecast in monthly_forecasts.items():
        marker = model_markers.get(name, 'o')
        ax3.plot(x_future, forecast,
                f'{marker}-', color=model_colors.get(name, 'gray'), 
                linewidth=2, markersize=7,
                label=f'{name} (from daily)', alpha=0.8, zorder=4)
    
    # Ensemble
    ax3.plot(x_future, ensemble_forecast,
            '*-', color='black', linewidth=3, markersize=12,
            label='Ensemble (Daily→Monthly)', zorder=6)
    
    ax3.axvline(x=len(months)-0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax3.text(len(months)-0.5, ax3.get_ylim()[1]*0.95, 
            'Historical | Forecast', 
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Add ensemble explanation
    ax3.text(0.02, 0.98, 'Ensemble = Average of all models', 
            transform=ax3.transAxes, ha='left', va='top', fontsize=8,
            style='italic', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.6))
    
    all_months = list(months) + future_months
    ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Revenue (Rs.)', fontsize=12, fontweight='bold')
    ax3.set_title('2025 Q1 Revenue Forecast - Daily-to-Monthly Aggregation', 
                 fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(range(len(all_months)))
    ax3.set_xticklabels(all_months, rotation=45, ha='right')
    
    # Adjust Y-axis
    all_values = list(monthly_revenue['Revenue'].values)
    for forecast in monthly_forecasts.values():
        all_values.extend(forecast)
    all_values.extend(ensemble_forecast)
    y_min = min(all_values) * 0.85
    y_max = max(all_values) * 1.15
    ax3.set_ylim(y_min, y_max)
    
    # Add data labels on historical points (offset upwards to avoid being hidden by points)
    for i, (x, y) in enumerate(zip(x_hist, monthly_revenue['Revenue'].values)):
        offset = (y_max - y_min) * 0.03  # 3% offset
        ax3.text(x, y + offset, f'{y/1e6:.1f}M', ha='center', va='bottom', fontsize=7.5, 
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='navy', alpha=0.85, edgecolor='white', linewidth=0.5))
    
    # Add data labels on forecast points with yellow highlight (offset upwards)
    for i, (x, y) in enumerate(zip(x_future, ensemble_forecast)):
        offset = (y_max - y_min) * 0.03
        ax3.text(x, y + offset, f'{y/1e6:.2f}M', ha='center', va='bottom', fontsize=9, 
                fontweight='bold', color='black', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9, 
                         edgecolor='orange', linewidth=1.5))
    
    ax3.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.95, bbox_to_anchor=(0.0, 0.98))
    ax3.grid(True, alpha=0.3)
    
    # ===== CHART 4: Model Error Comparison (4 metrics with independent Y-axes) =====
    ax4 = fig.add_subplot(gs[2, :2])
    ax4.axis('off')
    
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_sub = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2, :2], wspace=0.30)
    
    # Only compare top models (exclude Prophet if too poor)
    filtered_metrics = {k: v for k, v in model_metrics.items() if v['MAPE'] < 100}
    
    metric_names = ['MAPE', 'MAE', 'RMSE', 'R²']
    # Use consistent colors per model instead of per metric
    model_colors_bar = {'ARIMA': '#E74C3C', 'Exp Smoothing': '#27AE60', 'Prophet': '#8E44AD'}
    
    for i, metric in enumerate(metric_names):
        ax_sub = fig.add_subplot(gs_sub[0, i])
        
        model_names = list(filtered_metrics.keys())
        metric_key = metric if metric != 'R²' else 'R2'
        values = [filtered_metrics[model][metric_key] for model in model_names]
        x_pos = np.arange(len(filtered_metrics))
        
        # Assign colors based on model name, not metric
        bar_colors = [model_colors_bar.get(model, '#3498DB') for model in model_names]
        bars = ax_sub.bar(x_pos, values, alpha=0.8, color=bar_colors, 
                         edgecolor='black', linewidth=1.5)
        
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if metric == 'MAPE':
                ax_sub.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2f}%', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
            elif metric == 'R²':
                ax_sub.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
            else:
                ax_sub.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val/1000:.0f}K', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        ax_sub.set_xlabel('Model', fontsize=9, fontweight='bold')
        ylabel = f'{metric}' if metric in ['MAPE', 'R²'] else f'{metric} (Rs.)'
        ax_sub.set_ylabel(ylabel, fontsize=9, fontweight='bold')
        ax_sub.set_title(f'{metric}\\n(30-day validation)', fontsize=9, fontweight='bold', 
                        pad=8)
        ax_sub.set_xticks(x_pos)
        ax_sub.set_xticklabels(model_names, rotation=0, ha='center', fontsize=8)
        ax_sub.grid(True, alpha=0.3, axis='y')
        
        if i == 0:
            ax_sub.text(0.5, 0.95, 'Lower = Better', 
                       transform=ax_sub.transAxes, fontsize=7, style='italic',
                       ha='center', va='top')
        elif i == 3:
            ax_sub.text(0.5, 0.95, 'Higher = Better', 
                       transform=ax_sub.transAxes, fontsize=7, style='italic',
                       ha='center', va='top')
    
    # ===== CHART 5: Daily Revenue Volatility (Full Year with Moving Average) =====
    ax5 = fig.add_subplot(gs[2, 2])
    
    # Show full daily data with 7-day moving average
    daily_ma7 = daily_revenue['Revenue'].rolling(window=7, center=True).mean()
    overall_avg = daily_revenue['Revenue'].mean()
    
    # Plot every 7th day to avoid overcrowding
    sample_indices = range(0, len(daily_revenue), 7)
    ax5.scatter([i for i in sample_indices], 
               [daily_revenue['Revenue'].iloc[i]/1000 for i in sample_indices],
               color='lightblue', s=15, alpha=0.5, label='Daily Revenue (weekly sample)')
    
    ax5.plot(range(len(daily_revenue)), daily_ma7.values/1000,
            '-', color='navy', linewidth=2.5, label='7-Day Moving Avg')
    ax5.axhline(y=overall_avg/1000, color='red', 
               linestyle='--', linewidth=2, label=f'Overall Avg: Rs.{overall_avg/1000:.0f}K', alpha=0.7)
    
    ax5.set_xlabel('Days (2024 Full Year)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Revenue (K Rs.)', fontsize=10, fontweight='bold')
    ax5.set_title('Daily Revenue Volatility\n(365 days with 7-day MA)', 
                 fontsize=11, fontweight='bold', pad=10)
    ax5.legend(fontsize=7, loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, len(daily_revenue))
    
    # ===== CHART 6: Model Validation on Last 3 Months =====
    ax6 = fig.add_subplot(gs[3, 0])
    
    # FIXED: Align actual and predicted periods properly
    # Use last 3 COMPLETE months of 2024 as validation set
    validation_months = monthly_revenue.tail(3)
    val_months_labels = validation_months['YearMonth'].dt.strftime('%Y-%m').values
    val_actual = validation_months['Revenue'].values
    
    # Train on daily data UP TO the start of these 3 months
    # Find the date where the last 3 complete months start (convert Period to Timestamp)
    last_3_months_start = validation_months['YearMonth'].iloc[0].to_timestamp()
    
    # Get training data (all days before the last 3 months)
    train_daily = daily_revenue[daily_revenue['Date'] < last_3_months_start].copy()
    train_daily_series = train_daily.set_index('Date')['Revenue']
    
    # Get validation data (all days in the last 3 months)
    val_daily = daily_revenue[daily_revenue['Date'] >= last_3_months_start].copy()
    val_days = len(val_daily)
    
    # Predict on validation set using DAILY models, then aggregate to monthly
    val_predictions = {}
    
    # Get validation period dates (exactly matching the last 3 complete months)
    pred_dates = pd.date_range(start=last_3_months_start, periods=val_days, freq='D')
    
    # ARIMA prediction
    arima_daily_pred, _, _, _, _ = fit_arima_model(train_daily_series, forecast_steps=val_days)
    if arima_daily_pred is not None:
        # Aggregate daily predictions to monthly
        pred_df = pd.DataFrame({'Date': pred_dates, 'Revenue': arima_daily_pred})
        pred_df['YearMonth'] = pred_df['Date'].dt.to_period('M')
        monthly_pred = pred_df.groupby('YearMonth')['Revenue'].sum().values[:3]  # Ensure only 3 months
        val_predictions['ARIMA'] = monthly_pred
    
    # Exp Smoothing prediction
    exp_daily_pred, _ = fit_exponential_smoothing(train_daily_series, forecast_steps=val_days)
    pred_df = pd.DataFrame({'Date': pred_dates, 'Revenue': exp_daily_pred})
    pred_df['YearMonth'] = pred_df['Date'].dt.to_period('M')
    monthly_pred = pred_df.groupby('YearMonth')['Revenue'].sum().values[:3]  # Ensure only 3 months
    val_predictions['Exp Smoothing'] = monthly_pred
    
    # Prophet prediction
    prophet_daily_pred, _, _, _ = fit_prophet_model(train_daily_series, forecast_steps=val_days)
    if prophet_daily_pred is not None:
        pred_df = pd.DataFrame({'Date': pred_dates, 'Revenue': prophet_daily_pred})
        pred_df['YearMonth'] = pred_df['Date'].dt.to_period('M')
        monthly_pred = pred_df.groupby('YearMonth')['Revenue'].sum().values[:3]  # Ensure only 3 months
        val_predictions['Prophet'] = monthly_pred
    
    # Plot actual values
    x_pos = np.arange(3)
    width = 0.2
    
    ax6.bar(x_pos - 1.5*width, val_actual/1e6, width, alpha=0.9, color='navy', 
           label='Actual', edgecolor='black', linewidth=2)
    
    # Plot predictions for each model
    for i, (model_name, pred) in enumerate(val_predictions.items()):
        color = model_colors.get(model_name, 'gray')
        offset = (i - 0.5) * width
        ax6.bar(x_pos + offset, pred/1e6, width, alpha=0.7, color=color, 
               label=model_name, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, val in enumerate(val_actual):
        ax6.text(x_pos[i] - 1.5*width, val/1e6, f'{val/1e6:.2f}M',
                ha='center', va='bottom', fontsize=7, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='navy', alpha=0.8))
    
    # Add labels for predictions
    for i, (model_name, pred) in enumerate(val_predictions.items()):
        offset = (i - 0.5) * width
        for j, val in enumerate(pred):
            if val > 0:  # Only show positive values
                ax6.text(x_pos[j] + offset, val/1e6, f'{val/1e6:.2f}M',
                        ha='center', va='bottom', fontsize=6.5, rotation=0)
    
    # Adjust Y-axis asymmetrically: compress negative range, expand positive range
    all_vals = list(val_actual) + [v for pred in val_predictions.values() for v in pred]
    max_positive = max([v for v in all_vals if v > 0]) / 1e6
    min_negative = min([v for v in all_vals if v < 0]) / 1e6 if any(v < 0 for v in all_vals) else 0
    
    # Set asymmetric limits: negative range compressed to 20% of total range
    if min_negative < 0:
        positive_range = max_positive * 1.15
        negative_range = abs(min_negative) * 0.3  # Compress negative to 30%
        ax6.set_ylim(-negative_range, positive_range)
    else:
        ax6.set_ylim(0, max_positive * 1.15)
    
    ax6.set_ylabel('Monthly Revenue (M Rs.)', fontsize=10, fontweight='bold')
    ax6.set_title('Model Validation Performance\n(Last 3 Months of 2024)', 
                 fontsize=11, fontweight='bold', pad=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(val_months_labels, fontsize=9)
    ax6.legend(fontsize=8, loc='lower left', ncol=2)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)  # Add zero line
    
    # Add validation info
    ax6.text(0.98, 0.98, 'Train: Jan-Sep 2024 (9 months)\nValidate: Oct-Dec 2024 (3 months)', 
            transform=ax6.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    # ===== CHART 7: Model Performance Radar (5 Metrics) =====
    ax7_radar = fig.add_subplot(gs[3, 1], projection='polar')
    
    radar_metrics = ['MAPE', 'MAE', 'RMSE', 'MBE', 'R2']
    filtered_models_radar = {k: v for k, v in model_metrics.items() if v['MAPE'] < 100}
    
    # Store actual values for display
    actual_values = {}
    normalized_data = {}
    
    for model in filtered_models_radar.keys():
        actual_vals = []
        scores = []
        for metric in radar_metrics:
            metric_key = 'R2' if metric == 'R2' else metric
            val = model_metrics[model][metric_key]
            actual_vals.append(val)
            
            all_vals = [filtered_models_radar[m][metric_key] for m in filtered_models_radar.keys()]
            
            # Normalize to 0-100 scale - Larger circle = better performance
            if metric == 'R2':
                # R2: higher is better, scale to 0-100
                max_r2 = max(all_vals) if max(all_vals) > 0 else 1
                score = (val / max_r2) * 100
            elif metric == 'MBE':
                # MBE: closer to 0 is better, invert abs value
                abs_vals = [abs(v) for v in all_vals]
                max_abs = max(abs_vals) if max(abs_vals) > 0 else 1
                # Smaller absolute value gets higher score (better)
                score = 100 - (abs(val) / max_abs) * 100
            else:  # MAPE, MAE, RMSE: lower is better, invert
                max_val = max(all_vals)
                min_val = min(all_vals)
                if max_val > min_val:
                    # Lower value gets higher score (better)
                    score = 100 - ((val - min_val) / (max_val - min_val)) * 100
                else:
                    score = 100
            scores.append(max(0, min(100, score)))  # Clamp to 0-100
        
        actual_values[model] = actual_vals
        normalized_data[model] = scores
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    ax7_radar.set_theta_offset(np.pi / 2)
    ax7_radar.set_theta_direction(-1)
    ax7_radar.set_ylim(0, 115)
    
    colors_radar = {'ARIMA': '#E74C3C', 'Exp Smoothing': '#27AE60', 'Prophet': '#8E44AD'}
    
    # Plot each model
    for model in filtered_models_radar.keys():
        values = normalized_data[model]
        values += values[:1]
        color = colors_radar.get(model, '#3498DB')
        
        ax7_radar.plot(angles, values, 'o-', linewidth=2.8, label=model, 
                      color=color, markersize=7, markeredgecolor='white', 
                      markeredgewidth=1.5, zorder=3)
        ax7_radar.fill(angles, values, alpha=0.12, color=color, zorder=1)
    
    # Add metric labels with actual values
    for i, (angle, metric) in enumerate(zip(angles[:-1], radar_metrics)):
        # Display actual values for best model
        best_model_sorted = sorted(model_metrics.items(), key=lambda x: x[1]['MAPE'])
        best_model = best_model_sorted[0][0]
        metric_key = 'R2' if metric == 'R2' else metric
        actual_val = model_metrics[best_model][metric_key]
        
        if metric == 'MAPE':
            val_text = f'{actual_val:.1f}%'
        elif metric == 'R2':
            val_text = f'{actual_val:.3f}'
        elif metric == 'MBE':
            val_text = f'{actual_val/1000:+.0f}K'
        else:
            val_text = f'{actual_val/1000:.0f}K'
        
        # Position metric name and best model value
        ax7_radar.text(angle, 118, f'{metric}\n({val_text})', 
                      ha='center', va='top', fontsize=8, 
                      fontweight='bold', color='black',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                               alpha=0.8, edgecolor='gray', linewidth=1))
    
    ax7_radar.set_xticks(angles[:-1])
    ax7_radar.set_xticklabels([''] * len(radar_metrics), fontsize=8)  # Hide default labels
    ax7_radar.set_yticks([25, 50, 75, 100])
    ax7_radar.set_yticklabels(['25', '50', '75', '100'], fontsize=7, color='gray')
    ax7_radar.set_title('Model Performance Radar Chart\n(Normalized Score 0-100)', 
                 fontsize=11, fontweight='bold', pad=25)
    ax7_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9, 
                    framealpha=0.9, edgecolor='black')
    ax7_radar.grid(True, alpha=0.5, linestyle='--', linewidth=0.8)
    
    # Add comprehensive note in English
    note_text = 'Larger Circle = Better Performance\nValues in () are best model actual scores'
    ax7_radar.text(0.5, -0.15, note_text, 
                  transform=ax7_radar.transAxes, ha='center', fontsize=7, 
                  style='italic', color='darkblue',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', 
                           alpha=0.8, edgecolor='steelblue', linewidth=1))
    
    # ===== CHART 8: Model Summary & Q1 Forecast =====
    ax8_summary = fig.add_subplot(gs[3, 2])
    ax8_summary.axis('off')
    
    summary_text = "MODEL SUMMARY & Q1 FORECAST\n"
    summary_text += "=" * 35 + "\n\n"
    summary_text += "📊 Training: 365 daily points\n"
    summary_text += "✅ Validation: Last 30 days\n\n"
    
    sorted_models_list = sorted(model_metrics.items(), key=lambda x: x[1]['MAPE'])
    
    for rank, (model, metrics) in enumerate(sorted_models_list, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "❌"
        summary_text += f"{emoji} {model}:\n"
        summary_text += f"   MAPE: {metrics['MAPE']:.2f}%\n"
    
    summary_text += "\n" + "─" * 35 + "\n"
    summary_text += "💰 Q1 2025 FORECAST:\n\n"
    
    q1_total = 0
    for i, month in enumerate(future_months):
        month_rev = ensemble_forecast[i]
        q1_total += month_rev
        summary_text += f"  {month}: Rs.{month_rev/1e6:.1f}M\n"
    
    summary_text += "\n" + "─" * 35 + "\n"
    summary_text += f"Q1 Total: Rs.{q1_total/1e6:.1f}M\n\n"
    summary_text += "📊 Method:\n"
    summary_text += "  • Daily predictions\n"
    summary_text += "  • Ensemble average\n"
    summary_text += f"  • Best: {best_model_name}\n"
    
    ax8_summary.text(0.05, 0.95, summary_text, transform=ax8_summary.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4, pad=0.8))
    
    plt.suptitle('NCR RIDE BOOKINGS - Revenue Forecasting (Daily-to-Monthly Approach)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_path = f"{OUTPUT_DIR}/4_revenue_prediction.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Revenue prediction chart saved: {output_path}")
    
    return {
        'future_months': future_months,
        'models': monthly_forecasts,
        'ensemble_forecast': ensemble_forecast,
        'best_model': best_model_name,
        'best_mape': sorted(model_metrics.items(), key=lambda x: x[1]['MAPE'])[0][1]['MAPE'],
        'model_metrics': model_metrics,
        'arima_order': arima_order
    }


def save_prediction_results_daily(monthly_revenue, predictions):
    """Save daily-to-monthly prediction results"""
    # Similar to original but note the approach
    forecast_data = {'Month': predictions['future_months']}
    
    for model_name, forecast in predictions['models'].items():
        forecast_data[f'{model_name}_DailyAgg'] = forecast
    
    forecast_data['Ensemble_DailyAgg'] = predictions['ensemble_forecast']
    forecast_data['Best_Model'] = [predictions['best_model']] * 3
    forecast_data['Approach'] = ['Daily→Monthly'] * 3
    
    forecast_df = pd.DataFrame(forecast_data)
    
    forecast_path = f"{OUTPUT_DIR}/revenue_forecast_2025Q1_daily.csv"
    forecast_df.to_csv(forecast_path, index=False, encoding='utf-8-sig')
    print(f"✓ Forecast results saved: {forecast_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame(predictions['model_metrics']).T
    metrics_df['Model'] = metrics_df.index
    metrics_df = metrics_df[['Model', 'MAPE', 'MAE', 'RMSE', 'MBE', 'R2']]
    metrics_df = metrics_df.sort_values('MAPE')
    metrics_df['Note'] = 'Evaluated on daily data (30-day validation)'
    
    metrics_path = f"{OUTPUT_DIR}/model_performance_metrics_daily.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"✓ Model metrics saved: {metrics_path}")


# Old monthly prediction code removed - now using daily-to-monthly approach
