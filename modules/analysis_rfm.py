"""
Analysis Module 3: RFM Machine Learning Analysis
Complete RFM customer segmentation using K-Means clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .config import OUTPUT_DIR, get_segment_color, get_segment_strategy
from .rfm_visualizations import (
    visualize_3d_clusters,
    create_pie_chart,
    create_bar_comparison_chart,
    create_radar_chart
)


def calculate_rfm(df):
    """
    Calculate RFM metrics for each customer
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ride booking dataframe
        
    Returns:
    --------
    pd.DataFrame
        RFM metrics (Recency, Frequency, Monetary) for each customer
    """
    print("\n" + "="*50)
    print("Calculating RFM Metrics...")
    print("="*50)
    
    snapshot_date = df['Date'].max() + dt.timedelta(days=1)
    df['Booking Value'] = df['Booking Value'].fillna(0)
    
    rfm = df.groupby('Customer ID').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'Booking ID': 'count',
        'Booking Value': 'sum'
    }).reset_index()

    rfm.rename(columns={
        'Date': 'Recency',
        'Booking ID': 'Frequency',
        'Booking Value': 'Monetary'
    }, inplace=True)
    
    print(f"✓ RFM calculated for {len(rfm)} customers")
    print(f"\nRFM Statistics:")
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe())
    
    return rfm


def determine_optimal_k(rfm_scaled, max_k=10):
    """
    Determine optimal number of clusters using multiple methods
    
    Parameters:
    -----------
    rfm_scaled : numpy.ndarray
        Standardized RFM features
    max_k : int
        Maximum number of clusters to test
        
    Returns:
    --------
    int
        Optimal number of clusters
    """
    print("\n" + "="*50)
    print("Determining Optimal Number of Clusters...")
    print("="*50)
    
    # For large datasets, use sampling for faster computation
    n_samples = len(rfm_scaled)
    if n_samples > 10000:
        print(f"⚡ Large dataset detected ({n_samples} customers)")
        print(f"   Using stratified sampling for faster computation...")
        sample_size = min(10000, n_samples)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
        rfm_sample = rfm_scaled[sample_indices]
        print(f"   Sample size: {sample_size} customers")
    else:
        rfm_sample = rfm_scaled
        print(f"   Processing {n_samples} customers")
    
    # Calculate metrics for different K values
    k_range = range(2, max_k + 1)
    wcss = []  # Within-Cluster Sum of Squares
    silhouette_scores = []
    davies_bouldin_scores = []
    
    print(f"\n   Testing K values from 2 to {max_k}...")
    for i, k in enumerate(k_range, 1):
        print(f"   [{i}/{len(k_range)}] K={k}...", end=' ', flush=True)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(rfm_sample)
        
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_sample, kmeans.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(rfm_sample, kmeans.labels_))
        
        print(f"✓ Silhouette: {silhouette_scores[-1]:.3f}")
    
    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Elbow Method
    axes[0].plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('WCSS (Inertia)', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Silhouette Score (higher is better)
    axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Mark optimal K
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    axes[1].axvline(x=optimal_k_silhouette, color='r', linestyle='--', 
                    label=f'Optimal K = {optimal_k_silhouette}')
    axes[1].legend()
    
    # 3. Davies-Bouldin Index (lower is better)
    axes[2].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
    axes[2].set_title('Davies-Bouldin Index', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/3_rfm_elbow_silhouette.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart saved to: {save_path}")
    
    # Smart K selection: combine multiple methods
    print(f"\n📊 Cluster Selection Analysis:")
    
    # Method 1: Silhouette Score (but exclude K=2 for business reasons)
    valid_silhouette = silhouette_scores[1:]  # Exclude K=2
    optimal_k_silhouette = k_range[1:][np.argmax(valid_silhouette)]
    print(f"   Silhouette Score (K≥3): K = {optimal_k_silhouette} (Score: {max(valid_silhouette):.3f})")
    
    # Method 2: Elbow Method - find the "elbow point"
    wcss_diff = np.diff(wcss)
    wcss_diff_ratio = np.abs(wcss_diff[:-1] / wcss_diff[1:])
    elbow_k = k_range[np.argmax(wcss_diff_ratio) + 1]
    print(f"   Elbow Method: K = {elbow_k}")
    
    # Method 3: Davies-Bouldin Index (lower is better, exclude K=2)
    valid_db = davies_bouldin_scores[1:]
    optimal_k_db = k_range[1:][np.argmin(valid_db)]
    print(f"   Davies-Bouldin (K≥3): K = {optimal_k_db} (Score: {min(valid_db):.3f})")
    
    # Final decision: weighted voting with business constraints
    candidates = [optimal_k_silhouette, elbow_k, optimal_k_db]
    
    # Apply business rule: prefer K >= 4 for RFM
    business_preferred = [k for k in candidates if k >= 4]
    
    if business_preferred:
        from collections import Counter
        optimal_k = Counter(business_preferred).most_common(1)[0][0]
        print(f"\n✅ Recommended K: {optimal_k} (Business-optimized)")
    else:
        optimal_k = max(4, int(np.median(candidates)))
        print(f"\n✅ Recommended K: {optimal_k} (Median with K≥4 constraint)")
    
    print(f"\n💡 Reasoning:")
    print(f"   - K=2 excluded (too simple for RFM segmentation)")
    print(f"   - K≥4 preferred for meaningful customer groups")
    print(f"   - Combined multiple metrics for robust selection")
    
    return optimal_k


def perform_kmeans_clustering(rfm, rfm_scaled, optimal_k):
    """
    Perform K-Means clustering with optimal K
    
    Parameters:
    -----------
    rfm : pd.DataFrame
        RFM metrics dataframe
    rfm_scaled : numpy.ndarray
        Standardized RFM features
    optimal_k : int
        Number of clusters
        
    Returns:
    --------
    tuple
        (rfm_with_clusters, kmeans_model, cluster_summary)
    """
    print("\n" + "="*50)
    print(f"Performing K-Means Clustering (K={optimal_k})...")
    print("="*50)
    
    n_samples = len(rfm_scaled)
    print(f"   Clustering {n_samples} customers into {optimal_k} segments...")
    
    # For large datasets, use more efficient parameters
    if n_samples > 50000:
        print(f"   Using optimized parameters for large dataset...")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300, verbose=0)
    else:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    
    print(f"   Training K-Means model...", end=' ', flush=True)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    print("✓")
    
    # Calculate cluster centers in original scale
    scaler = StandardScaler()
    scaler.fit(rfm[['Recency', 'Frequency', 'Monetary']])
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create cluster summary
    print(f"\n✓ Clustering complete! Created {optimal_k} customer segments\n")
    print("Cluster Centers (Original Scale):")
    print("-" * 70)
    
    cluster_summary = pd.DataFrame(
        cluster_centers,
        columns=['Recency', 'Frequency', 'Monetary']
    )
    cluster_summary['Cluster'] = range(optimal_k)
    cluster_summary['Size'] = rfm['Cluster'].value_counts().sort_index().values
    
    print(cluster_summary.to_string(index=False))
    
    return rfm, kmeans, cluster_summary


def assign_business_labels(rfm, cluster_summary):
    """
    Assign business-meaningful labels to clusters based on RFM characteristics
    
    Parameters:
    -----------
    rfm : pd.DataFrame
        RFM metrics with cluster assignments
    cluster_summary : pd.DataFrame
        Cluster summary statistics
        
    Returns:
    --------
    tuple
        (rfm_with_labels, cluster_summary_with_labels)
    """
    print("\n" + "="*50)
    print("Assigning Business Labels to Clusters...")
    print("="*50)
    
    # Normalize cluster centers for comparison
    cluster_summary['R_Norm'] = (cluster_summary['Recency'] - cluster_summary['Recency'].min()) / \
                                  (cluster_summary['Recency'].max() - cluster_summary['Recency'].min())
    cluster_summary['F_Norm'] = (cluster_summary['Frequency'] - cluster_summary['Frequency'].min()) / \
                                  (cluster_summary['Frequency'].max() - cluster_summary['Frequency'].min())
    cluster_summary['M_Norm'] = (cluster_summary['Monetary'] - cluster_summary['Monetary'].min()) / \
                                  (cluster_summary['Monetary'].max() - cluster_summary['Monetary'].min())
    
    # Invert Recency (lower recency is better)
    cluster_summary['R_Norm'] = 1 - cluster_summary['R_Norm']
    
    # Define labeling rules
    def label_cluster(row):
        r, f, m = row['R_Norm'], row['F_Norm'], row['M_Norm']
        
        if f > 0.7 and m > 0.7 and r > 0.6:
            return 'VIP Champions'
        elif f > 0.6 and r > 0.5:
            return 'Loyal Customers'
        elif f > 0.5 and m > 0.5 and r < 0.3:
            return 'At Risk'
        elif f < 0.3 and r > 0.6:
            return 'New Customers'
        elif f > 0.3 and r > 0.4:
            return 'Potential Loyalists'
        elif r < 0.4 and f < 0.5:
            return 'Hibernating'
        elif r < 0.2 and f < 0.3:
            return 'Lost'
        else:
            return 'High Potential'
    
    cluster_summary['Business_Label'] = cluster_summary.apply(label_cluster, axis=1)
    
    # Map labels back to main RFM dataframe
    label_map = dict(zip(cluster_summary['Cluster'], cluster_summary['Business_Label']))
    rfm['Business_Label'] = rfm['Cluster'].map(label_map)
    
    # Print business labels
    print("\n[Customer Segment Business Labels]")
    print("-" * 90)
    for _, row in cluster_summary.iterrows():
        print(f"Cluster {int(row['Cluster'])}: {row['Business_Label']:25s} | "
              f"Size: {int(row['Size']):4d} customers | "
              f"R: {row['Recency']:6.1f} | F: {row['Frequency']:5.1f} | M: {row['Monetary']:8.0f}")
    
    return rfm, cluster_summary


def save_ml_results(rfm, cluster_summary):
    """
    Save ML clustering results to CSV files
    
    Parameters:
    -----------
    rfm : pd.DataFrame
        RFM metrics with cluster assignments and labels
    cluster_summary : pd.DataFrame
        Cluster summary statistics
    """
    print("\n" + "="*50)
    print("Saving Results...")
    print("="*50)
    
    # Save detailed customer data
    rfm_output = rfm[['Customer ID', 'Recency', 'Frequency', 'Monetary', 
                      'Cluster', 'Business_Label']]
    output_path = f"{OUTPUT_DIR}/rfm_ml_result.csv"
    rfm_output.to_csv(output_path, index=False)
    print(f"✓ Customer-level results saved to: {output_path}")
    
    # Save cluster summary
    summary_path = f"{OUTPUT_DIR}/rfm_cluster_summary.csv"
    cluster_summary_output = cluster_summary[['Cluster', 'Business_Label', 'Size', 
                                               'Recency', 'Frequency', 'Monetary']]
    cluster_summary_output.to_csv(summary_path, index=False)
    print(f"✓ Cluster summary saved to: {summary_path}")


def print_marketing_recommendations(cluster_summary):
    """
    Print actionable marketing recommendations for each cluster
    
    Parameters:
    -----------
    cluster_summary : pd.DataFrame
        Cluster summary with business labels
    """
    print("\n" + "="*80)
    print("MARKETING RECOMMENDATIONS BY CUSTOMER SEGMENT")
    print("="*80)
    
    recommendations = {
        'VIP Champions': 'Exclusive rewards, early access to new features, personal account manager',
        'Loyal Customers': 'Upsell premium services, loyalty rewards, referral bonuses',
        'Potential Loyalists': 'Engagement campaigns, membership programs, personalized offers',
        'At Risk': 'Win-back campaigns, satisfaction surveys, special discount offers',
        'Hibernating': 'Re-engagement emails, limited-time offers, feedback requests',
        'Lost': 'Last attempt win-back or remove from active campaigns',
        'New Customers': 'Welcome series, onboarding support, first ride incentives',
        'High Potential': 'Target with premium offers, monitor behavior closely, A/B test strategies'
    }
    
    for _, row in cluster_summary.iterrows():
        label = row['Business_Label']
        size = int(row['Size'])
        r, f, m = row['Recency'], row['Frequency'], row['Monetary']
        
        print(f"\n{label}")
        print(f"  Size: {size} customers ({size/cluster_summary['Size'].sum()*100:.1f}%)")
        print(f"  Profile: Avg Recency={r:.0f} days | Avg Frequency={f:.1f} orders | Avg Monetary=Rs.{m:.0f}")
        print(f"  Strategy: {recommendations.get(label, 'Develop targeted approach')}")
        print("-" * 80)


def analyze_rfm_ml(df):
    """
    Main function for ML-based RFM analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned ride booking dataframe
    """
    # Step 1: Calculate RFM
    rfm = calculate_rfm(df)
    
    # Check data quality
    print("\n" + "="*50)
    print("Data Quality Check...")
    print("="*50)
    single_purchase = (rfm['Frequency'] == 1).sum()
    total_customers = len(rfm)
    print(f"   Total customers: {total_customers}")
    print(f"   Single-purchase customers: {single_purchase} ({single_purchase/total_customers*100:.1f}%)")
    print(f"   Multi-purchase customers: {total_customers - single_purchase} ({(total_customers-single_purchase)/total_customers*100:.1f}%)")
    
    # Step 2: Standardize features
    print("\n" + "="*50)
    print("Standardizing RFM Features...")
    print("="*50)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    print("✓ Features standardized using StandardScaler")
    
    # Step 3: Determine optimal K
    max_k = 8 if len(rfm) > 50000 else 10
    print(f"\n   Testing up to K={max_k} clusters (optimized for dataset size)")
    optimal_k = determine_optimal_k(rfm_scaled, max_k=max_k)
    
    # Step 4: Perform K-Means clustering
    rfm, kmeans, cluster_summary = perform_kmeans_clustering(rfm, rfm_scaled, optimal_k)
    
    # Step 5: Assign business labels
    rfm, cluster_summary = assign_business_labels(rfm, cluster_summary)
    
    # Step 6: Visualizations
    print("\n" + "="*60)
    print("GENERATING RFM VISUALIZATIONS")
    print("="*60)
    
    visualize_3d_clusters(rfm, cluster_summary)
    create_pie_chart(cluster_summary)
    create_bar_comparison_chart(cluster_summary)
    create_radar_chart(cluster_summary)
    
    print("\n✅ All visualizations generated successfully!")
    
    # Step 7: Save results
    save_ml_results(rfm, cluster_summary)
    
    # Step 8: Print marketing recommendations
    print_marketing_recommendations(cluster_summary)
