#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Data Analysis System for Negative Space Imaging Project
Author: Stephen Bilodeau
Date: August 13, 2025

This module provides advanced automated analysis of imaging data,
including pattern recognition, anomaly detection, and statistical analysis.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import cluster, decomposition, ensemble, manifold, metrics
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_analysis')

# Import performance monitoring to track resource usage
try:
    from performance_monitor import PerformanceMonitor
    performance_monitor = PerformanceMonitor()
    PERFORMANCE_MONITORING = True
except ImportError:
    logger.warning("Performance monitoring not available")
    PERFORMANCE_MONITORING = False


class DataAnalysisSystem:
    """Advanced data analysis system for negative space imaging data."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data analysis system.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.results_dir = self.config.get('results_directory', 'analysis_results')

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize analysis components
        self.analysis_modules = {
            'statistical': self._statistical_analysis,
            'clustering': self._clustering_analysis,
            'dimensionality': self._dimensionality_reduction,
            'anomaly': self._anomaly_detection,
            'pattern': self._pattern_recognition,
            'trend': self._trend_analysis,
            'correlation': self._correlation_analysis
        }

        logger.info(f"Data Analysis System initialized with config: {config_path}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        default_config = {
            'results_directory': 'analysis_results',
            'default_analysis_types': ['statistical', 'clustering', 'anomaly'],
            'visualization_enabled': True,
            'export_formats': ['json', 'csv'],
            'clustering': {
                'max_clusters': 10,
                'methods': ['kmeans', 'dbscan']
            },
            'anomaly_detection': {
                'contamination': 0.05,
                'methods': ['isolation_forest', 'local_outlier_factor']
            },
            'dimensionality_reduction': {
                'target_dimensions': 2,
                'methods': ['pca', 'tsne']
            }
        }

        if not config_path or not os.path.exists(config_path):
            logger.info("Using default configuration")
            return default_config

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing values
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict) and isinstance(config[key], dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return default_config

    def analyze_data(self,
                    data_path: str,
                    analysis_types: Optional[List[str]] = None,
                    output_prefix: Optional[str] = None,
                    visualization: bool = True) -> Dict[str, Any]:
        """Perform automated analysis on the provided data.

        Args:
            data_path: Path to data file (CSV, JSON, or numpy array)
            analysis_types: List of analysis types to perform
            output_prefix: Prefix for output files
            visualization: Whether to generate visualizations

        Returns:
            Dictionary of analysis results
        """
        if PERFORMANCE_MONITORING:
            performance_monitor.start_monitoring()

        logger.info(f"Starting analysis of {data_path}")

        # Set default analysis types if not provided
        if not analysis_types:
            analysis_types = self.config.get('default_analysis_types',
                                            ['statistical', 'clustering', 'anomaly'])

        # Load data
        try:
            data = self._load_data(data_path)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {'error': f"Data loading failed: {str(e)}"}

        # Generate output prefix if not provided
        if not output_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"analysis_{timestamp}"

        # Perform requested analyses
        results = {'metadata': {
            'data_path': data_path,
            'analysis_date': datetime.now().isoformat(),
            'analysis_types': analysis_types
        }}

        for analysis_type in analysis_types:
            if analysis_type in self.analysis_modules:
                logger.info(f"Performing {analysis_type} analysis")
                try:
                    if PERFORMANCE_MONITORING:
                        performance_monitor.add_checkpoint(f"Before {analysis_type}")

                    # Call the appropriate analysis function
                    analysis_results = self.analysis_modules[analysis_type](data)
                    results[analysis_type] = analysis_results

                    if PERFORMANCE_MONITORING:
                        performance_monitor.add_checkpoint(f"After {analysis_type}")
                except Exception as e:
                    logger.error(f"Error during {analysis_type} analysis: {e}")
                    results[analysis_type] = {'error': str(e)}
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}")

        # Generate visualizations if requested
        if visualization and self.config.get('visualization_enabled', True):
            self._generate_visualizations(data, results, output_prefix)

        # Export results
        self._export_results(results, output_prefix)

        if PERFORMANCE_MONITORING:
            performance_data = performance_monitor.stop_monitoring()
            results['performance'] = performance_data

        logger.info(f"Analysis completed, results saved with prefix {output_prefix}")
        return results

    def _load_data(self, data_path: str) -> Union[np.ndarray, pd.DataFrame]:
        """Load data from various file formats.

        Args:
            data_path: Path to data file

        Returns:
            Loaded data as numpy array or pandas DataFrame
        """
        file_ext = os.path.splitext(data_path)[1].lower()

        if file_ext == '.csv':
            return pd.read_csv(data_path)
        elif file_ext == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
            # Convert to DataFrame if it's a list of records
            if isinstance(data, list):
                return pd.DataFrame(data)
            return data
        elif file_ext in ['.npy', '.npz']:
            return np.load(data_path)
        elif file_ext in ['.fits', '.fit']:
            from astropy.io import fits
            return fits.getdata(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def _statistical_analysis(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform statistical analysis on the data.

        Args:
            data: Input data

        Returns:
            Dictionary of statistical results
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=['value'])
            else:
                data = pd.DataFrame(data)

        # Calculate basic statistics
        try:
            stats_results = {
                'summary': data.describe().to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }

            # Calculate additional statistics for numeric columns
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                numeric_data = data[numeric_cols]
                stats_results['skewness'] = numeric_data.skew().to_dict()
                stats_results['kurtosis'] = numeric_data.kurtosis().to_dict()

                # Check for normality
                normality_tests = {}
                for col in numeric_cols:
                    if data[col].dropna().shape[0] > 8:  # Minimum sample size for the test
                        stat, p_value = stats.shapiro(data[col].dropna())
                        normality_tests[col] = {
                            'statistic': stat,
                            'p_value': p_value,
                            'is_normal': p_value > 0.05
                        }
                stats_results['normality_tests'] = normality_tests

            return stats_results
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {'error': str(e)}

    def _clustering_analysis(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform clustering analysis on the data.

        Args:
            data: Input data

        Returns:
            Dictionary of clustering results
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=['value'])
            else:
                data = pd.DataFrame(data)

        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] == 0:
            return {'error': 'No numeric data available for clustering'}

        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        clustering_results = {}

        # K-means clustering
        if 'kmeans' in self.config.get('clustering', {}).get('methods', ['kmeans']):
            kmeans_results = {}
            max_clusters = self.config.get('clustering', {}).get('max_clusters', 10)

            # Find optimal number of clusters using silhouette score
            silhouette_scores = []
            for n_clusters in range(2, min(max_clusters + 1, len(numeric_data) // 5 + 1)):
                kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                silhouette_avg = metrics.silhouette_score(scaled_data, cluster_labels)
                silhouette_scores.append((n_clusters, silhouette_avg))

            # Get optimal number of clusters
            optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0] if silhouette_scores else 2

            # Perform K-means with optimal clusters
            kmeans = cluster.KMeans(n_clusters=optimal_clusters, random_state=42)
            labels = kmeans.fit_predict(scaled_data)

            # Calculate metrics
            kmeans_results['optimal_clusters'] = optimal_clusters
            kmeans_results['silhouette_score'] = metrics.silhouette_score(scaled_data, labels)
            kmeans_results['inertia'] = kmeans.inertia_
            kmeans_results['cluster_centers'] = kmeans.cluster_centers_.tolist()
            kmeans_results['labels'] = labels.tolist()

            # Calculate cluster statistics
            cluster_stats = []
            for i in range(optimal_clusters):
                cluster_data = numeric_data.iloc[labels == i]
                cluster_stats.append({
                    'cluster_id': i,
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(numeric_data) * 100,
                    'mean': cluster_data.mean().to_dict(),
                    'std': cluster_data.std().to_dict()
                })
            kmeans_results['cluster_stats'] = cluster_stats

            clustering_results['kmeans'] = kmeans_results

        # DBSCAN clustering
        if 'dbscan' in self.config.get('clustering', {}).get('methods', ['dbscan']):
            dbscan_results = {}

            # Estimate eps parameter using nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(5, len(numeric_data) - 1))
            nn.fit(scaled_data)
            distances, _ = nn.kneighbors(scaled_data)
            distances = np.sort(distances[:, -1])

            # Heuristic to find "elbow" point for eps
            eps = np.percentile(distances, 90) * 0.5

            # Run DBSCAN
            dbscan = cluster.DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(scaled_data)

            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            dbscan_results['n_clusters'] = n_clusters
            dbscan_results['eps'] = eps
            dbscan_results['labels'] = labels.tolist()
            dbscan_results['noise_points'] = sum(1 for l in labels if l == -1)

            # Calculate cluster statistics
            cluster_stats = []
            for i in set(labels):
                if i != -1:  # Skip noise points
                    cluster_data = numeric_data.iloc[labels == i]
                    cluster_stats.append({
                        'cluster_id': int(i),
                        'size': len(cluster_data),
                        'percentage': len(cluster_data) / len(numeric_data) * 100,
                        'mean': cluster_data.mean().to_dict(),
                        'std': cluster_data.std().to_dict()
                    })
            dbscan_results['cluster_stats'] = cluster_stats

            clustering_results['dbscan'] = dbscan_results

        return clustering_results

    def _dimensionality_reduction(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform dimensionality reduction on the data.

        Args:
            data: Input data

        Returns:
            Dictionary of dimensionality reduction results
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=['value'])
            else:
                data = pd.DataFrame(data)

        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] == 0:
            return {'error': 'No numeric data available for dimensionality reduction'}

        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        dr_results = {}
        target_dims = self.config.get('dimensionality_reduction', {}).get('target_dimensions', 2)

        # PCA
        if 'pca' in self.config.get('dimensionality_reduction', {}).get('methods', ['pca']):
            pca = decomposition.PCA(n_components=min(target_dims, numeric_data.shape[1]))
            pca_result = pca.fit_transform(scaled_data)

            dr_results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': sum(pca.explained_variance_ratio_),
                'n_components': pca.n_components_,
                'components': pca.components_.tolist(),
                'transformed_data': pca_result.tolist()
            }

        # t-SNE
        if 'tsne' in self.config.get('dimensionality_reduction', {}).get('methods', ['tsne']):
            # Only run t-SNE if we have a reasonable number of samples
            if len(numeric_data) <= 10000:
                tsne = manifold.TSNE(n_components=min(target_dims, numeric_data.shape[1]),
                                    random_state=42)
                tsne_result = tsne.fit_transform(scaled_data)

                dr_results['tsne'] = {
                    'kl_divergence': tsne.kl_divergence_,
                    'n_components': tsne.n_components,
                    'transformed_data': tsne_result.tolist()
                }
            else:
                dr_results['tsne'] = {
                    'error': 'Dataset too large for t-SNE, skipping (>10000 samples)'
                }

        # UMAP if available
        try:
            import umap
            if 'umap' in self.config.get('dimensionality_reduction', {}).get('methods', []):
                umap_reducer = umap.UMAP(n_components=min(target_dims, numeric_data.shape[1]),
                                        random_state=42)
                umap_result = umap_reducer.fit_transform(scaled_data)

                dr_results['umap'] = {
                    'n_components': umap_reducer.n_components,
                    'transformed_data': umap_result.tolist()
                }
        except ImportError:
            logger.warning("UMAP not available, skipping UMAP dimensionality reduction")

        return dr_results

    def _anomaly_detection(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform anomaly detection on the data.

        Args:
            data: Input data

        Returns:
            Dictionary of anomaly detection results
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=['value'])
            else:
                data = pd.DataFrame(data)

        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] == 0:
            return {'error': 'No numeric data available for anomaly detection'}

        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())

        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)

        anomaly_results = {}
        contamination = self.config.get('anomaly_detection', {}).get('contamination', 0.05)

        # Isolation Forest
        if 'isolation_forest' in self.config.get('anomaly_detection', {}).get('methods',
                                                                            ['isolation_forest']):
            iso_forest = ensemble.IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(scaled_data)

            # Convert predictions (-1 for outliers, 1 for inliers) to anomaly flag (True for outliers)
            anomalies = np.where(predictions == -1, True, False)
            anomaly_indices = np.where(anomalies)[0].tolist()

            anomaly_results['isolation_forest'] = {
                'anomaly_count': sum(anomalies),
                'anomaly_percentage': sum(anomalies) / len(anomalies) * 100,
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': iso_forest.decision_function(scaled_data).tolist()
            }

            # Add anomaly details if we have indices
            if anomaly_indices and isinstance(data, pd.DataFrame):
                anomaly_details = []
                for idx in anomaly_indices:
                    if idx < len(data):
                        row_data = data.iloc[idx].to_dict()
                        score = iso_forest.decision_function(scaled_data[idx].reshape(1, -1))[0]
                        anomaly_details.append({
                            'index': idx,
                            'score': float(score),
                            'data': row_data
                        })
                anomaly_results['isolation_forest']['anomaly_details'] = anomaly_details

        # Local Outlier Factor
        if 'local_outlier_factor' in self.config.get('anomaly_detection', {}).get('methods',
                                                                                ['local_outlier_factor']):
            lof = neighbors.LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            predictions = lof.fit_predict(scaled_data)

            # Convert predictions (-1 for outliers, 1 for inliers) to anomaly flag (True for outliers)
            anomalies = np.where(predictions == -1, True, False)
            anomaly_indices = np.where(anomalies)[0].tolist()

            # Calculate the negative outlier factor
            negative_of = lof.negative_outlier_factor_

            anomaly_results['local_outlier_factor'] = {
                'anomaly_count': sum(anomalies),
                'anomaly_percentage': sum(anomalies) / len(anomalies) * 100,
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': negative_of.tolist()
            }

            # Add anomaly details if we have indices
            if anomaly_indices and isinstance(data, pd.DataFrame):
                anomaly_details = []
                for idx in anomaly_indices:
                    if idx < len(data):
                        row_data = data.iloc[idx].to_dict()
                        score = float(negative_of[idx])
                        anomaly_details.append({
                            'index': idx,
                            'score': score,
                            'data': row_data
                        })
                anomaly_results['local_outlier_factor']['anomaly_details'] = anomaly_details

        return anomaly_results

    def _pattern_recognition(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform pattern recognition on the data.

        Args:
            data: Input data

        Returns:
            Dictionary of pattern recognition results
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=['value'])
            else:
                data = pd.DataFrame(data)

        pattern_results = {}

        # Time series analysis if we have timestamp column
        time_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            try:
                # Convert to datetime
                data[time_col] = pd.to_datetime(data[time_col])
                data = data.sort_values(time_col)

                # Select numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns
                numeric_cols = [col for col in numeric_cols if col != time_col]

                if numeric_cols:
                    time_patterns = {}

                    # Check for seasonality
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    for col in numeric_cols[:5]:  # Limit to first 5 columns
                        try:
                            # Create a time series
                            ts = data.set_index(time_col)[col].dropna()
                            if len(ts) > 10:  # Need sufficient data
                                # Try to infer frequency
                                if ts.index.inferred_freq is None:
                                    # If frequency can't be inferred, resample to daily
                                    ts = ts.resample('D').mean().dropna()

                                if len(ts) > 10:  # Still need sufficient data after resampling
                                    # Decompose time series
                                    result = seasonal_decompose(ts, model='additive', extrapolate_trend='freq')

                                    time_patterns[col] = {
                                        'trend': result.trend.dropna().tolist(),
                                        'seasonal': result.seasonal.dropna().tolist(),
                                        'residual': result.resid.dropna().tolist(),
                                        'strength_of_seasonality': 1 - (np.var(result.resid) / np.var(result.seasonal + result.resid))
                                    }
                        except Exception as e:
                            logger.warning(f"Could not perform seasonal decomposition for {col}: {e}")

                    if time_patterns:
                        pattern_results['time_series'] = time_patterns
            except Exception as e:
                logger.warning(f"Error in time series analysis: {e}")

        # Feature correlation patterns
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] > 1:
            try:
                corr_matrix = numeric_data.corr()

                # Find highly correlated features
                high_corr_pairs = []
                for i in range(corr_matrix.shape[0]):
                    for j in range(i+1, corr_matrix.shape[1]):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': corr_matrix.iloc[i, j]
                            })

                pattern_results['feature_correlations'] = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'high_correlation_pairs': high_corr_pairs
                }
            except Exception as e:
                logger.warning(f"Error in correlation analysis: {e}")

        return pattern_results

    def _trend_analysis(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform trend analysis on the data.

        Args:
            data: Input data

        Returns:
            Dictionary of trend analysis results
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=['value'])
            else:
                data = pd.DataFrame(data)

        trend_results = {}

        # Time series analysis if we have timestamp column
        time_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            try:
                # Convert to datetime
                data[time_col] = pd.to_datetime(data[time_col])
                data = data.sort_values(time_col)

                # Select numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns
                numeric_cols = [col for col in numeric_cols if col != time_col]

                if numeric_cols:
                    time_trends = {}

                    for col in numeric_cols[:5]:  # Limit to first 5 columns
                        try:
                            # Create a time series
                            ts = data.set_index(time_col)[col].dropna()
                            if len(ts) > 10:  # Need sufficient data
                                # Linear trend
                                from scipy import stats
                                x = np.arange(len(ts))
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts.values)

                                # Check if trend is statistically significant
                                is_significant = p_value < 0.05

                                # Direction of trend
                                trend_direction = "increasing" if slope > 0 else "decreasing"

                                # Strength of trend
                                trend_strength = abs(r_value)

                                time_trends[col] = {
                                    'slope': slope,
                                    'intercept': intercept,
                                    'r_squared': r_value**2,
                                    'p_value': p_value,
                                    'std_error': std_err,
                                    'is_significant': is_significant,
                                    'direction': trend_direction,
                                    'strength': trend_strength
                                }
                        except Exception as e:
                            logger.warning(f"Could not perform trend analysis for {col}: {e}")

                    if time_trends:
                        trend_results['linear_trends'] = time_trends
            except Exception as e:
                logger.warning(f"Error in trend analysis: {e}")

        return trend_results

    def _correlation_analysis(self, data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """Perform correlation analysis on the data.

        Args:
            data: Input data

        Returns:
            Dictionary of correlation analysis results
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=['value'])
            else:
                data = pd.DataFrame(data)

        correlation_results = {}

        # Pearson correlation
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] > 1:
            try:
                pearson_corr = numeric_data.corr(method='pearson')

                # Find statistically significant correlations
                from scipy import stats
                significant_corrs = []
                for i in range(pearson_corr.shape[0]):
                    for j in range(i+1, pearson_corr.shape[1]):
                        x = numeric_data.iloc[:, i].dropna()
                        y = numeric_data.iloc[:, j].dropna()

                        # Get common indices
                        common_idx = x.index.intersection(y.index)
                        if len(common_idx) > 2:
                            x = x.loc[common_idx]
                            y = y.loc[common_idx]

                            r, p = stats.pearsonr(x, y)
                            if p < 0.05:
                                significant_corrs.append({
                                    'feature1': pearson_corr.columns[i],
                                    'feature2': pearson_corr.columns[j],
                                    'correlation': r,
                                    'p_value': p
                                })

                correlation_results['pearson'] = {
                    'correlation_matrix': pearson_corr.to_dict(),
                    'significant_correlations': significant_corrs
                }
            except Exception as e:
                logger.warning(f"Error in Pearson correlation analysis: {e}")

        # Spearman rank correlation (non-parametric)
        if numeric_data.shape[1] > 1:
            try:
                spearman_corr = numeric_data.corr(method='spearman')

                # Find statistically significant correlations
                from scipy import stats
                significant_corrs = []
                for i in range(spearman_corr.shape[0]):
                    for j in range(i+1, spearman_corr.shape[1]):
                        x = numeric_data.iloc[:, i].dropna()
                        y = numeric_data.iloc[:, j].dropna()

                        # Get common indices
                        common_idx = x.index.intersection(y.index)
                        if len(common_idx) > 2:
                            x = x.loc[common_idx]
                            y = y.loc[common_idx]

                            r, p = stats.spearmanr(x, y)
                            if p < 0.05:
                                significant_corrs.append({
                                    'feature1': spearman_corr.columns[i],
                                    'feature2': spearman_corr.columns[j],
                                    'correlation': r,
                                    'p_value': p
                                })

                correlation_results['spearman'] = {
                    'correlation_matrix': spearman_corr.to_dict(),
                    'significant_correlations': significant_corrs
                }
            except Exception as e:
                logger.warning(f"Error in Spearman correlation analysis: {e}")

        return correlation_results

    def _generate_visualizations(self,
                                data: Union[np.ndarray, pd.DataFrame],
                                results: Dict[str, Any],
                                output_prefix: str) -> None:
        """Generate visualizations for analysis results.

        Args:
            data: Input data
            results: Analysis results
            output_prefix: Prefix for output files
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame(data, columns=['value'])
            else:
                data = pd.DataFrame(data)

        # Create visualizations directory
        vis_dir = os.path.join(self.results_dir, f"{output_prefix}_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Set up matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')

        # Generate statistical visualizations
        if 'statistical' in results:
            try:
                numeric_data = data.select_dtypes(include=['number'])
                if not numeric_data.empty:
                    # Histogram for each numeric column
                    for col in numeric_data.columns[:10]:  # Limit to first 10
                        plt.figure(figsize=(10, 6))
                        sns.histplot(numeric_data[col].dropna(), kde=True)
                        plt.title(f'Distribution of {col}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(vis_dir, f"histogram_{col}.png"))
                        plt.close()

                    # Box plots
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(data=numeric_data.iloc[:, :10])  # First 10 columns
                    plt.title('Box Plot of Numeric Features')
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, "boxplot_features.png"))
                    plt.close()

                    # Correlation heatmap
                    if numeric_data.shape[1] > 1:
                        plt.figure(figsize=(12, 10))
                        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
                        plt.title('Correlation Heatmap')
                        plt.tight_layout()
                        plt.savefig(os.path.join(vis_dir, "correlation_heatmap.png"))
                        plt.close()
            except Exception as e:
                logger.warning(f"Error generating statistical visualizations: {e}")

        # Generate clustering visualizations
        if 'clustering' in results and 'dimensionality' in results:
            try:
                # K-means visualization with PCA
                if 'kmeans' in results['clustering'] and 'pca' in results['dimensionality']:
                    kmeans_results = results['clustering']['kmeans']
                    pca_results = results['dimensionality']['pca']

                    if 'labels' in kmeans_results and 'transformed_data' in pca_results:
                        labels = kmeans_results['labels']
                        pca_data = pca_results['transformed_data']

                        if len(labels) == len(pca_data) and len(pca_data[0]) >= 2:
                            plt.figure(figsize=(10, 8))
                            scatter = plt.scatter([p[0] for p in pca_data],
                                                [p[1] for p in pca_data],
                                                c=labels,
                                                cmap='viridis',
                                                alpha=0.8)
                            plt.colorbar(scatter, label='Cluster')
                            plt.title('K-means Clustering (PCA Projection)')
                            plt.xlabel('Principal Component 1')
                            plt.ylabel('Principal Component 2')
                            plt.tight_layout()
                            plt.savefig(os.path.join(vis_dir, "kmeans_clustering_pca.png"))
                            plt.close()
            except Exception as e:
                logger.warning(f"Error generating clustering visualizations: {e}")

        # Generate anomaly detection visualizations
        if 'anomaly' in results and 'dimensionality' in results:
            try:
                # Isolation Forest visualization with PCA
                if ('isolation_forest' in results['anomaly'] and
                    'pca' in results['dimensionality']):
                    anomaly_results = results['anomaly']['isolation_forest']
                    pca_results = results['dimensionality']['pca']

                    if ('anomaly_indices' in anomaly_results and
                        'transformed_data' in pca_results):
                        anomaly_indices = set(anomaly_results['anomaly_indices'])
                        pca_data = pca_results['transformed_data']

                        if pca_data and len(pca_data[0]) >= 2:
                            # Create anomaly flag array
                            anomaly_flags = [i in anomaly_indices for i in range(len(pca_data))]

                            plt.figure(figsize=(10, 8))
                            normal_points = [p for i, p in enumerate(pca_data) if not anomaly_flags[i]]
                            anomaly_points = [p for i, p in enumerate(pca_data) if anomaly_flags[i]]

                            if normal_points:
                                plt.scatter([p[0] for p in normal_points],
                                            [p[1] for p in normal_points],
                                            c='blue',
                                            label='Normal',
                                            alpha=0.6)

                            if anomaly_points:
                                plt.scatter([p[0] for p in anomaly_points],
                                            [p[1] for p in anomaly_points],
                                            c='red',
                                            label='Anomaly',
                                            alpha=0.8)

                            plt.title('Anomaly Detection (PCA Projection)')
                            plt.xlabel('Principal Component 1')
                            plt.ylabel('Principal Component 2')
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(os.path.join(vis_dir, "anomaly_detection_pca.png"))
                            plt.close()
            except Exception as e:
                logger.warning(f"Error generating anomaly detection visualizations: {e}")

        # Generate trend analysis visualizations
        if 'trend' in results:
            try:
                time_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
                if time_cols and 'linear_trends' in results['trend']:
                    time_col = time_cols[0]
                    linear_trends = results['trend']['linear_trends']

                    # Convert to datetime
                    data[time_col] = pd.to_datetime(data[time_col])
                    data = data.sort_values(time_col)

                    for feature, trend_info in linear_trends.items():
                        plt.figure(figsize=(12, 6))
                        sns.lineplot(x=time_col, y=feature, data=data)

                        # Add trend line
                        if 'slope' in trend_info and 'intercept' in trend_info:
                            x = np.arange(len(data))
                            y = trend_info['slope'] * x + trend_info['intercept']
                            plt.plot(data[time_col], y, 'r--',
                                    label=f"Trend: y={trend_info['slope']:.4f}x + {trend_info['intercept']:.4f}")

                        plt.title(f'Time Series Trend for {feature}')
                        plt.xlabel('Time')
                        plt.ylabel(feature)
                        plt.xticks(rotation=45)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(vis_dir, f"trend_{feature}.png"))
                        plt.close()
            except Exception as e:
                logger.warning(f"Error generating trend visualizations: {e}")

        # Create visualization index
        vis_files = [f for f in os.listdir(vis_dir) if f.endswith('.png')]
        if vis_files:
            with open(os.path.join(vis_dir, "visualizations.html"), "w") as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Analysis Visualizations: {output_prefix}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .vis-container {{ display: flex; flex-wrap: wrap; }}
        .vis-item {{ margin: 10px; box-shadow: 0 0 5px rgba(0,0,0,0.2); }}
        .vis-item img {{ max-width: 500px; }}
        .vis-item h3 {{ padding: 10px; margin: 0; background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>Analysis Visualizations: {output_prefix}</h1>
    <div class="vis-container">
""")

                for vis_file in sorted(vis_files):
                    # Create a nice title from filename
                    title = ' '.join(os.path.splitext(vis_file)[0].split('_')).title()
                    f.write(f"""
        <div class="vis-item">
            <h3>{title}</h3>
            <img src="{vis_file}" alt="{title}">
        </div>
""")

                f.write("""
    </div>
</body>
</html>
""")

        logger.info(f"Generated visualizations in {vis_dir}")

    def _export_results(self, results: Dict[str, Any], output_prefix: str) -> None:
        """Export analysis results to files.

        Args:
            results: Analysis results
            output_prefix: Prefix for output files
        """
        export_formats = self.config.get('export_formats', ['json', 'csv'])

        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export as JSON (always)
        json_path = os.path.join(self.results_dir, f"{output_prefix}_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Export as CSV if requested
        if 'csv' in export_formats:
            # Export each analysis type as a separate CSV file if possible
            for analysis_type, analysis_results in results.items():
                if analysis_type in ['metadata', 'performance']:
                    continue

                try:
                    # Try to convert to DataFrame
                    csv_data = self._results_to_dataframe(analysis_results, analysis_type)
                    if csv_data is not None:
                        csv_path = os.path.join(self.results_dir,
                                               f"{output_prefix}_{analysis_type}_{timestamp}.csv")
                        csv_data.to_csv(csv_path, index=True)
                except Exception as e:
                    logger.warning(f"Could not export {analysis_type} results to CSV: {e}")

        logger.info(f"Results exported to {self.results_dir}")

    def _results_to_dataframe(self,
                             results: Dict[str, Any],
                             analysis_type: str) -> Optional[pd.DataFrame]:
        """Convert analysis results to a pandas DataFrame for CSV export.

        Args:
            results: Analysis results for a specific type
            analysis_type: Type of analysis

        Returns:
            DataFrame or None if conversion is not possible
        """
        if analysis_type == 'statistical':
            if 'summary' in results:
                return pd.DataFrame(results['summary'])

        elif analysis_type == 'clustering':
            if 'kmeans' in results and 'cluster_stats' in results['kmeans']:
                return pd.DataFrame(results['kmeans']['cluster_stats'])

        elif analysis_type == 'anomaly':
            if 'isolation_forest' in results and 'anomaly_details' in results['isolation_forest']:
                return pd.DataFrame(results['isolation_forest']['anomaly_details'])

        elif analysis_type == 'trend':
            if 'linear_trends' in results:
                return pd.DataFrame(results['linear_trends']).T

        elif analysis_type == 'correlation':
            if 'pearson' in results and 'significant_correlations' in results['pearson']:
                return pd.DataFrame(results['pearson']['significant_correlations'])

        return None


def main():
    """Command-line interface for the data analysis system."""
    parser = argparse.ArgumentParser(description='Automated Data Analysis System')
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', help='Output prefix for results')
    parser.add_argument('--analysis', nargs='+',
                       choices=['statistical', 'clustering', 'dimensionality',
                               'anomaly', 'pattern', 'trend', 'correlation', 'all'],
                       help='Analysis types to perform')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Disable visualization generation')

    args = parser.parse_args()

    # Initialize the data analysis system
    analyzer = DataAnalysisSystem(args.config)

    # Prepare analysis types
    if args.analysis:
        if 'all' in args.analysis:
            analysis_types = list(analyzer.analysis_modules.keys())
        else:
            analysis_types = args.analysis
    else:
        analysis_types = None  # Use defaults from config

    # Run analysis
    results = analyzer.analyze_data(
        data_path=args.data,
        analysis_types=analysis_types,
        output_prefix=args.output,
        visualization=not args.no_visualizations
    )

    print(f"Analysis completed. Results saved in {analyzer.results_dir}")

    # Print summary
    if 'statistical' in results:
        print("\nSummary Statistics:")
        if 'summary' in results['statistical']:
            for feature, stats in results['statistical']['summary'].items():
                if 'mean' in stats:
                    print(f"  {feature}: mean={stats['mean']:.4f}, std={stats.get('std', 'N/A')}")

    if 'anomaly' in results:
        print("\nAnomaly Detection:")
        for method, anomaly_results in results['anomaly'].items():
            if 'anomaly_count' in anomaly_results:
                print(f"  {method}: detected {anomaly_results['anomaly_count']} anomalies "
                     f"({anomaly_results.get('anomaly_percentage', 0):.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
