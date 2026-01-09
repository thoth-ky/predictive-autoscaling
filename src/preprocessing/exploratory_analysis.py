"""
Exploratory Data Analysis (EDA) Module

Provides visualization and analysis functions to help understand the training data,
including distribution plots, time series visualizations, and correlation analysis.
"""

import os
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


class ExploratoryAnalyzer:
    """
    Performs exploratory data analysis on training data.

    Generates text-based visualizations and statistics that work well
    in terminal environments.
    """

    def __init__(self):
        self.analyses = {}

    def analyze_raw_data(
        self, df: pd.DataFrame, metric_name: str
    ) -> Dict[str, Any]:
        """
        Analyze raw data distribution and patterns.

        Args:
            df: Raw DataFrame
            metric_name: Name of metric

        Returns:
            Dictionary of analysis results
        """
        analysis = {
            "metric_name": metric_name,
            "histogram": self._create_histogram(df["value"].dropna(), bins=20),
            "time_series_summary": self._summarize_time_series(df),
        }

        if "container_labels" in df.columns or "container_name" in df.columns:
            container_col = (
                "container_name"
                if "container_name" in df.columns
                else "container_labels"
            )
            analysis["container_summary"] = self._summarize_containers(
                df, container_col
            )

        return analysis

    def analyze_feature_correlations(
        self,
        processed: pd.DataFrame,
        feature_columns: List[str],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze correlations between features and target.

        Args:
            processed: Processed DataFrame with features
            feature_columns: List of feature column names
            top_n: Number of top correlations to show

        Returns:
            Dictionary of correlation analysis
        """
        if "value" not in processed.columns:
            return {"error": "No 'value' column found for correlation analysis"}

        # Compute correlations with target
        numeric_features = [
            col
            for col in feature_columns
            if col in processed.columns
            and pd.api.types.is_numeric_dtype(processed[col])
        ]

        if not numeric_features:
            return {"error": "No numeric features found"}

        correlations = {}
        for feature in numeric_features:
            corr = processed[feature].corr(processed["value"])
            if not np.isnan(corr):
                correlations[feature] = float(corr)

        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return {
            "total_features": len(numeric_features),
            "top_positive_correlations": [
                {"feature": k, "correlation": v}
                for k, v in sorted_corr[:top_n]
                if v > 0
            ],
            "top_negative_correlations": [
                {"feature": k, "correlation": v}
                for k, v in sorted_corr[:top_n]
                if v < 0
            ],
            "all_correlations": dict(sorted_corr),
        }

    def analyze_windows(
        self,
        X: np.ndarray,
        y_dict: Dict[int, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze window characteristics.

        Args:
            X: Input windows
            y_dict: Target dictionaries
            feature_names: Names of features

        Returns:
            Dictionary of window analysis
        """
        analysis = {
            "window_stats": {
                "total_windows": len(X),
                "shape": X.shape,
            },
            "feature_variance": self._analyze_feature_variance(X, feature_names),
            "target_distributions": self._analyze_target_distributions(y_dict),
        }

        return analysis

    def _create_histogram(
        self, values: pd.Series, bins: int = 20
    ) -> Dict[str, Any]:
        """
        Create a text-based histogram.

        Args:
            values: Series of values
            bins: Number of histogram bins

        Returns:
            Dictionary with histogram data and text visualization
        """
        counts, bin_edges = np.histogram(values, bins=bins)

        # Create text-based visualization
        max_count = max(counts) if len(counts) > 0 else 1
        bar_width = 50

        hist_lines = []
        for i, count in enumerate(counts):
            bar_length = int((count / max_count) * bar_width)
            bar = "█" * bar_length
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            hist_lines.append(
                f"  [{bin_start:8.2f} - {bin_end:8.2f}]: {bar} ({count})"
            )

        return {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "text_visualization": "\n".join(hist_lines),
        }

    def _summarize_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize time series patterns."""
        if "timestamp" not in df.columns:
            return {}

        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

        # Detect sampling frequency
        time_diffs = df_copy["timestamp"].diff().dt.total_seconds()

        return {
            "total_points": len(df_copy),
            "median_interval_seconds": float(time_diffs.median()),
            "gaps_detected": int((time_diffs > time_diffs.median() * 2).sum()),
            "start_time": str(df_copy["timestamp"].min()),
            "end_time": str(df_copy["timestamp"].max()),
        }

    def _summarize_containers(
        self, df: pd.DataFrame, container_col: str
    ) -> Dict[str, Any]:
        """Summarize container distribution."""
        container_counts = df[container_col].value_counts()

        return {
            "unique_containers": len(container_counts),
            "distribution": {
                str(name): int(count)
                for name, count in container_counts.head(20).items()
            },
            "balance_ratio": float(
                container_counts.max() / container_counts.min()
                if container_counts.min() > 0
                else np.inf
            ),
        }

    def _analyze_feature_variance(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze variance across features."""
        if len(X.shape) != 3:
            return {"error": "Expected 3D array (samples, timesteps, features)"}

        n_samples, n_timesteps, n_features = X.shape

        # Compute variance for each feature across all samples and timesteps
        feature_variances = []
        for f in range(n_features):
            feature_data = X[:, :, f].flatten()
            variance = float(np.var(feature_data))
            mean = float(np.mean(feature_data))

            feature_name = (
                feature_names[f] if feature_names and f < len(feature_names) else f"feature_{f}"
            )

            feature_variances.append(
                {"feature": feature_name, "variance": variance, "mean": mean}
            )

        # Sort by variance (high to low)
        feature_variances.sort(key=lambda x: x["variance"], reverse=True)

        return {
            "n_features": n_features,
            "top_variance_features": feature_variances[:10],
            "low_variance_features": feature_variances[-5:],
        }

    def _analyze_target_distributions(
        self, y_dict: Dict[int, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze target value distributions for each horizon."""
        distributions = {}

        for horizon, y in y_dict.items():
            horizon_minutes = horizon // 4

            # Create histogram
            hist_data = self._create_histogram(pd.Series(y), bins=15)

            distributions[f"{horizon_minutes}min"] = {
                "horizon_timesteps": horizon,
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "histogram": hist_data,
            }

        return distributions

    def generate_eda_report(
        self,
        raw_df: Optional[pd.DataFrame] = None,
        processed_df: Optional[pd.DataFrame] = None,
        X: Optional[np.ndarray] = None,
        y_dict: Optional[Dict[int, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        metric_name: str = "unknown",
    ) -> str:
        """
        Generate comprehensive EDA report.

        Args:
            raw_df: Raw DataFrame
            processed_df: Processed DataFrame
            X: Window data
            y_dict: Target data
            feature_names: Feature names
            metric_name: Metric name

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("EXPLORATORY DATA ANALYSIS REPORT")
        lines.append("=" * 80)

        # Raw data analysis
        if raw_df is not None:
            lines.append("\n### RAW DATA ANALYSIS ###")
            raw_analysis = self.analyze_raw_data(raw_df, metric_name)

            if "histogram" in raw_analysis:
                lines.append(f"\nValue Distribution:")
                lines.append(raw_analysis["histogram"]["text_visualization"])

            if "container_summary" in raw_analysis:
                cs = raw_analysis["container_summary"]
                lines.append(f"\nContainer Distribution:")
                lines.append(
                    f"  Unique containers: {cs.get('unique_containers', 0)}"
                )
                lines.append(
                    f"  Balance ratio: {cs.get('balance_ratio', 0):.2f}"
                )
                if "distribution" in cs:
                    for name, count in list(cs["distribution"].items())[:10]:
                        lines.append(f"    {name}: {count:,}")

        # Feature correlations
        if processed_df is not None and feature_names:
            lines.append("\n### FEATURE CORRELATIONS ###")
            corr_analysis = self.analyze_feature_correlations(
                processed_df, feature_names, top_n=10
            )

            if "top_positive_correlations" in corr_analysis:
                lines.append(f"\nTop Positive Correlations with Target:")
                for item in corr_analysis["top_positive_correlations"][:5]:
                    lines.append(
                        f"  {item['feature']}: {item['correlation']:.4f}"
                    )

            if "top_negative_correlations" in corr_analysis:
                lines.append(f"\nTop Negative Correlations with Target:")
                for item in corr_analysis["top_negative_correlations"][:5]:
                    lines.append(
                        f"  {item['feature']}: {item['correlation']:.4f}"
                    )

        # Window analysis
        if X is not None and y_dict is not None:
            lines.append("\n### WINDOW ANALYSIS ###")
            window_analysis = self.analyze_windows(X, y_dict, feature_names)

            if "feature_variance" in window_analysis:
                fv = window_analysis["feature_variance"]
                lines.append(f"\nFeature Variance (Top 5):")
                for item in fv.get("top_variance_features", [])[:5]:
                    lines.append(
                        f"  {item['feature']}: variance={item['variance']:.6f}, mean={item['mean']:.4f}"
                    )

            if "target_distributions" in window_analysis:
                lines.append(f"\nTarget Distributions by Horizon:")
                td = window_analysis["target_distributions"]
                for horizon_name, hdata in sorted(td.items()):
                    lines.append(
                        f"\n  {horizon_name} (mean={hdata['mean']:.4f}, std={hdata['std']:.4f}):"
                    )
                    # Show mini histogram (first 10 bars)
                    hist_lines = hdata["histogram"]["text_visualization"].split("\n")
                    for line in hist_lines[:10]:
                        lines.append(f"  {line}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def save_report(self, report: str, filepath: str):
        """Save EDA report to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(report)
        print(f"EDA report saved to: {filepath}")
