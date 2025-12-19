#!/usr/bin/env python3
"""
Kubernetes metrics export for container monitoring via Prometheus
"""
import json
import requests
from datetime import datetime, timedelta
import csv
import os
import re


class KubernetesMetricsExporter:
    def __init__(self, prometheus_url="http://localhost:9090", metrics=None):
        self.prom_url = prometheus_url
        # Default metrics to export if none specified
        self.metrics_to_export = metrics or [
            "container_cpu_rate",
            "container_memory_usage",
            "container_network_receive_rate",
            "container_network_transmit_rate",
            "container_fs_reads_rate",
            "container_fs_writes_rate",
        ]

        # Define metric query templates
        self.metric_queries = {
            "container_cpu_rate": 'rate(container_cpu_usage_seconds_total{id=~"/kubepods.*"}[1m])',
            "container_memory_usage": 'container_memory_usage_bytes{id=~"/kubepods.*"}',
            "container_memory_limit": 'container_spec_memory_limit_bytes{id=~"/kubepods.*"}',
            "container_network_receive_rate": 'rate(container_network_receive_bytes_total{id=~"/kubepods.*"}[1m])',
            "container_network_transmit_rate": 'rate(container_network_transmit_bytes_total{id=~"/kubepods.*"}[1m])',
            "container_fs_reads_rate": 'rate(container_fs_reads_total{id=~"/kubepods.*"}[1m])',
            "container_fs_writes_rate": 'rate(container_fs_writes_total{id=~"/kubepods.*"}[1m])',
            "container_fs_read_bytes_rate": 'rate(container_fs_read_bytes_total{id=~"/kubepods.*"}[1m])',
            "container_fs_write_bytes_rate": 'rate(container_fs_write_bytes_total{id=~"/kubepods.*"}[1m])',
        }

    def test_connection(self):
        """Test Prometheus connection"""
        try:
            response = requests.get(f"{self.prom_url}/api/v1/status/config", timeout=5)
            response.raise_for_status()
            print("✅ Connected to Prometheus")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Prometheus: {e}")
            return False

    def get_all_available_metrics(self):
        """Get exhaustive list of all available metrics"""
        print("🔍 Discovering all available metrics...")

        try:
            response = requests.get(
                f"{self.prom_url}/api/v1/label/__name__/values", timeout=10
            )
            response.raise_for_status()
            result = response.json()

            if result["status"] != "success":
                print("❌ Failed to query available metrics")
                return []

            metrics = result["data"]
            print(f"📊 Found {len(metrics)} total metrics")

            return sorted(metrics)

        except Exception as e:
            print(f"❌ Error discovering metrics: {e}")
            return []

    def save_available_metrics(self, metrics, output_dir):
        """Save exhaustive list of available metrics to CSV"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(__file__), output_dir)
        os.makedirs(output_path, exist_ok=True)

        metrics_file = os.path.join(
            output_path, f"available_metrics_{timestamp_str}.csv"
        )

        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric_name", "discovery_timestamp"])
            for metric in metrics:
                writer.writerow([metric, timestamp_str])

        print(f"📋 Saved {len(metrics)} available metrics to: {metrics_file}")
        return metrics_file

    def compress_labels(self, labels_dict):
        """Compress all labels into a single string"""
        # Remove common labels we store separately
        filtered_labels = {
            k: v
            for k, v in labels_dict.items()
            if k not in ["id", "__name__", "job", "instance"]
        }

        # Create compressed string: key1=value1,key2=value2
        if not filtered_labels:
            return ""

        return ",".join([f"{k}={v}" for k, v in sorted(filtered_labels.items())])

    def export_metrics(self, seconds=900, output_dir="./data/raw/metrics", all=False):
        """Export configurable metrics for ALL Kubernetes containers"""
        if not self.test_connection():
            return None

        # Get all available metrics first
        all_metrics = self.get_all_available_metrics()
        if all_metrics:
            self.save_available_metrics(all_metrics, output_dir)

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=seconds)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n📊 Exporting {len(self.metrics_to_export)} metrics:")
        print(f"   Metrics: {', '.join(self.metrics_to_export)}")
        print(f"   From: {start_time}")
        print(f"   To:   {end_time}")
        print(f"   Duration: {seconds} seconds")

        exported_files = []

        # Export each metric separately
        for metric_name in (self.metrics_to_export if not all else all_metrics):
            if metric_name not in self.metric_queries:
                print(f"\n⚠️  Unknown metric '{metric_name}' - skipping")
                continue
                # TODO: Create on the fly queries for new metrics if needed

            query = self.metric_queries[metric_name]
            print(f"\n📥 Fetching {metric_name} for ALL containers")
            print(f"   Query: {query}")

            try:
                params = {
                    "query": query,
                    "start": int(start_time.timestamp()),
                    "end": int(end_time.timestamp()),
                    "step": "15s",
                }

                response = requests.get(
                    f"{self.prom_url}/api/v1/query_range", params=params, timeout=60
                )
                response.raise_for_status()

                result = response.json()

                if result["status"] != "success":
                    print(f"   ⚠️  Query failed: {result.get('error', 'unknown error')}")
                    continue

                data = result["data"]["result"]

                if not data:
                    print(f"   ⚠️  No data returned")
                    continue

                # Prepare data for CSV
                csv_data = []
            except Exception as e:
                print(f"   ❌ Error exporting {metric_name}: {e}")
                continue

            # Process each time series
            count = 0
            for series in data:
                metric_labels = series.get("metric", {})
                values = series.get("values", [])

                # Compress all labels into single column
                container_labels = self.compress_labels(metric_labels)
                container_id = metric_labels.get("id", "")

                for timestamp, value in values:
                    try:
                        numeric_value = float(value) if value != "NaN" else 0.0
                    except (ValueError, TypeError):
                        numeric_value = 0.0

                    row = {
                        "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                        "container_labels": container_labels,
                        "value": numeric_value,
                        "container_id": container_id,  # Keep for debugging
                    }

                    csv_data.append(row)
                    count += 1

                print(f"   ✅ Collected {count:,} points from {len(data)} series")

                if not csv_data:
                    print("\n❌ No data collected!")
                    continue

                # Create output directory
                output_path = os.path.join(os.path.dirname(__file__), output_dir)
                os.makedirs(output_path, exist_ok=True)

                # Save to CSV with timestamp in filename
                output_file = os.path.join(
                    output_path, f"{metric_name}_{timestamp_str}.csv"
                )

                # Write CSV
                fieldnames = ["timestamp", "container_labels", "value", "container_id"]
                with open(output_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_data)

                file_size_mb = os.path.getsize(output_file) / 1024 / 1024

                print(f"\n✅ Export complete!")
                print(f"   Total records: {len(csv_data):,}")
                print(f"   File size: {file_size_mb:.2f} MB")
                print(f"   Saved to: {output_file}")

                # Show summary by unique container labels
                unique_containers = set(row["container_labels"] for row in csv_data)
                print(f"\n📈 Unique containers found: {len(unique_containers)}")

                # Show sample of container labels
                if unique_containers:
                    print("📊 Sample container labels:")
                    for i, labels in enumerate(sorted(unique_containers)[:5]):
                        # Extract container name from labels string
                        container_name = ""
                        for label in labels.split(","):
                            if label.startswith("container="):
                                container_name = label
                                break
                        print(f"   {i+1}: {container_name}")
                    if len(unique_containers) > 5:
                        print(f"   ... and {len(unique_containers) - 5} more")

                exported_files.append(output_file)

        return exported_files


def main():
    import argparse

    default_metrics = [
        "container_cpu_rate",
        "container_memory_usage",
        "container_memory_limit",
        "container_network_receive_rate",
        "container_network_transmit_rate",
        "container_fs_reads_rate",
        "container_fs_writes_rate",
        "container_fs_read_bytes_rate",
        "container_fs_write_bytes_rate",
    ]

    parser = argparse.ArgumentParser(
        description="Export configurable metrics from ALL Kubernetes containers via Prometheus"
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=900,
        help="Number of seconds to export (default: 900)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw/metrics",
        help="Output directory (default: ./data/raw/metrics)",
    )
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default="http://localhost:9090",
        help="Prometheus URL",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=default_metrics,
        help="Metrics to export (default: container_cpu_rate)",
    )

    args = parser.parse_args()

    # Show available metrics
    print("📋 Available metrics:")

    for metric in default_metrics:
        status = "✅" if metric in args.metrics else "  "
        print(f"   {status} {metric}")

    exporter = KubernetesMetricsExporter(args.prometheus_url, metrics=args.metrics)
    exporter.export_metrics(seconds=args.seconds, output_dir=args.output)


if __name__ == "__main__":
    main()
