#!/usr/bin/env python3
"""
Kubernetes metrics export for container monitoring via Prometheus
Optimized for large-scale exports with filtering, chunking, and async processing
"""
import requests
import aiohttp
import asyncio
from datetime import datetime, timedelta
import csv
import os
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging


class KubernetesMetricsExporter:
    def __init__(
        self,
        prometheus_url="http://localhost:9090",
        metrics=None,
        container_filter=None,
        namespace_filter=None,
        label_filters=None,
        chunk_hours=1,
        max_concurrent=5,
        step_interval="15s",
    ):
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

        # Filtering options
        self.container_filter = container_filter  # e.g., "webapp|database"
        self.namespace_filter = namespace_filter  # e.g., "default|production"
        self.label_filters = label_filters or {}  # e.g., {"app": "myapp"}

        # Performance options
        self.chunk_hours = chunk_hours  # Split queries into chunks
        self.max_concurrent = max_concurrent  # Max parallel requests
        self.step_interval = step_interval  # Data point interval

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Define metric query templates (will be modified with filters)
        self.metric_queries_base = {
            "container_cpu_rate": 'rate(container_cpu_usage_seconds_total{filters}[1m])',
            "container_memory_usage": 'container_memory_usage_bytes{filters}',
            "container_memory_limit": 'container_spec_memory_limit_bytes{filters}',
            "container_network_receive_rate": 'rate(container_network_receive_bytes_total{filters}[1m])',
            "container_network_transmit_rate": 'rate(container_network_transmit_bytes_total{filters}[1m])',
            "container_fs_reads_rate": 'rate(container_fs_reads_total{filters}[1m])',
            "container_fs_writes_rate": 'rate(container_fs_writes_total{filters}[1m])',
            "container_fs_read_bytes_rate": 'rate(container_fs_read_bytes_total{filters}[1m])',
            "container_fs_write_bytes_rate": 'rate(container_fs_write_bytes_total{filters}[1m])',
        }

    def _build_label_selector(self) -> str:
        """Build PromQL label selector with filters"""
        selectors = ['id=~"/kubepods.*"']

        if self.container_filter:
            selectors.append(f'container=~"{self.container_filter}"')

        if self.namespace_filter:
            selectors.append(f'namespace=~"{self.namespace_filter}"')

        for label, value in self.label_filters.items():
            if "|" in value or ".*" in value:
                selectors.append(f'{label}=~"{value}"')
            else:
                selectors.append(f'{label}="{value}"')

        return ",".join(selectors)

    def _get_query_for_metric(self, metric_name: str) -> Optional[str]:
        """Get the PromQL query for a metric with filters applied"""
        if metric_name not in self.metric_queries_base:
            return None

        label_selector = self._build_label_selector()
        return self.metric_queries_base[metric_name].replace("{filters}", f"{{{label_selector}}}")

    def _split_time_range(
        self, start_time: datetime, end_time: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Split time range into chunks to avoid timeouts"""
        chunks = []
        current_start = start_time
        chunk_delta = timedelta(hours=self.chunk_hours)

        while current_start < end_time:
            current_end = min(current_start + chunk_delta, end_time)
            chunks.append((current_start, current_end))
            current_start = current_end

        return chunks

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

    async def _fetch_metric_chunk_async(
        self,
        session: aiohttp.ClientSession,
        query: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[Dict]:
        """Fetch a single metric chunk asynchronously"""
        params = {
            "query": query,
            "start": int(start_time.timestamp()),
            "end": int(end_time.timestamp()),
            "step": self.step_interval,
        }

        try:
            async with session.get(
                f"{self.prom_url}/api/v1/query_range",
                params=params,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if result["status"] != "success":
                    logging.error(f"Query failed: {result.get('error', 'unknown')}")
                    return None

                return result["data"]["result"]

        except asyncio.TimeoutError:
            logging.error(f"Timeout fetching chunk {start_time} to {end_time}")
            return None
        except Exception as e:
            logging.error(f"Error fetching chunk: {e}")
            return None

    async def _fetch_metric_parallel(
        self, metric_name: str, start_time: datetime, end_time: datetime
    ) -> List[Dict]:
        """Fetch metric data in parallel chunks"""
        query = self._get_query_for_metric(metric_name)
        if not query:
            logging.warning(f"Unknown metric: {metric_name}")
            return []

        # Split into time chunks
        chunks = self._split_time_range(start_time, end_time)
        logging.info(
            f"Fetching {metric_name} in {len(chunks)} chunks "
            f"({self.chunk_hours}h each)"
        )

        # Fetch chunks in parallel with concurrency limit
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for chunk_start, chunk_end in chunks:
                task = self._fetch_metric_chunk_async(
                    session, query, chunk_start, chunk_end
                )
                tasks.append(task)

            # Execute with progress bar
            all_data = []
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"Fetching {metric_name}",
                unit="chunk",
            ):
                result = await coro
                if result:
                    all_data.extend(result)

        return all_data

    def _write_metric_data_streaming(
        self,
        metric_name: str,
        data: List[Dict],
        output_file: str,
    ) -> Tuple[int, int]:
        """Write metric data to CSV with streaming to minimize memory usage"""
        fieldnames = ["timestamp", "container_labels", "value", "container_id"]
        count = 0
        unique_containers = set()

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Process each time series
            for series in data:
                metric_labels = series.get("metric", {})
                values = series.get("values", [])

                container_labels = self.compress_labels(metric_labels)
                container_id = metric_labels.get("id", "")
                unique_containers.add(container_labels)

                # Stream write each data point
                for timestamp, value in values:
                    try:
                        numeric_value = float(value) if value != "NaN" else 0.0
                    except (ValueError, TypeError):
                        numeric_value = 0.0

                    row = {
                        "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                        "container_labels": container_labels,
                        "value": numeric_value,
                        "container_id": container_id,
                    }

                    writer.writerow(row)
                    count += 1

        return count, len(unique_containers)

    async def _export_metric_async(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        output_dir: str,
        timestamp_str: str,
    ) -> Optional[str]:
        """Export a single metric asynchronously"""
        logging.info(f"Starting export of {metric_name}")

        # Fetch data in parallel chunks
        data = await self._fetch_metric_parallel(metric_name, start_time, end_time)

        if not data:
            logging.warning(f"No data returned for {metric_name}")
            return None

        # Create output file
        output_path = os.path.join(os.path.dirname(__file__), output_dir)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{metric_name}_{timestamp_str}.csv")

        # Write data with streaming
        count, unique_containers = self._write_metric_data_streaming(
            metric_name, data, output_file
        )

        file_size_mb = os.path.getsize(output_file) / 1024 / 1024

        logging.info(
            f"✅ {metric_name}: {count:,} records, "
            f"{unique_containers} containers, {file_size_mb:.2f} MB"
        )

        return output_file

    def export_metrics(
        self, seconds=900, output_dir="data/raw/metrics", all=False
    ):
        """
        Export metrics with optimizations:
        - Async/parallel fetching
        - Time-range chunking
        - Container/namespace filtering
        - Streaming CSV writes
        """
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

        # Display export info
        print(f"\n📊 Exporting {len(self.metrics_to_export)} metrics:")
        print(f"   Metrics: {', '.join(self.metrics_to_export)}")
        print(f"   From: {start_time}")
        print(f"   To:   {end_time}")
        print(f"   Duration: {seconds} seconds ({seconds/3600:.2f} hours)")
        print(f"   Step interval: {self.step_interval}")
        print(f"   Chunk size: {self.chunk_hours} hour(s)")
        print(f"   Max concurrent requests: {self.max_concurrent}")

        if self.container_filter:
            print(f"   Container filter: {self.container_filter}")
        if self.namespace_filter:
            print(f"   Namespace filter: {self.namespace_filter}")
        if self.label_filters:
            print(f"   Label filters: {self.label_filters}")

        # Run async export
        return asyncio.run(
            self._export_all_metrics_async(
                start_time, end_time, output_dir, timestamp_str, all_metrics if all else None
            )
        )

    async def _export_all_metrics_async(
        self,
        start_time: datetime,
        end_time: datetime,
        output_dir: str,
        timestamp_str: str,
        all_metrics: Optional[List[str]] = None,
    ) -> List[str]:
        """Export all metrics in parallel"""
        metrics_to_process = all_metrics if all_metrics else self.metrics_to_export

        # Create tasks for all metrics
        tasks = []
        for metric_name in metrics_to_process:
            if metric_name not in self.metric_queries_base:
                logging.warning(f"Unknown metric '{metric_name}' - skipping")
                continue

            task = self._export_metric_async(
                metric_name, start_time, end_time, output_dir, timestamp_str
            )
            tasks.append(task)

        # Execute all tasks and collect results
        exported_files = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Exporting metrics",
            unit="metric",
        ):
            result = await coro
            if result:
                exported_files.append(result)

        print(f"\n✅ Export complete! {len(exported_files)} metrics exported")
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
        description="Export metrics from Kubernetes containers via Prometheus (optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export default metrics for 2 hours
  %(prog)s --seconds 7200

  # Export with container filter
  %(prog)s --container-filter "webapp|database" --seconds 7200

  # Export with namespace filter
  %(prog)s --namespace-filter "production" --seconds 3600

  # Large export with bigger chunks and more concurrency
  %(prog)s --seconds 21600 --chunk-hours 2 --max-concurrent 10 --step 30s

  # Filter by custom labels
  %(prog)s --label-filter app=myapp --label-filter env=prod
        """
    )

    # Time range
    parser.add_argument(
        "--seconds",
        type=int,
        default=900,
        help="Number of seconds to export (default: 900)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw/metrics",
        help="Output directory (default: ./data/raw/metrics)",
    )

    # Prometheus
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default="http://localhost:9090",
        help="Prometheus URL (default: http://localhost:9090)",
    )

    # Metrics selection
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=default_metrics,
        help="Metrics to export (space-separated)",
    )

    # Filters
    parser.add_argument(
        "--container-filter",
        type=str,
        help='Container name regex filter (e.g., "webapp|database")',
    )
    parser.add_argument(
        "--namespace-filter",
        type=str,
        help='Namespace regex filter (e.g., "production|staging")',
    )
    parser.add_argument(
        "--label-filter",
        action="append",
        dest="label_filters",
        metavar="KEY=VALUE",
        help='Label filter (e.g., app=myapp). Can be used multiple times.',
    )

    # Performance tuning
    parser.add_argument(
        "--chunk-hours",
        type=float,
        default=1.0,
        help="Split queries into chunks of this many hours (default: 1.0)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="15s",
        help="Prometheus step interval (default: 15s). Use larger for big exports (e.g., 30s, 60s)",
    )

    args = parser.parse_args()

    # Parse label filters
    label_filters = {}
    if args.label_filters:
        for label_filter in args.label_filters:
            if "=" in label_filter:
                key, value = label_filter.split("=", 1)
                label_filters[key] = value
            else:
                print(f"⚠️  Invalid label filter format: {label_filter}")

    # Show configuration
    print("📋 Export Configuration:")
    print(f"   Metrics: {len(args.metrics)} selected")
    for metric in args.metrics[:5]:
        print(f"      - {metric}")
    if len(args.metrics) > 5:
        print(f"      ... and {len(args.metrics) - 5} more")

    if args.container_filter:
        print(f"   🔍 Container filter: {args.container_filter}")
    if args.namespace_filter:
        print(f"   🔍 Namespace filter: {args.namespace_filter}")
    if label_filters:
        print(f"   🔍 Label filters: {label_filters}")

    print(f"\n⚡ Performance settings:")
    print(f"   Chunk size: {args.chunk_hours} hour(s)")
    print(f"   Max concurrent: {args.max_concurrent}")
    print(f"   Step interval: {args.step}")

    # Create exporter with all options
    exporter = KubernetesMetricsExporter(
        prometheus_url=args.prometheus_url,
        metrics=args.metrics,
        container_filter=args.container_filter,
        namespace_filter=args.namespace_filter,
        label_filters=label_filters,
        chunk_hours=args.chunk_hours,
        max_concurrent=args.max_concurrent,
        step_interval=args.step,
    )

    # Run export
    exporter.export_metrics(seconds=args.seconds, output_dir=args.output)


if __name__ == "__main__":
    main()
