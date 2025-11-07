#!/usr/bin/env python3
"""
Docker Compose specific metrics export
Optimized for the predictive-autoscaling Docker Compose setup
"""
import json
import requests
from datetime import datetime, timedelta
import csv
import os
import re


class DockerComposeMetricsExporter:
    def __init__(self, prometheus_url="http://localhost:9090"):
        self.prom_url = prometheus_url
        
        # These are the containers we care about in Docker Compose
        self.target_containers = [
            'webapp', 'web', 'app',
            'db', 'database', 'postgres', 'postgresql',
            'cache', 'redis',
            'prometheus', 'prom',
            'grafana',
            'cadvisor'
        ]
        
        # Metrics we want to focus on
        self.target_metrics = {
            'cpu_usage': 'rate(container_cpu_usage_seconds_total{id!="/",id!="/init.scope"}[1m])',
            'memory_usage': 'container_memory_usage_bytes{id!="/",id!="/init.scope"}',
            'memory_limit': 'container_spec_memory_limit_bytes{id!="/",id!="/init.scope"}',
            'disk_reads': 'rate(container_fs_reads_total{id!="/",id!="/init.scope"}[1m])',
            'disk_writes': 'rate(container_fs_writes_total{id!="/",id!="/init.scope"}[1m])',
            'network_rx': 'rate(container_network_receive_bytes_total{id!="/",id!="/init.scope"}[1m])',
            'network_tx': 'rate(container_network_transmit_bytes_total{id!="/",id!="/init.scope"}[1m])',
            'http_requests': 'rate(http_requests_total[1m])',
            'http_duration_p95': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))',
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
    
    def discover_containers(self):
        """Discover available containers and their characteristics"""
        print("🔍 Discovering containers...")
        
        try:
            # Get all container CPU metrics to see what's available
            response = requests.get(f"{self.prom_url}/api/v1/query", 
                                   params={'query': 'container_cpu_usage_seconds_total'}, 
                                   timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result['status'] != 'success':
                print("❌ Failed to query containers")
                return {}
            
            containers = {}
            for series in result['data']['result']:
                metric = series['metric']
                container_id = metric.get('id', '')
                
                # Skip system containers
                if container_id in ['/', '/init.scope'] or 'kubepods.slice' in container_id:
                    continue
                
                # Analyze the container
                container_info = self._analyze_container(container_id, metric)
                if container_info:
                    containers[container_id] = container_info
            
            print(f"📦 Found {len(containers)} containers:")
            for cid, info in containers.items():
                print(f"   {info['service_name']} -> {cid[:60]}...")
            
            return containers
            
        except Exception as e:
            print(f"❌ Error discovering containers: {e}")
            return {}
    
    def _analyze_container(self, container_id, metric_labels):
        """Analyze a container to determine its service type"""
        container_name = metric_labels.get('name', '')
        image = metric_labels.get('image', '')
        
        # Try to identify the service
        service_name = 'unknown'
        
        # Check container name patterns
        for target in self.target_containers:
            if target in container_name.lower() or target in container_id.lower():
                service_name = target
                break
        
        # Check image patterns  
        if service_name == 'unknown' and image:
            image_lower = image.lower()
            if 'postgres' in image_lower:
                service_name = 'database'
            elif 'redis' in image_lower:
                service_name = 'cache'
            elif 'prometheus' in image_lower or 'prom/' in image_lower:
                service_name = 'prometheus'
            elif 'grafana' in image_lower:
                service_name = 'grafana'
            elif 'cadvisor' in image_lower:
                service_name = 'cadvisor'
            elif any(web in image_lower for web in ['flask', 'python', 'nginx', 'apache']):
                service_name = 'webapp'
        
        # Extract container hash for identification
        container_hash = ''
        if 'cri-containerd-' in container_id:
            match = re.search(r'cri-containerd-([a-f0-9]{12})', container_id)
            if match:
                container_hash = match.group(1)
        
        return {
            'service_name': service_name,
            'container_name': container_name,
            'image': image,
            'container_hash': container_hash,
            'is_target': service_name in self.target_containers
        }
    
    def export_metrics(self, seconds=900, output_dir='../data/raw', target_only=True):
        """Export metrics with container filtering"""
        if not self.test_connection():
            return None
        
        # Discover containers
        containers = self.discover_containers()
        
        if target_only:
            # Filter to only target containers
            target_container_ids = [
                cid for cid, info in containers.items() 
                if info['is_target']
            ]
            print(f"🎯 Targeting {len(target_container_ids)} containers")
        else:
            target_container_ids = list(containers.keys())
            print(f"📊 Exporting all {len(target_container_ids)} containers")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=seconds)
        
        print(f"\n📊 Exporting metrics:")
        print(f"   From: {start_time}")
        print(f"   To:   {end_time}")
        print(f"   Duration: {seconds} seconds")
        
        all_data = []
        
        for metric_name, base_query in self.target_metrics.items():
            print(f"\n📥 Fetching: {metric_name}")
            
            try:
                params = {
                    'query': base_query,
                    'start': int(start_time.timestamp()),
                    'end': int(end_time.timestamp()),
                    'step': '15s'
                }
                
                response = requests.get(f"{self.prom_url}/api/v1/query_range", 
                                       params=params, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                
                if result['status'] != 'success':
                    print(f"   ⚠️  Query failed: {result.get('error', 'unknown error')}")
                    continue
                
                data = result['data']['result']
                
                if not data:
                    print(f"   ⚠️  No data returned")
                    continue
                
                # Process each time series
                count = 0
                filtered_count = 0
                
                for series in data:
                    metric_labels = series.get('metric', {})
                    values = series.get('values', [])
                    
                    container_id = metric_labels.get('id', '')
                    
                    # Filter by target containers if requested
                    if target_only and container_id not in target_container_ids:
                        filtered_count += 1
                        continue
                    
                    container_info = containers.get(container_id, {})
                    
                    for timestamp, value in values:
                        try:
                            numeric_value = float(value) if value != 'NaN' else 0.0
                        except (ValueError, TypeError):
                            numeric_value = 0.0
                        
                        row = {
                            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                            'metric_name': metric_name,
                            'value': numeric_value,
                            'service_name': container_info.get('service_name', 'unknown'),
                            'container_id': container_id,
                            'container_name': container_info.get('container_name', ''),
                            'container_image': container_info.get('image', ''),
                            'container_hash': container_info.get('container_hash', ''),
                        }
                        
                        # Add selected original labels
                        for k, v in metric_labels.items():
                            if k in ['job', 'instance', 'device', 'interface']:
                                row[f'label_{k}'] = v
                        
                        all_data.append(row)
                        count += 1
                
                print(f"   ✅ Collected {count:,} points from {len(data)} series")
                if filtered_count > 0:
                    print(f"   🚫 Filtered out {filtered_count} non-target series")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        if not all_data:
            print("\n❌ No data collected!")
            return None
        
        # Create output directory
        output_path = os.path.join(os.path.dirname(__file__), output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Save to CSV
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = 'targeted' if target_only else 'all'
        output_file = os.path.join(output_path, f'metrics_docker_{suffix}_{timestamp_str}.csv')
        
        # Get all unique keys
        all_keys = set()
        for row in all_data:
            all_keys.update(row.keys())
        all_keys = sorted(all_keys)
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(all_data)
        
        file_size_mb = os.path.getsize(output_file) / 1024 / 1024
        
        print(f"\n✅ Export complete!")
        print(f"   Total records: {len(all_data):,}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Saved to: {output_file}")
        
        # Show summary by service
        service_counts = {}
        for row in all_data:
            service = row['service_name']
            service_counts[service] = service_counts.get(service, 0) + 1
        
        print(f"\n📈 Data points by service:")
        for service, count in sorted(service_counts.items()):
            percentage = (count / len(all_data)) * 100
            print(f"   {service}: {count:,} points ({percentage:.1f}%)")
        
        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Docker Compose container metrics')
    parser.add_argument('--seconds', type=int, default=900, 
                       help='Number of seconds to export (default: 900)')
    parser.add_argument('--output', type=str, default='../data/raw',
                       help='Output directory')
    parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090',
                       help='Prometheus URL')
    parser.add_argument('--all-containers', action='store_true',
                       help='Export all containers, not just targets')
    
    args = parser.parse_args()
    
    exporter = DockerComposeMetricsExporter(args.prometheus_url)
    exporter.export_metrics(
        seconds=args.seconds, 
        output_dir=args.output,
        target_only=not args.all_containers
    )


if __name__ == '__main__':
    main()