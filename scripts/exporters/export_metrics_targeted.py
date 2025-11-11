#!/usr/bin/env python3
"""
Targeted metrics export for Docker Compose containers
Focuses on specific containers: webapp, db, cache, prometheus, grafana
"""
import json
import requests
from datetime import datetime, timedelta
import csv
import os
import re


class ContainerMetricsExporter:
    def __init__(self, prometheus_url="http://localhost:9090"):
        self.prom_url = prometheus_url
        self.target_containers = {
            'webapp': ['metrics-webapp', 'webapp'],
            'database': ['metrics-db', 'postgres', 'db'],
            'cache': ['metrics-cache', 'redis', 'cache'],
            'prometheus': ['metrics-prometheus', 'prometheus'],
            'grafana': ['metrics-grafana', 'grafana'],
            'cadvisor': ['metrics-cadvisor', 'cadvisor'],
            'load-generator': ['metrics-load-generator', 'load-generator']
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
    
    def get_container_mapping(self):
        """Get mapping of container IDs to readable names"""
        print("🔍 Discovering containers...")
        
        # Query for container labels to build mapping
        query = 'container_cpu_usage_seconds_total'
        
        try:
            response = requests.get(f"{self.prom_url}/api/v1/query", 
                                   params={'query': query}, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result['status'] != 'success':
                print("❌ Failed to query container metrics")
                return {}
            
            container_mapping = {}
            for series in result['data']['result']:
                metric = series['metric']
                container_id = metric.get('id', '')
                container_name = metric.get('name', '')
                image = metric.get('image', '')
                
                # Try to extract container name from different sources
                friendly_name = self._extract_container_name(container_id, container_name, image)
                
                if friendly_name:
                    container_mapping[container_id] = {
                        'name': friendly_name,
                        'original_name': container_name,
                        'image': image,
                        'id': container_id
                    }
            
            print(f"📦 Found {len(container_mapping)} containers:")
            for cid, info in container_mapping.items():
                print(f"   {info['name']} -> {cid[:50]}...")
            
            return container_mapping
            
        except Exception as e:
            print(f"❌ Error discovering containers: {e}")
            return {}
    
    def _extract_container_name(self, container_id, container_name, image):
        """Extract a friendly container name"""
        # First try container name if available
        if container_name:
            for service, names in self.target_containers.items():
                if any(name in container_name.lower() for name in names):
                    return service
        
        # Try image name
        if image:
            for service, names in self.target_containers.items():
                if any(name in image.lower() for name in names):
                    return service
        
        # Try to extract from Docker container ID pattern
        if '/docker/containers/' in container_id:
            # Extract container hash and try to map it
            match = re.search(r'/docker/containers/([a-f0-9]{12})', container_id)
            if match:
                return f"container_{match.group(1)}"
        
        # For Kubernetes containers, try to extract pod/container information
        if 'kubepods' in container_id and 'pod' in container_id:
            # Extract pod UID and try to identify service
            pod_match = re.search(r'pod([a-f0-9]{8}_[a-f0-9]{4}_[a-f0-9]{4}_[a-f0-9]{4}_[a-f0-9]{12})', container_id)
            if pod_match:
                pod_uid = pod_match.group(1).replace('_', '-')
                
                # Try to identify container by the cri-containerd part
                container_match = re.search(r'cri-containerd-([a-f0-9]{12})', container_id)
                if container_match:
                    container_hash = container_match.group(1)
                    
                    # For now, return a descriptive name
                    # In a real setup, you'd query Kubernetes API to get pod/container names
                    return f"k8s_container_{container_hash}"
        
        # For system containers, use a simple mapping
        if container_id == '/':
            return 'system_root'
        elif '/init.scope' in container_id:
            return 'system_init'
        elif '/libpod_parent' in container_id:
            return 'podman_container'
        elif 'kubepods' in container_id and container_id.count('/') <= 2:
            return 'k8s_system'
        
        return None
    
    def get_targeted_queries(self, container_mapping):
        """Build queries targeting specific containers"""
        # Get container IDs for our target containers
        target_ids = []
        for cid, info in container_mapping.items():
            if info['name'] in self.target_containers.keys():
                target_ids.append(cid)
        
        if not target_ids:
            print("⚠️  No target containers found, using all containers")
            # Fallback to container name filtering
            container_filter = '|'.join([
                name for names in self.target_containers.values() 
                for name in names
            ])
            id_filter = f'id=~".*({container_filter}).*"'
        else:
            # Create regex pattern for target container IDs
            escaped_ids = [re.escape(cid) for cid in target_ids]
            id_filter = f'id=~"^({"|".join(escaped_ids)})$"'
        
        queries = {
            'container_cpu_rate': f'rate(container_cpu_usage_seconds_total{{{id_filter}}}[1m])',
            'container_memory': f'container_memory_usage_bytes{{{id_filter}}}',
            'container_memory_limit': f'container_spec_memory_limit_bytes{{{id_filter}}}',
            'container_fs_reads': f'rate(container_fs_reads_total{{{id_filter}}}[1m])',
            'container_fs_writes': f'rate(container_fs_writes_total{{{id_filter}}}[1m])',
            'container_network_rx': f'rate(container_network_receive_bytes_total{{{id_filter}}}[1m])',
            'container_network_tx': f'rate(container_network_transmit_bytes_total{{{id_filter}}}[1m])',
            'http_requests': 'rate(http_requests_total[1m])',
            'http_request_duration': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))',
        }
        
        return queries
    
    def export_metrics(self, seconds=900, output_dir='../data/raw'):
        """Export targeted metrics"""
        if not self.test_connection():
            return None
        
        # Get container mapping
        container_mapping = self.get_container_mapping()
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=seconds)
        
        print(f"\n📊 Exporting metrics:")
        print(f"   From: {start_time}")
        print(f"   To:   {end_time}")
        print(f"   Duration: {seconds} seconds")
        
        # Get targeted queries
        queries = self.get_targeted_queries(container_mapping)
        
        all_data = []
        
        for metric_name, query in queries.items():
            print(f"\n📥 Fetching: {metric_name}")
            print(f"   Query: {query}")
            
            try:
                params = {
                    'query': query,
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
                for series in data:
                    metric_labels = series.get('metric', {})
                    values = series.get('values', [])
                    
                    container_id = metric_labels.get('id', '')
                    container_info = container_mapping.get(container_id, {})
                    
                    for timestamp, value in values:
                        try:
                            numeric_value = float(value) if value != 'NaN' else 0.0
                        except (ValueError, TypeError):
                            numeric_value = 0.0
                        
                        row = {
                            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                            'metric_name': metric_name,
                            'value': numeric_value,
                            'container_name': container_info.get('name', 'unknown'),
                            'container_id': container_id,
                            'container_image': container_info.get('image', ''),
                        }
                        
                        # Add original labels with prefix
                        for k, v in metric_labels.items():
                            if k not in ['id', 'name', 'image']:  # Avoid duplicates
                                row[f'label_{k}'] = v
                        
                        all_data.append(row)
                        count += 1
                
                print(f"   ✅ Collected {count:,} points from {len(data)} series")
                
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
        output_file = os.path.join(output_path, f'metrics_targeted_{timestamp_str}.csv')
        
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
        
        # Show summary by container
        container_counts = {}
        for row in all_data:
            container_name = row['container_name']
            container_counts[container_name] = container_counts.get(container_name, 0) + 1
        
        print(f"\n📈 Data points by container:")
        for container, count in sorted(container_counts.items()):
            percentage = (count / len(all_data)) * 100
            print(f"   {container}: {count:,} points ({percentage:.1f}%)")
        
        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Export targeted Prometheus metrics for Docker Compose containers')
    parser.add_argument('--seconds', type=int, default=900, 
                       help='Number of seconds to export (default: 900)')
    parser.add_argument('--output', type=str, default='../data/raw',
                       help='Output directory')
    parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090',
                       help='Prometheus URL')
    
    args = parser.parse_args()
    
    exporter = ContainerMetricsExporter(args.prometheus_url)
    exporter.export_metrics(seconds=args.seconds, output_dir=args.output)


if __name__ == '__main__':
    main()