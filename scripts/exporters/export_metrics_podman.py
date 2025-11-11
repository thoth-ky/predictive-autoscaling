#!/usr/bin/env python3
"""
Podman Compose metrics export
Optimized for Podman Compose containers with proper label handling
"""
import requests
import csv
import os
import re
from datetime import datetime, timedelta


class PodmanComposeMetricsExporter:
    def __init__(self, prometheus_url="http://localhost:9090"):
        self.prom_url = prometheus_url
        
        # Podman Compose containers we care about
        self.target_containers = {
            'webapp': ['metrics-webapp', 'webapp'],
            'database': ['metrics-db', 'db', 'postgres'],
            'cache': ['metrics-cache', 'cache', 'redis'],
            'prometheus': ['metrics-prometheus', 'prometheus'],
            'grafana': ['metrics-grafana', 'grafana'],
            'cadvisor': ['metrics-cadvisor', 'cadvisor'],
            'load_generator': ['metrics-load-generator', 'load-generator']
        }
        
        # Metrics focused on container performance
        self.metrics = {
            'cpu_rate': 'rate(container_cpu_usage_seconds_total{id!="/",id!="/init.scope",id!~"/user.slice.*"}[1m])',
            'memory_usage': 'container_memory_usage_bytes{id!="/",id!="/init.scope",id!~"/user.slice.*"}',
            'memory_limit': 'container_spec_memory_limit_bytes{id!="/",id!="/init.scope",id!~"/user.slice.*"}',
            'disk_reads': 'rate(container_fs_reads_total{id!="/",id!="/init.scope",id!~"/user.slice.*"}[1m])',
            'disk_writes': 'rate(container_fs_writes_total{id!="/",id!="/init.scope",id!~"/user.slice.*"}[1m])',
            'network_rx': 'rate(container_network_receive_bytes_total{id!="/",id!="/init.scope",id!~"/user.slice.*"}[1m])',
            'network_tx': 'rate(container_network_transmit_bytes_total{id!="/",id!="/init.scope",id!~"/user.slice.*"}[1m])',
            'http_requests': 'rate(http_requests_total[1m])',
            'http_duration': 'rate(http_request_duration_seconds[1m])',
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
    
    def discover_podman_containers(self):
        """Discover Podman containers and map them to services"""
        print("🔍 Discovering Podman containers...")
        
        try:
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
                container_name = metric.get('name', '')
                image = metric.get('image', '')
                
                # Skip system containers
                if self._is_system_container(container_id):
                    continue
                
                # Analyze the container
                service_info = self._analyze_podman_container(container_id, container_name, image)
                if service_info:
                    containers[container_id] = service_info
            
            print(f"📦 Found {len(containers)} Podman containers:")
            for cid, info in containers.items():
                container_short = cid[:60] + "..." if len(cid) > 60 else cid
                print(f"   {info['service']}: {info['name']} -> {container_short}")
            
            return containers
            
        except Exception as e:
            print(f"❌ Error discovering containers: {e}")
            return {}
    
    def _is_system_container(self, container_id):
        """Check if container is a system container to skip"""
        system_patterns = [
            '/',
            '/init.scope',
            '/user.slice',
            '/system.slice'
        ]
        return any(pattern in container_id for pattern in system_patterns)
    
    def _analyze_podman_container(self, container_id, container_name, image):
        """Analyze a Podman container to determine its service (updated for label-less metrics)"""
        # Since Podman container metrics don't include name/image labels,
        # we need to identify containers by their ID patterns
        
        service = 'unknown'
        
        # For Podman containers, we can only work with the container ID
        # Try to identify based on known patterns or make API calls
        
        # Check if this is a libpod container
        if '/libpod_parent/libpod-' in container_id:
            # Extract the container hash
            match = re.search(r'libpod-([a-f0-9]+)', container_id)
            if match:
                container_hash = match.group(1)
                
                # Try to identify the container by querying Podman API or using known patterns
                # For now, we'll mark it as a potential target and let the user identify manually
                service = f'podman_container'
                
                # You can map specific container hashes here if known:
                # Known container from inspect: 67f472e6d7e8614971228bf139cb5a32e9027ae2b326f430ee8dd0a2e2600488
                if container_hash.startswith('67f472e6d7e8'):
                    service = 'webapp'  # This is your metrics-webapp container
                # Add more mappings as needed
        
        # Extract Podman container info
        podman_id = ''
        if '/libpod_parent/libpod-' in container_id:
            match = re.search(r'libpod-([a-f0-9]+)', container_id)
            if match:
                podman_id = match.group(1)[:12]  # Short ID
        
        return {
            'service': service,
            'name': container_name or f'podman_{podman_id}',
            'image': image,
            'podman_id': podman_id,
            'is_target': service in self.target_containers.keys() or service == 'podman_container',
            'container_path': container_id
        }
    
    def export_metrics(self, duration_seconds=900, output_dir='../data/raw', target_only=True):
        """Export Podman container metrics"""
        if not self.test_connection():
            return None
        
        print(f"🎯 Podman Compose Metrics Export")
        print(f"   Duration: {duration_seconds} seconds")
        print(f"   Target services: {', '.join(self.target_containers.keys())}")
        
        # Discover containers
        containers = self.discover_podman_containers()
        
        if target_only:
            target_container_ids = [
                cid for cid, info in containers.items() 
                if info['is_target']
            ]
            print(f"   🎯 Targeting {len(target_container_ids)} containers")
        else:
            target_container_ids = list(containers.keys())
            print(f"   📊 Exporting all {len(target_container_ids)} containers")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=duration_seconds)
        
        print(f"   📅 Time range: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
        
        all_data = []
        
        for metric_name, query in self.metrics.items():
            print(f"\n📊 Fetching {metric_name}...")
            
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
                    print(f"   ⚠️ Query failed: {result.get('error', 'unknown')}")
                    continue
                
                data = result['data']['result']
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
                            'service': container_info.get('service', 'unknown'),
                            'container_name': container_info.get('name', ''),
                            'container_image': container_info.get('image', ''),
                            'podman_id': container_info.get('podman_id', ''),
                            'container_path': container_id,
                            'job': metric_labels.get('job', ''),
                            'instance': metric_labels.get('instance', ''),
                        }
                        
                        all_data.append(row)
                        count += 1
                
                print(f"   ✅ {count:,} points from {len(data)} series", end="")
                if filtered_count > 0:
                    print(f" (filtered {filtered_count} non-target)")
                else:
                    print()
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        if not all_data:
            print("\n❌ No data collected!")
            return None
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = 'targeted' if target_only else 'all'
        output_file = os.path.join(output_dir, f'metrics_podman_{suffix}_{timestamp_str}.csv')
        
        fieldnames = ['timestamp', 'metric_name', 'value', 'service', 'container_name', 
                      'container_image', 'podman_id', 'container_path', 'job', 'instance']
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        
        file_size_mb = os.path.getsize(output_file) / 1024 / 1024
        
        print(f"\n✅ Export complete!")
        print(f"   Records: {len(all_data):,}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Saved to: {output_file}")
        
        # Show breakdown by service
        service_counts = {}
        for row in all_data:
            service = row['service']
            service_counts[service] = service_counts.get(service, 0) + 1
        
        print(f"\n📈 Breakdown by service:")
        total_records = len(all_data)
        for service, count in sorted(service_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_records) * 100
            print(f"   {service}: {count:,} ({percentage:.1f}%)")
        
        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Podman Compose container metrics')
    parser.add_argument('--seconds', type=int, default=900,
                       help='Duration in seconds (default: 900)')
    parser.add_argument('--output', type=str, default='../data/raw',
                       help='Output directory')
    parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090',
                       help='Prometheus URL')
    parser.add_argument('--all-containers', action='store_true',
                       help='Export all containers, not just targets')
    
    args = parser.parse_args()
    
    exporter = PodmanComposeMetricsExporter(args.prometheus_url)
    exporter.export_metrics(
        duration_seconds=args.seconds,
        output_dir=args.output,
        target_only=not args.all_containers
    )


if __name__ == '__main__':
    main()