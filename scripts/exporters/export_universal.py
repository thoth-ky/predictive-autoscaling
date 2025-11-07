#!/usr/bin/env python3
"""
Universal container metrics export
Works with any container runtime (Docker, Podman, Kubernetes) 
with smart pattern matching and manual container mapping
"""
import requests
import csv
import os
import re
from datetime import datetime, timedelta


# MANUAL CONTAINER MAPPING
# Update this based on your actual containers
KNOWN_CONTAINERS = {
    # From your podman.inspect, the webapp container:
    '67f472e6d7e8': 'webapp',  # metrics-webapp
    
    # The large Podman container found in metrics (likely your compose stack):
    'f9f452510256': 'podman_compose_stack',  # Main compose container
    
    # Add more as you discover them:
    # 'container_hash': 'service_name'
    # You can find these by running: podman ps --format "{{.ID}} {{.Names}}"
}


class UniversalMetricsExporter:
    def __init__(self, prometheus_url="http://localhost:9090"):
        self.prom_url = prometheus_url
        
        # Service categories we care about
        self.target_services = ['webapp', 'database', 'cache', 'prometheus', 'grafana', 'cadvisor', 'load_generator']
        
        # Metrics to collect
        self.metrics = {
            'cpu_rate': 'rate(container_cpu_usage_seconds_total{id!="/",id!="/init.scope"}[1m])',
            'memory_usage': 'container_memory_usage_bytes{id!="/",id!="/init.scope"}',
            'memory_limit': 'container_spec_memory_limit_bytes{id!="/",id!="/init.scope"}',
            'disk_reads': 'rate(container_fs_reads_total{id!="/",id!="/init.scope"}[1m])',
            'disk_writes': 'rate(container_fs_writes_total{id!="/",id!="/init.scope"}[1m])',
            'network_rx': 'rate(container_network_receive_bytes_total{id!="/",id!="/init.scope"}[1m])',
            'network_tx': 'rate(container_network_transmit_bytes_total{id!="/",id!="/init.scope"}[1m])',
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
    
    def identify_container_service(self, container_id, container_name='', image=''):
        """Identify service from container ID and optional metadata"""
        
        # Method 1: Check manual mapping first
        for known_hash, service in KNOWN_CONTAINERS.items():
            if known_hash in container_id:
                return service, f'manual_mapping_{known_hash}'
        
        # Method 2: Pattern matching in container ID
        text = f"{container_id} {container_name} {image}".lower()
        
        # Common patterns
        patterns = {
            'webapp': ['webapp', 'web', 'app', 'flask'],
            'database': ['db', 'postgres', 'postgresql', 'mysql'],
            'cache': ['redis', 'cache', 'memcached'],
            'prometheus': ['prometheus', 'prom'],
            'grafana': ['grafana'],
            'cadvisor': ['cadvisor'],
            'load_generator': ['load', 'generator']
        }
        
        for service, keywords in patterns.items():
            if any(keyword in text for keyword in keywords):
                return service, 'pattern_match'
        
        # Method 3: Container runtime identification
        if '/libpod_parent/libpod-' in container_id:
            # Extract container hash for manual identification
            match = re.search(r'libpod-([a-f0-9]{12})', container_id)
            if match:
                hash_part = match.group(1)
                return 'podman_unknown', f'podman_{hash_part}'
        elif 'kubepods' in container_id and 'cri-containerd' in container_id:
            match = re.search(r'cri-containerd-([a-f0-9]{12})', container_id)
            if match:
                hash_part = match.group(1)
                return 'k8s_unknown', f'k8s_{hash_part}'
        elif container_id in ['/', '/init.scope']:
            return 'system', 'system'
        elif '/user.slice' in container_id:
            return 'user_systemd', 'user'
        
        return 'unknown', 'unidentified'
    
    def export_metrics(self, duration_seconds=900, output_dir='../data/raw', target_only=False):
        """Export container metrics with smart identification"""
        if not self.test_connection():
            return None
        
        print(f"🎯 Universal Container Metrics Export")
        print(f"   Duration: {duration_seconds} seconds")
        print(f"   Known containers: {len(KNOWN_CONTAINERS)}")
        print(f"   Target services: {', '.join(self.target_services)}")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=duration_seconds)
        
        print(f"   📅 Time range: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
        
        all_data = []
        containers_found = set()
        
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
                    container_name = metric_labels.get('name', '')
                    image = metric_labels.get('image', '')
                    
                    # Identify the service
                    service, identifier = self.identify_container_service(container_id, container_name, image)
                    containers_found.add((container_id, service, identifier))
                    
                    # Apply filtering if requested
                    if target_only and service not in self.target_services:
                        filtered_count += 1
                        continue
                    
                    for timestamp, value in values:
                        try:
                            numeric_value = float(value) if value != 'NaN' else 0.0
                        except (ValueError, TypeError):
                            numeric_value = 0.0
                        
                        row = {
                            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                            'metric_name': metric_name,
                            'value': numeric_value,
                            'service': service,
                            'identifier': identifier,
                            'container_id': container_id,
                            'container_name': container_name,
                            'container_image': image,
                            'job': metric_labels.get('job', ''),
                            'instance': metric_labels.get('instance', ''),
                        }
                        
                        all_data.append(row)
                        count += 1
                
                print(f"   ✅ {count:,} points from {len(data)} series", end="")
                if filtered_count > 0:
                    print(f" (filtered {filtered_count})")
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
        output_file = os.path.join(output_dir, f'metrics_universal_{suffix}_{timestamp_str}.csv')
        
        fieldnames = ['timestamp', 'metric_name', 'value', 'service', 'identifier', 
                      'container_id', 'container_name', 'container_image', 'job', 'instance']
        
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
        
        # Show container discovery info
        print(f"\n🔍 Container discovery summary:")
        print(f"   Total unique containers: {len(containers_found)}")
        
        # Group by service
        by_service = {}
        for container_id, service, identifier in containers_found:
            if service not in by_service:
                by_service[service] = []
            by_service[service].append((container_id[:60], identifier))
        
        for service, containers in by_service.items():
            print(f"\n   {service.upper()} ({len(containers)} containers):")
            for container_id, identifier in containers[:3]:  # Show first 3
                print(f"     {identifier}: {container_id}...")
            if len(containers) > 3:
                print(f"     ... and {len(containers) - 3} more")
        
        # Show suggestions for unknown containers
        unknown_containers = [c for c in containers_found if c[1] in ['podman_unknown', 'k8s_unknown', 'unknown']]
        if unknown_containers:
            print(f"\n💡 To improve identification, add these to KNOWN_CONTAINERS:")
            print("KNOWN_CONTAINERS = {")
            for container_id, service, identifier in unknown_containers[:5]:
                if 'podman_' in identifier:
                    hash_part = identifier.replace('podman_', '')
                    print(f"    '{hash_part}': 'your_service_name',  # {container_id[:40]}...")
            print("}")
        
        return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal container metrics export')
    parser.add_argument('--seconds', type=int, default=900,
                       help='Duration in seconds (default: 900)')
    parser.add_argument('--output', type=str, default='../data/raw',
                       help='Output directory')
    parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090',
                       help='Prometheus URL')
    parser.add_argument('--target-only', action='store_true',
                       help='Export only target services')
    
    args = parser.parse_args()
    
    exporter = UniversalMetricsExporter(args.prometheus_url)
    exporter.export_metrics(
        duration_seconds=args.seconds,
        output_dir=args.output,
        target_only=args.target_only
    )


if __name__ == '__main__':
    main()