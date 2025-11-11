#!/usr/bin/env python3
"""
Simple configurable metrics export for Docker Compose containers
Easy to customize for specific container targeting
"""
import requests
import csv
import os
from datetime import datetime, timedelta


# CONFIGURATION - Customize this for your setup
CONFIG = {
    'prometheus_url': 'http://localhost:9090',
    
    # Container identification patterns for Podman Compose
    'target_containers': {
        'webapp': ['webapp', 'web', 'app', 'flask', 'metrics-webapp'],
        'database': ['db', 'postgres', 'postgresql', 'mysql', 'metrics-db'],
        'cache': ['redis', 'cache', 'memcached', 'metrics-cache'],
        'prometheus': ['prometheus', 'prom', 'metrics-prometheus'],
        'grafana': ['grafana', 'metrics-grafana'],
        'cadvisor': ['cadvisor', 'metrics-cadvisor'],
        'load_gen': ['load', 'generator', 'test', 'metrics-load-generator']
    },
    
    # Metrics to collect
    'metrics': {
        'cpu_rate': 'rate(container_cpu_usage_seconds_total[1m])',
        'memory_usage': 'container_memory_usage_bytes',
        'memory_limit': 'container_spec_memory_limit_bytes',
        'disk_reads': 'rate(container_fs_reads_total[1m])',
        'disk_writes': 'rate(container_fs_writes_total[1m])',
        'network_rx': 'rate(container_network_receive_bytes_total[1m])',
        'network_tx': 'rate(container_network_transmit_bytes_total[1m])',
        'http_requests': 'rate(http_requests_total[1m])',
        'http_duration': 'rate(http_request_duration_seconds[1m])',
    },
    
    # Filters to exclude unwanted containers (updated for Podman)
    'exclude_patterns': [
        'id="/"',           # System root
        'id="/init.scope"', # System init
        'id=~"/user.slice.*"'  # User systemd slices
    ],
    
    # Export settings
    'step_interval': '15s',
    'default_duration': 900,  # 15 minutes
    'output_dir': '../data/raw'
}


def build_container_filter():
    """Build a Prometheus filter for target containers"""
    patterns = []
    
    # Add target container patterns
    for service, names in CONFIG['target_containers'].items():
        for name in names:
            patterns.extend([
                f'id=~".*{name}.*"',
                f'name=~".*{name}.*"',
                f'image=~".*{name}.*"'
            ])
    
    # Combine with OR logic
    include_filter = '|'.join(f'({p})' for p in patterns)
    
    # Add exclusion filters
    exclude_filter = ','.join(CONFIG['exclude_patterns'])
    
    if include_filter and exclude_filter:
        return f'{{{include_filter},{exclude_filter}}}'
    elif exclude_filter:
        return f'{{{exclude_filter}}}'
    else:
        return ''


def identify_container_service(container_id, container_name, image):
    """Identify which service a container belongs to (updated for Podman)"""
    # Combine all text for pattern matching
    text = f"{container_id} {container_name} {image}".lower()
    
    # Check each service pattern
    for service, patterns in CONFIG['target_containers'].items():
        if any(pattern in text for pattern in patterns):
            return service
    
    # Check for known system containers
    if container_id == '/':
        return 'system_root'
    elif '/init.scope' in container_id:
        return 'system_init'
    elif '/user.slice' in container_id:
        return 'user_systemd'
    elif '/libpod_parent' in container_id:
        # This is a Podman container - try to extract more info
        if 'libpod-' in container_id:
            # Extract container hash for identification
            import re
            match = re.search(r'libpod-([a-f0-9]{12})', container_id)
            if match:
                return f'podman_{match.group(1)}'
        return 'podman_unknown'
    elif 'kubepods' in container_id:
        return 'k8s_pod'  # Podman runs containers as pods
    
    return 'unknown'


def export_simple_targeted(duration_seconds=None, output_dir=None):
    """Export metrics with simple container targeting"""
    duration = duration_seconds or CONFIG['default_duration']
    output_path = output_dir or CONFIG['output_dir']
    
    print(f"🎯 Simple Targeted Metrics Export")
    print(f"   Duration: {duration} seconds")
    print(f"   Target services: {', '.join(CONFIG['target_containers'].keys())}")
    
    # Test Prometheus connection
    try:
        response = requests.get(f"{CONFIG['prometheus_url']}/api/v1/status/config", timeout=5)
        response.raise_for_status()
        print(f"   ✅ Connected to Prometheus")
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        return None
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=duration)
    
    print(f"   📅 Time range: {start_time} to {end_time}")
    
    # Build filter
    container_filter = build_container_filter()
    print(f"   🔍 Filter: {container_filter[:100]}...")
    
    all_data = []
    
    # Query each metric
    for metric_name, base_query in CONFIG['metrics'].items():
        print(f"\n📊 Fetching {metric_name}...")
        
        # Apply container filter to query
        if container_filter:
            query = base_query.replace('{', container_filter, 1) if '{' in base_query else f"{base_query}{container_filter}"
        else:
            query = base_query
        
        try:
            params = {
                'query': query,
                'start': int(start_time.timestamp()),
                'end': int(end_time.timestamp()),
                'step': CONFIG['step_interval']
            }
            
            response = requests.get(f"{CONFIG['prometheus_url']}/api/v1/query_range", 
                                   params=params, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if result['status'] != 'success':
                print(f"   ⚠️ Query failed: {result.get('error', 'unknown')}")
                continue
            
            data = result['data']['result']
            count = 0
            
            for series in data:
                metric_labels = series.get('metric', {})
                values = series.get('values', [])
                
                container_id = metric_labels.get('id', '')
                container_name = metric_labels.get('name', '')
                image = metric_labels.get('image', '')
                
                # Identify service
                service = identify_container_service(container_id, container_name, image)
                
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
                        'container_id': container_id,
                        'container_name': container_name,
                        'image': image,
                        'job': metric_labels.get('job', ''),
                        'instance': metric_labels.get('instance', ''),
                    }
                    
                    all_data.append(row)
                    count += 1
            
            print(f"   ✅ {count:,} data points from {len(data)} series")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if not all_data:
        print(f"\n❌ No data collected!")
        return None
    
    # Save to CSV
    os.makedirs(output_path, exist_ok=True)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_path, f'metrics_simple_{timestamp_str}.csv')
    
    fieldnames = ['timestamp', 'metric_name', 'value', 'service', 'container_id', 
                  'container_name', 'image', 'job', 'instance']
    
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
    for service, count in sorted(service_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_data)) * 100
        print(f"   {service}: {count:,} ({percentage:.1f}%)")
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple targeted metrics export')
    parser.add_argument('--seconds', type=int, default=CONFIG['default_duration'],
                       help=f'Duration in seconds (default: {CONFIG["default_duration"]})')
    parser.add_argument('--output', type=str, default=CONFIG['output_dir'],
                       help='Output directory')
    
    args = parser.parse_args()
    
    export_simple_targeted(args.seconds, args.output)