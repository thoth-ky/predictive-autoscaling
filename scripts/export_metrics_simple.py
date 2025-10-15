#!/usr/bin/env python3
"""
Simplified metrics export from Prometheus - works without external dependencies
"""
import json
import requests
from datetime import datetime, timedelta
import csv
import os

def export_metrics_simple(seconds=900, output_dir='../data/raw'):
    """Export metrics using basic requests library"""
    
    prom_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    
    print(f"🔗 Connecting to Prometheus at {prom_url}...")
    
    # Test connection
    try:
        response = requests.get(f"{prom_url}/api/v1/status/config", timeout=5)
        response.raise_for_status()
        print("✅ Connected to Prometheus\n")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=seconds)
    
    print(f"📊 Exporting metrics:")
    print(f"   From: {start_time}")
    print(f"   To:   {end_time}")
    print(f"   Duration: {seconds} seconds\n")
    
    # Queries that include container name and other useful labels
    queries = {
        'container_cpu': 'rate(container_cpu_usage_seconds_total{name!=""}[1m])',
        'container_memory': 'container_memory_usage_bytes{name!=""}',
        'container_network_rx': 'rate(container_network_receive_bytes_total{name!=""}[1m])',
        'container_network_tx': 'rate(container_network_transmit_bytes_total{name!=""}[1m])',
        'http_requests': 'rate(http_requests_total[1m])',
        'http_request_duration': 'rate(http_request_duration_seconds_bucket[1m])',
        
    }
    
    all_data = []
    
    for metric_name, query in queries.items():
        print(f"📥 Fetching: {metric_name}")
        
        try:
            params = {
                'query': query,
                'start': int(start_time.timestamp()),
                'end': int(end_time.timestamp()),
                'step': '15s'
            }
            
            response = requests.get(f"{prom_url}/api/v1/query_range", 
                                   params=params, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if result['status'] != 'success':
                print(f"   ⚠️  Query failed")
                continue
            
            data = result['data']['result']
            
            if not data:
                print(f"   ⚠️  No data")
                continue
            
            # Process each time series
            count = 0
            for series in data:
                metric_labels = series.get('metric', {})
                values = series.get('values', [])
                
                for timestamp, value in values:
                    row = {
                        'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                        'metric_name': metric_name,
                        'value': float(value) if value != 'NaN' else 0.0,
                    }
                    # Add all labels as columns
                    row.update({f'label_{k}': v for k, v in metric_labels.items()})
                    all_data.append(row)
                    count += 1
            
            print(f"   ✅ Collected {count:,} points from {len(data)} series")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    if not all_data:
        print("\n❌ No data collected!")
        return
    
    # Create output directory
    output_path = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Save to CSV (simpler than parquet)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_path, f'metrics_{timestamp_str}.csv')
    
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
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Prometheus metrics to CSV')
    parser.add_argument('--seconds', type=int, default=900, 
                       help='Number of seconds to export (default: 900)')
    parser.add_argument('--output', type=str, default='../data/raw',
                       help='Output directory')
    
    args = parser.parse_args()
    
    export_metrics_simple(seconds=args.seconds, output_dir=args.output)
