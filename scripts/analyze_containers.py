#!/usr/bin/env python3
"""
Container discovery and analysis tool
Helps identify available containers for targeted metrics export
"""
import requests
import json
import re
from collections import defaultdict


def analyze_containers(prometheus_url="http://localhost:9090"):
    """Analyze available containers and their labeling"""
    print("🔍 Container Discovery and Analysis")
    print("=" * 50)
    
    try:
        # Test connection
        response = requests.get(f"{prometheus_url}/api/v1/status/config", timeout=5)
        response.raise_for_status()
        print("✅ Connected to Prometheus")
    except Exception as e:
        print(f"❌ Failed to connect to Prometheus: {e}")
        return
    
    # Get all container metrics
    print("\n📊 Analyzing container metrics...")
    
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query", 
                              params={'query': 'container_cpu_usage_seconds_total'}, 
                              timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if result['status'] != 'success':
            print("❌ Failed to query container metrics")
            return
        
        # Analyze containers
        containers = {}
        label_patterns = defaultdict(set)
        
        for series in result['data']['result']:
            metric = series['metric']
            container_id = metric.get('id', '')
            
            # Collect label patterns
            for key, value in metric.items():
                if key != '__name__':
                    label_patterns[key].add(value)
            
            # Analyze container ID patterns
            container_type = classify_container(container_id)
            
            containers[container_id] = {
                'type': container_type,
                'labels': metric,
                'name': metric.get('name', ''),
                'image': metric.get('image', ''),
            }
        
        print(f"📦 Found {len(containers)} containers")
        
        # Show container types
        type_counts = defaultdict(int)
        for info in containers.values():
            type_counts[info['type']] += 1
        
        print(f"\n🏷️  Container types:")
        for container_type, count in sorted(type_counts.items()):
            print(f"   {container_type}: {count} containers")
        
        # Show label structure
        print(f"\n📋 Available labels:")
        for label, values in sorted(label_patterns.items()):
            unique_values = len(values)
            sample_values = list(values)[:3]
            print(f"   {label}: {unique_values} unique values")
            if sample_values:
                print(f"      Examples: {', '.join(sample_values)}")
        
        # Show potential target containers
        print(f"\n🎯 Potential target containers:")
        target_keywords = ['webapp', 'web', 'app', 'db', 'database', 'postgres', 
                          'redis', 'cache', 'prometheus', 'grafana', 'cadvisor']
        
        targets_found = defaultdict(list)
        for cid, info in containers.items():
            container_text = f"{cid} {info['name']} {info['image']}".lower()
            
            for keyword in target_keywords:
                if keyword in container_text:
                    targets_found[keyword].append({
                        'id': cid,
                        'name': info['name'],
                        'image': info['image'],
                        'type': info['type']
                    })
        
        if targets_found:
            for keyword, matches in targets_found.items():
                print(f"\n   {keyword.upper()}:")
                for match in matches:
                    print(f"     ID: {match['id'][:60]}...")
                    if match['name']:
                        print(f"     Name: {match['name']}")
                    if match['image']:
                        print(f"     Image: {match['image']}")
                    print(f"     Type: {match['type']}")
        else:
            print("   No target containers found with standard naming")
        
        # Show container ID patterns for targeting
        print(f"\n🔧 Container ID patterns for filtering:")
        
        id_patterns = defaultdict(list)
        for cid in containers.keys():
            if 'kubepods' in cid:
                id_patterns['kubernetes'].append(cid)
            elif '/docker/' in cid:
                id_patterns['docker'].append(cid)
            elif cid in ['/', '/init.scope']:
                id_patterns['system'].append(cid)
            else:
                id_patterns['other'].append(cid)
        
        for pattern_type, ids in id_patterns.items():
            if ids:
                print(f"\n   {pattern_type.upper()} pattern:")
                print(f"     Count: {len(ids)}")
                if ids:
                    print(f"     Example: {ids[0]}")
                
                # Suggest filtering query
                if pattern_type == 'kubernetes':
                    print(f"     Filter: id=~\".*kubepods.*\"")
                elif pattern_type == 'docker':
                    print(f"     Filter: id=~\".*/docker/.*\"")
                elif pattern_type == 'podman_container':
                    print(f"     Filter: id=~\".*libpod_parent.*\"")
                elif pattern_type == 'system':
                    print(f"     Filter: id!=\"/\",id!=\"/init.scope\"")
        
        # Generate suggested queries
        print(f"\n💡 Suggested queries for your setup:")
        
        # Exclude system containers
        print(f"\n   Exclude system containers:")
        print(f"   rate(container_cpu_usage_seconds_total{{id!=\"/\",id!=\"/init.scope\",id!~\"/user.slice.*\"}}[1m])")
        
        # Target specific container types
        if 'kubernetes' in id_patterns:
            print(f"\n   Kubernetes containers only:")
            print(f"   rate(container_cpu_usage_seconds_total{{id=~\".*kubepods.*\"}}[1m])")
        
        if 'docker' in id_patterns:
            print(f"\n   Docker containers only:")
            print(f"   rate(container_cpu_usage_seconds_total{{id=~\".*/docker/.*\"}}[1m])")
        
        if 'podman_container' in id_patterns:
            print(f"\n   Podman containers only:")
            print(f"   rate(container_cpu_usage_seconds_total{{id=~\".*libpod_parent.*\"}}[1m])")
        
        # If we found target containers, show how to filter for them
        if targets_found:
            all_target_ids = []
            for matches in targets_found.values():
                for match in matches:
                    all_target_ids.append(match['id'])
            
            if all_target_ids:
                # Create regex pattern for target IDs
                escaped_ids = [re.escape(cid) for cid in all_target_ids[:5]]  # Limit to first 5
                print(f"\n   Target containers only:")
                print(f"   rate(container_cpu_usage_seconds_total{{id=~\"^({'|'.join(escaped_ids)})$\"}}[1m])")
        
    except Exception as e:
        print(f"❌ Error analyzing containers: {e}")


def classify_container(container_id):
    """Classify container based on ID pattern (updated for Podman)"""
    if container_id == '/':
        return 'system_root'
    elif '/init.scope' in container_id:
        return 'system_init'
    elif '/user.slice' in container_id:
        return 'user_systemd'
    elif '/libpod_parent' in container_id:
        return 'podman_container'
    elif 'kubepods' in container_id:
        if 'cri-containerd' in container_id:
            return 'kubernetes_container'
        else:
            return 'kubernetes_pod'
    elif '/docker/' in container_id:
        return 'docker_container'
    else:
        return 'unknown'


def suggest_export_strategy(prometheus_url="http://localhost:9090"):
    """Suggest the best export strategy based on available containers"""
    print("\n🎲 Export Strategy Recommendations")
    print("=" * 50)
    
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query", 
                              params={'query': 'container_cpu_usage_seconds_total'}, 
                              timeout=10)
        result = response.json()
        
        total_containers = len(result['data']['result'])
        k8s_containers = sum(1 for s in result['data']['result'] 
                            if 'kubepods' in s['metric'].get('id', ''))
        docker_containers = sum(1 for s in result['data']['result'] 
                               if '/docker/' in s['metric'].get('id', ''))
        system_containers = sum(1 for s in result['data']['result'] 
                               if s['metric'].get('id', '') in ['/', '/init.scope'])
        
        print(f"📊 Container breakdown:")
        print(f"   Total: {total_containers}")
        print(f"   Kubernetes: {k8s_containers}")
        print(f"   Docker: {docker_containers}")
        print(f"   System: {system_containers}")
        
        print(f"\n💡 Recommendations:")
        
        if k8s_containers > docker_containers:
            print(f"   ✅ Use Kubernetes-focused export")
            print(f"   ✅ Filter with: id=~\".*kubepods.*cri-containerd.*\"")
            print(f"   ✅ Expected data reduction: ~{(system_containers/total_containers)*100:.0f}%")
        elif docker_containers > 0:
            print(f"   ✅ Use Docker-focused export")
            print(f"   ✅ Filter with: id=~\".*/docker/.*\"")
            print(f"   ✅ Expected data reduction: ~{((total_containers-docker_containers)/total_containers)*100:.0f}%")
        else:
            print(f"   ⚠️  No clear container runtime detected")
            print(f"   ✅ Use system exclusion: id!=\"/\",id!=\"/init.scope\"")
        
        print(f"\n🎯 For your specific use case (webapp, db, cache, etc.):")
        print(f"   1. Use export_metrics_docker.py for Docker Compose")
        print(f"   2. Use export_metrics_targeted.py for Kubernetes")
        print(f"   3. Customize container_mapping in the scripts")
        
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze available containers for metrics export')
    parser.add_argument('--prometheus-url', type=str, default='http://localhost:9090',
                       help='Prometheus URL')
    
    args = parser.parse_args()
    
    analyze_containers(args.prometheus_url)
    suggest_export_strategy(args.prometheus_url)