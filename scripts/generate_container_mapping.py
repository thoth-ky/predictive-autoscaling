#!/usr/bin/env python3
"""
Generate container mapping for Podman containers
This helps identify which container IDs correspond to which services
"""
import subprocess
import json
import re


def get_podman_containers():
    """Get list of running Podman containers with their details"""
    try:
        # Run podman ps to get container info
        result = subprocess.run(['podman', 'ps', '--format', 'json'], 
                               capture_output=True, text=True, check=True)
        containers = json.loads(result.stdout)
        return containers
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error running 'podman ps': {e}")
        return []


def get_podman_inspect(container_id):
    """Get detailed inspect information for a container"""
    try:
        result = subprocess.run(['podman', 'inspect', container_id], 
                               capture_output=True, text=True, check=True)
        inspect_data = json.loads(result.stdout)
        return inspect_data[0] if inspect_data else None
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error inspecting container {container_id}: {e}")
        return None


def generate_container_mapping():
    """Generate mapping between Prometheus container IDs and Podman containers"""
    print("🔍 Discovering Podman containers...")
    
    containers = get_podman_containers()
    if not containers:
        print("❌ No containers found or podman command failed")
        return {}
    
    print(f"📦 Found {len(containers)} running containers")
    
    mapping = {}
    
    for container in containers:
        container_id = container.get('Id', '')
        names = container.get('Names', [])
        image = container.get('Image', '')
        
        if names:
            name = names[0]  # Primary name
        else:
            name = container_id[:12]
        
        # Get detailed info
        inspect_data = get_podman_inspect(container_id)
        if inspect_data:
            cgroup_path = inspect_data.get('State', {}).get('CgroupPath', '')
            labels = inspect_data.get('Config', {}).get('Labels', {})
            
            # Extract service info from labels
            service_name = labels.get('monitoring.name', '')
            service_type = labels.get('monitoring.type', '')
            compose_service = labels.get('podman.compose.service', '')
            
            # Determine service category
            if compose_service:
                if 'webapp' in compose_service or 'web' in compose_service:
                    service = 'webapp'
                elif 'db' in compose_service or 'postgres' in compose_service:
                    service = 'database'
                elif 'cache' in compose_service or 'redis' in compose_service:
                    service = 'cache'
                elif 'prometheus' in compose_service:
                    service = 'prometheus'
                elif 'grafana' in compose_service:
                    service = 'grafana'
                elif 'cadvisor' in compose_service:
                    service = 'cadvisor'
                elif 'load' in compose_service:
                    service = 'load_generator'
                else:
                    service = compose_service
            else:
                service = 'unknown'
            
            mapping[container_id] = {
                'name': name,
                'service': service,
                'image': image,
                'cgroup_path': cgroup_path,
                'compose_service': compose_service,
                'monitoring_name': service_name,
                'monitoring_type': service_type,
                'short_id': container_id[:12]
            }
            
            print(f"   {service}: {name} ({container_id[:12]})")
    
    return mapping


def generate_python_mapping(mapping):
    """Generate Python code for container mapping"""
    print(f"\n🐍 Python mapping code:")
    print("# Add this to your export script:")
    print("CONTAINER_MAPPING = {")
    
    for container_id, info in mapping.items():
        short_id = info['short_id']
        service = info['service']
        print(f"    '{short_id}': '{service}',  # {info['name']}")
    
    print("}")
    
    print(f"\n# Usage in _analyze_podman_container:")
    print("if container_hash.startswith(tuple(CONTAINER_MAPPING.keys())):")
    print("    for short_id, service_name in CONTAINER_MAPPING.items():")
    print("        if container_hash.startswith(short_id):")
    print("            service = service_name")
    print("            break")


def generate_prometheus_queries(mapping):
    """Generate Prometheus queries for target containers"""
    target_services = ['webapp', 'database', 'cache', 'prometheus', 'grafana', 'cadvisor']
    target_containers = []
    
    for container_id, info in mapping.items():
        if info['service'] in target_services:
            target_containers.append(container_id)
    
    if target_containers:
        print(f"\n📊 Prometheus queries for target containers:")
        
        # Create regex pattern for container IDs
        escaped_ids = [re.escape(f"/libpod_parent/libpod-{cid}") for cid in target_containers]
        pattern = "|".join(escaped_ids)
        
        print(f"\n# Target containers only:")
        print(f'rate(container_cpu_usage_seconds_total{{id=~"^({pattern})"}}[1m])')
        
        print(f"\n# Exclude system, include targets:")
        print(f'rate(container_cpu_usage_seconds_total{{id!="/",id!="/init.scope",id=~".*libpod_parent.*"}}[1m])')


def main():
    print("🐋 Podman Container Mapping Generator")
    print("=" * 50)
    
    mapping = generate_container_mapping()
    
    if mapping:
        generate_python_mapping(mapping)
        generate_prometheus_queries(mapping)
        
        print(f"\n✅ Generated mapping for {len(mapping)} containers")
        print("💡 Copy the CONTAINER_MAPPING to your export script for automatic service identification")
    else:
        print("❌ No container mapping generated")


if __name__ == '__main__':
    main()