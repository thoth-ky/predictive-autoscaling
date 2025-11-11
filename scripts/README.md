# Scripts Directory

This directory contains scripts for exporting and processing metrics from the predictive autoscaling system.

## Quick Start - Choose Your Export Method

### 🐋 For Podman Compose (YOUR SETUP)

```bash
# Podman-optimized export with proper container identification
python3 export_metrics_podman.py --seconds 900

# Simple and configurable - edit the CONFIG section for your containers
python3 export_simple_targeted.py --seconds 900
```

### 🎯 For Docker Compose

```bash
# Advanced Docker Compose targeting 
python3 export_metrics_docker.py --seconds 900
```

### 🚀 For Kubernetes

```bash
# Advanced container discovery and labeling
python3 export_metrics_targeted.py --seconds 900
```

### 📊 For Everything (Legacy)

```bash
# Exports all metrics without filtering
python3 export_metrics_simple.py --seconds 900
```

## Export Scripts Overview

### � `export_metrics_podman.py` - **PODMAN OPTIMIZED**

**Perfect for:** Podman Compose setups (YOUR ENVIRONMENT)

**Features:**
- 🐋 Podman-specific container discovery
- 🏷️ Handles `/libpod_parent/` container paths
- 📋 Uses Podman Compose labels correctly
- 🎯 Targets your specific containers: webapp, database, cache, etc.
- 🚫 Filters out system and user systemd containers

### �🔧 `export_simple_targeted.py` - **EASIEST TO CUSTOMIZE**

**Perfect for:** Any setup, quick customization

**Features:**
- 📝 Easy configuration at the top of the file
- 🎯 Simple container pattern matching (updated for Podman)
- 🏷️ Clean service labeling (webapp, database, cache, etc.)
- 📊 Built-in export summary

### 🎯 `export_metrics_docker.py` - **DOCKER OPTIMIZED**

**Perfect for:** Docker Compose with advanced filtering needs

**Features:**
- 🐳 Docker-specific container discovery
- 🔍 Smart image and name analysis
- 🚫 Automatic system container exclusion
- � Detailed container breakdown

### 🚀 `export_metrics_targeted.py` - **KUBERNETES OPTIMIZED**

**Perfect for:** Kubernetes deployments, complex container hierarchies

**Features:**
- ☸️ Kubernetes-aware container parsing
- 🏷️ Advanced label processing
- 🎯 Regex-based container targeting
- 📊 Comprehensive metrics collection

### 📊 `export_metrics_simple.py` - **LEGACY ALL-DATA**

**Perfect for:** When you need everything, debugging, initial exploration

**Features:**
- 📈 Exports all available metrics
- 🔓 No filtering or targeting
- 📊 Raw Prometheus data
- 🗂️ Large output files

## Utility Scripts

### � `analyze_containers.py` - **CONTAINER DISCOVERY**

```bash
# Discover available containers and get targeting recommendations
python3 analyze_containers.py
```

**Use this to:**
- 🕵️ See what containers are available
- 📋 Understand label structures
- 💡 Get query suggestions for your setup
- 🎯 Identify container naming patterns

### 🎮 `demo_targeted_export.py` - **TEST & DEMO**

```bash
# Quick demo with analysis
python3 demo_targeted_export.py
```

## Container Targeting Guide

### Docker Compose Container Names

Your containers should match these patterns in ID, name, or image:

```yaml
# docker-compose.yml
services:
  webapp:
    container_name: metrics-webapp    # Matches 'webapp'
    image: flask:latest              # Matches 'webapp' via 'flask'
    
  db:
    container_name: metrics-db       # Matches 'database'
    image: postgres:15               # Matches 'database' via 'postgres'
    
  cache:
    container_name: metrics-cache    # Matches 'cache'
    image: redis:7                   # Matches 'cache' via 'redis'
```

### Container Label Patterns

The scripts look for these patterns in:
- Container ID paths
- Container names 
- Docker image names

**Webapp:** `webapp`, `web`, `app`, `flask`, `django`, `nginx`
**Database:** `db`, `database`, `postgres`, `mysql`, `mongodb`
**Cache:** `redis`, `cache`, `memcached`
**Monitoring:** `prometheus`, `grafana`, `cadvisor`

## Quick Start Scripts

### 🚀 One-line Export

```bash
# Use the shell wrapper
./run_targeted_export.sh 900    # Export last 15 minutes
./run_targeted_export.sh 3600   # Export last hour
```

## Output Formats

All exports create CSV files with these core columns:

```csv
timestamp,metric_name,value,service,container_id,container_name,image
2025-11-06T20:35:12,cpu_rate,0.85,webapp,/kubepods.slice/...,metrics-webapp,flask:latest
2025-11-06T20:35:12,memory_usage,536870912,database,/kubepods.slice/...,metrics-db,postgres:15
```

### Targeted Export Columns:
- `service` - Friendly name (webapp, database, cache)
- `container_id` - Full container path
- `container_name` - Original container name
- `image` - Docker image
- `job`, `instance` - Prometheus labels

## Example Usage & Analysis

### Quick Analysis

```python
import pandas as pd

# Load exported data
df = pd.read_csv('data/raw/metrics_simple_20251106_201831.csv')

# See what services were captured
print(df['service'].value_counts())

# Analyze webapp CPU over time
webapp_cpu = df[
    (df['service'] == 'webapp') & 
    (df['metric_name'] == 'cpu_rate')
].sort_values('timestamp')

print(f"Webapp CPU: {webapp_cpu['value'].mean():.4f} avg")

# Memory usage by service
memory_data = df[df['metric_name'] == 'memory_usage']
memory_by_service = memory_data.groupby('service')['value'].mean()
print(memory_by_service / 1024**3)  # Convert to GB
```

### Time Series Analysis

```python
# Convert timestamp and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Resample to 1-minute averages
webapp_metrics = df[df['service'] == 'webapp']
minutely = webapp_metrics.groupby(['metric_name']).resample('1T')['value'].mean()

# Plot CPU and memory trends
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# CPU rate
cpu_data = minutely.loc['cpu_rate']
cpu_data.plot(ax=ax1, title='Webapp CPU Rate')
ax1.set_ylabel('CPU Cores')

# Memory usage  
memory_data = minutely.loc['memory_usage'] / 1024**3
memory_data.plot(ax=ax2, title='Webapp Memory Usage')
ax2.set_ylabel('Memory (GB)')

plt.tight_layout()
plt.show()
```

## Configuration & Customization

### Customize Target Containers

Edit `export_simple_targeted.py`:

```python
CONFIG = {
    'target_containers': {
        'webapp': ['webapp', 'web', 'app', 'your-app-name'],
        'database': ['db', 'postgres', 'your-db-name'],
        'cache': ['redis', 'cache', 'your-cache-name'],
        # Add your custom services
        'api': ['api', 'rest', 'graphql'],
        'worker': ['worker', 'celery', 'background'],
    }
}
```

### Customize Metrics

```python
CONFIG = {
    'metrics': {
        'cpu_rate': 'rate(container_cpu_usage_seconds_total[1m])',
        'memory_usage': 'container_memory_usage_bytes',
        # Add custom metrics
        'cpu_throttle': 'rate(container_cpu_cfs_throttled_seconds_total[1m])',
        'disk_usage': 'container_fs_usage_bytes',
    }
}
```

## Monitoring Setup Verification

Make sure your monitoring stack is running:

```bash
cd ../local-setup

# Start the stack
docker-compose up -d

# Verify services
docker-compose ps

# Check Prometheus targets
curl http://localhost:9090/targets

# Check cAdvisor metrics
curl http://localhost:8080/metrics | head -20

# Check container names
docker-compose ps --format "table {{.Name}}\t{{.Image}}\t{{.Status}}"
```

## Troubleshooting

### No Data Exported
1. **Check services:** `docker-compose ps`
2. **Test Prometheus:** `curl http://localhost:9090/targets`
3. **Check cAdvisor:** `curl http://localhost:8080/metrics`
4. **Wait for data:** Let services run for 2-3 minutes

### No Target Containers Found
1. **Run analysis:** `python3 analyze_containers.py`
2. **Check container names:** `docker ps --format "table {{.Names}}\t{{.Image}}"`
3. **Customize patterns:** Edit CONFIG in export scripts
4. **Use Docker labels:** Add monitoring labels to docker-compose.yml

### Large File Sizes
1. **Use targeting:** Always use targeted exports
2. **Shorter duration:** Export 15-30 minutes instead of hours
3. **Fewer metrics:** Comment out unused metrics in CONFIG
4. **Filter containers:** Be more specific with container patterns

### Connection Issues
1. **Check URL:** Verify Prometheus is at `http://localhost:9090`
2. **Port mapping:** Check docker-compose.yml ports
3. **Firewall:** Ensure ports 9090, 8080 are accessible
4. **Custom URL:** Use `--prometheus-url` parameter

## File Size Guide

Typical file sizes for targeted export:

| Duration | Containers | File Size |
|----------|------------|-----------|
| 15 min   | 5 services | 50-200 KB |
| 1 hour   | 5 services | 200-800 KB |
| 24 hours | 5 services | 5-20 MB |
| 15 min   | All (unfiltered) | 2-10 MB |

Targeted exports are **10-50x smaller** than unfiltered exports!