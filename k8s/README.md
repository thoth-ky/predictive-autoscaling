# Kubernetes Deployment for Predictive Autoscaling

This directory contains Kubernetes manifests to deploy the predictive autoscaling monitoring stack.

## Prerequisites

1. **Kubernetes Cluster**: Minikube, Kind, or any Kubernetes cluster
2. **kubectl**: Kubernetes CLI tool
3. **Docker**: To build container images

## Architecture

The deployment includes:
- **WebApp**: Flask application with metrics endpoints
- **PostgreSQL**: Database for persistent storage
- **Redis**: Cache layer
- **Prometheus**: Metrics collection and storage
- **cAdvisor**: Container metrics (DaemonSet)
- **Grafana**: Metrics visualization
- **Load Generator**: Continuous traffic generator

## Quick Start

### 1. Build Docker Images

First, build the required Docker images:

```powershell
# Build webapp image
cd ../local-setup/webapp
docker build -t metrics-webapp:latest .

# Build load generator image
cd ..
docker build -t load-generator:latest -f load_generator.Dockerfile .
```

If using Minikube, load images into Minikube:
```powershell
minikube image load metrics-webapp:latest
minikube image load load-generator:latest
```

### 2. Deploy to Kubernetes

Deploy all components in order:

```powershell
# Create namespace
kubectl apply -f namespace.yaml

# Deploy databases and cache
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n predictive-autoscaling --timeout=120s
kubectl wait --for=condition=ready pod -l app=redis -n predictive-autoscaling --timeout=120s

# Deploy monitoring stack
kubectl apply -f prometheus-rbac.yaml
kubectl apply -f prometheus-configmap.yaml
kubectl apply -f prometheus-deployment.yaml
kubectl apply -f cadvisor-daemonset.yaml
kubectl apply -f grafana-deployment.yaml

# Deploy application
kubectl apply -f webapp-deployment.yaml

# Deploy load generator
kubectl apply -f load-generator-deployment.yaml
```

### 3. Access Services

Get service URLs:

```powershell
# For Minikube
minikube service -n predictive-autoscaling webapp-nodeport --url
minikube service -n predictive-autoscaling prometheus-nodeport --url
minikube service -n predictive-autoscaling grafana-nodeport --url
```

Or use NodePort directly:
- **WebApp**: `http://<node-ip>:30500`
- **Prometheus**: `http://<node-ip>:30090`
- **Grafana**: `http://<node-ip>:30300` (admin/admin)

### 4. Port Forwarding (Alternative)

If NodePort is not accessible:

```powershell
# WebApp
kubectl port-forward -n predictive-autoscaling svc/webapp 5000:5000

# Prometheus
kubectl port-forward -n predictive-autoscaling svc/prometheus 9090:9090

# Grafana
kubectl port-forward -n predictive-autoscaling svc/grafana 3000:3000
```

## Verify Deployment

Check all pods are running:

```powershell
kubectl get pods -n predictive-autoscaling
```

Expected output:
```
NAME                              READY   STATUS    RESTARTS   AGE
cadvisor-xxxxx                    1/1     Running   0          2m
grafana-xxxxx                     1/1     Running   0          2m
load-generator-xxxxx              1/1     Running   0          1m
postgres-xxxxx                    1/1     Running   0          3m
prometheus-xxxxx                  1/1     Running   0          2m
redis-xxxxx                       1/1     Running   0          3m
webapp-xxxxx                      1/1     Running   0          1m
webapp-yyyyy                      1/1     Running   0          1m
```

Check services:

```powershell
kubectl get svc -n predictive-autoscaling
```

## Configuration

### Scaling WebApp

To scale the webapp deployment:

```powershell
kubectl scale deployment webapp -n predictive-autoscaling --replicas=5
```

### Changing Load Pattern

Edit the load-generator deployment to use different patterns:

```powershell
kubectl edit deployment load-generator -n predictive-autoscaling
```

Change the command to one of:
- `["python", "load_generator.py", "baseline", "--duration", "86400"]`
- `["python", "load_generator.py", "spike", "--duration", "300"]`
- `["python", "load_generator.py", "periodic", "--duration", "3600"]`
- `["python", "load_generator.py", "gradual", "--duration", "3600"]`
- `["python", "load_generator.py", "chaos", "--duration", "3600"]`

### Update Prometheus Configuration

Edit the ConfigMap:

```powershell
kubectl edit configmap prometheus-config -n predictive-autoscaling
```

Reload Prometheus:

```powershell
kubectl rollout restart deployment prometheus -n predictive-autoscaling
```

## Monitoring

### View Logs

```powershell
# WebApp logs
kubectl logs -n predictive-autoscaling -l app=webapp -f

# Load generator logs
kubectl logs -n predictive-autoscaling -l app=load-generator -f

# Prometheus logs
kubectl logs -n predictive-autoscaling -l app=prometheus -f
```

### Check Metrics

```powershell
# Test webapp metrics endpoint
kubectl port-forward -n predictive-autoscaling svc/webapp 5000:5000
curl http://localhost:5000/metrics
```

## Cleanup

Remove all resources:

```powershell
kubectl delete namespace predictive-autoscaling
```

Or delete individual components:

```powershell
kubectl delete -f load-generator-deployment.yaml
kubectl delete -f webapp-deployment.yaml
kubectl delete -f grafana-deployment.yaml
kubectl delete -f cadvisor-daemonset.yaml
kubectl delete -f prometheus-deployment.yaml
kubectl delete -f prometheus-configmap.yaml
kubectl delete -f prometheus-rbac.yaml
kubectl delete -f redis-deployment.yaml
kubectl delete -f postgres-deployment.yaml
kubectl delete -f namespace.yaml
```

## Troubleshooting

### Pods not starting

```powershell
kubectl describe pod <pod-name> -n predictive-autoscaling
```

### Image pull errors

If using Minikube, ensure images are loaded:

```powershell
minikube image ls | findstr metrics
```

### Database connection issues

Check if PostgreSQL is ready:

```powershell
kubectl exec -it -n predictive-autoscaling <postgres-pod> -- psql -U metrics -d metricsdb -c "\l"
```

### Prometheus not scraping

Check Prometheus targets:

```powershell
# Access Prometheus UI and go to Status -> Targets
```

Or check configuration:

```powershell
kubectl exec -it -n predictive-autoscaling <prometheus-pod> -- cat /etc/prometheus/prometheus.yml
```

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Operator](https://prometheus-operator.dev/)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [cAdvisor Documentation](https://github.com/google/cadvisor)

## Next Steps

1. Set up Horizontal Pod Autoscaling (HPA)
2. Configure Ingress for external access
3. Add custom Grafana dashboards
4. Implement predictive scaling based on collected metrics
5. Set up persistent storage classes for production
6. Configure resource quotas and limits
7. Add network policies for security
