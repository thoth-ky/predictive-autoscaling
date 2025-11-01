# PowerShell script to deploy the predictive-autoscaling stack to Kubernetes

param(
    [Parameter(Mandatory=$false)]
    [string]$Action = "deploy",
    
    [Parameter(Mandatory=$false)]
    [switch]$Minikube = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== Predictive Autoscaling K8s Deployment ===" -ForegroundColor Cyan

function Build-Images {
    Write-Host "`n[1/3] Building Docker images..." -ForegroundColor Yellow
    
    $currentDir = Get-Location
    $localSetupDir = Join-Path (Split-Path $PSScriptRoot -Parent) "local-setup"
    
    # Build webapp
    Write-Host "Building webapp image..." -ForegroundColor Green
    Set-Location (Join-Path $localSetupDir "webapp")
    docker build -t metrics-webapp:latest .
    if ($LASTEXITCODE -ne 0) { throw "Failed to build webapp image" }
    
    # Build load generator
    Write-Host "Building load-generator image..." -ForegroundColor Green
    Set-Location $localSetupDir
    docker build -t load-generator:latest -f load_generator.Dockerfile .
    if ($LASTEXITCODE -ne 0) { throw "Failed to build load-generator image" }
    
    Set-Location $currentDir
    
    if ($Minikube) {
        Write-Host "Loading images into Minikube..." -ForegroundColor Green
        minikube image load metrics-webapp:latest
        minikube image load load-generator:latest
    }
    
    Write-Host "Images built successfully!" -ForegroundColor Green
}

function Deploy-Stack {
    Write-Host "`n[2/3] Deploying to Kubernetes..." -ForegroundColor Yellow
    
    # Create namespace
    Write-Host "Creating namespace..." -ForegroundColor Green
    kubectl apply -f namespace.yaml
    
    # Deploy databases
    Write-Host "Deploying PostgreSQL..." -ForegroundColor Green
    kubectl apply -f postgres-deployment.yaml
    
    Write-Host "Deploying Redis..." -ForegroundColor Green
    kubectl apply -f redis-deployment.yaml
    
    # Wait for databases
    Write-Host "Waiting for databases to be ready..." -ForegroundColor Green
    kubectl wait --for=condition=ready pod -l app=postgres -n predictive-autoscaling --timeout=120s
    kubectl wait --for=condition=ready pod -l app=redis -n predictive-autoscaling --timeout=120s
    
    # Deploy monitoring
    Write-Host "Deploying Prometheus..." -ForegroundColor Green
    kubectl apply -f prometheus-rbac.yaml
    kubectl apply -f prometheus-configmap.yaml
    kubectl apply -f prometheus-deployment.yaml
    
    Write-Host "Deploying cAdvisor..." -ForegroundColor Green
    kubectl apply -f cadvisor-daemonset.yaml
    
    Write-Host "Deploying Grafana..." -ForegroundColor Green
    kubectl apply -f grafana-deployment.yaml
    
    # Deploy application
    Write-Host "Deploying WebApp..." -ForegroundColor Green
    kubectl apply -f webapp-deployment.yaml
    
    # Deploy load generator
    Write-Host "Deploying Load Generator..." -ForegroundColor Green
    kubectl apply -f load-generator-deployment.yaml
    
    Write-Host "Deployment complete!" -ForegroundColor Green
}

function Show-Status {
    Write-Host "`n[3/3] Checking deployment status..." -ForegroundColor Yellow
    
    Write-Host "`nPods:" -ForegroundColor Cyan
    kubectl get pods -n predictive-autoscaling
    
    Write-Host "`nServices:" -ForegroundColor Cyan
    kubectl get svc -n predictive-autoscaling
    
    if ($Minikube) {
        Write-Host "`n=== Access URLs (Minikube) ===" -ForegroundColor Cyan
        Write-Host "Run these commands to get service URLs:" -ForegroundColor Yellow
        Write-Host "  minikube service -n predictive-autoscaling webapp-nodeport --url" -ForegroundColor White
        Write-Host "  minikube service -n predictive-autoscaling prometheus-nodeport --url" -ForegroundColor White
        Write-Host "  minikube service -n predictive-autoscaling grafana-nodeport --url" -ForegroundColor White
    } else {
        Write-Host "`n=== Access URLs (NodePort) ===" -ForegroundColor Cyan
        Write-Host "WebApp:     http://<node-ip>:30500" -ForegroundColor White
        Write-Host "Prometheus: http://<node-ip>:30090" -ForegroundColor White
        Write-Host "Grafana:    http://<node-ip>:30300 (admin/admin)" -ForegroundColor White
    }
    
    Write-Host "`n=== Port Forwarding (Alternative) ===" -ForegroundColor Cyan
    Write-Host "kubectl port-forward -n predictive-autoscaling svc/webapp 5000:5000" -ForegroundColor White
    Write-Host "kubectl port-forward -n predictive-autoscaling svc/prometheus 9090:9090" -ForegroundColor White
    Write-Host "kubectl port-forward -n predictive-autoscaling svc/grafana 3000:3000" -ForegroundColor White
}

function Remove-Stack {
    Write-Host "`nRemoving all resources..." -ForegroundColor Yellow
    kubectl delete namespace predictive-autoscaling
    Write-Host "Cleanup complete!" -ForegroundColor Green
}

# Main execution
try {
    switch ($Action.ToLower()) {
        "build" {
            Build-Images
        }
        "deploy" {
            Build-Images
            Deploy-Stack
            Show-Status
        }
        "status" {
            Show-Status
        }
        "cleanup" {
            Remove-Stack
        }
        default {
            Write-Host "Unknown action: $Action" -ForegroundColor Red
            Write-Host "Valid actions: build, deploy, status, cleanup" -ForegroundColor Yellow
            exit 1
        }
    }
    
    Write-Host "`n=== Done! ===" -ForegroundColor Green
} catch {
    Write-Host "`nError: $_" -ForegroundColor Red
    exit 1
}
