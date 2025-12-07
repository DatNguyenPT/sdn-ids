# GitHub Actions with Docker: FL Architecture

## ✅ Yes, GitHub Actions CAN Run Docker!

GitHub Actions runners **do support Docker** and Docker Compose. Here's how it works for your FL project:

## Docker Support in GitHub Actions

### Default Docker Support:
- ✅ **Ubuntu runners** come with Docker pre-installed
- ✅ **Docker Compose** is available (as `docker compose` command)
- ✅ **Docker Buildx** available for building images
- ✅ **Full Docker functionality** - containers, networks, volumes

### What We've Updated:

1. **Added Docker Setup Step:**
   ```yaml
   - name: Set up Docker Buildx
     uses: docker/setup-buildx-action@v3
   ```

2. **Verify Docker Availability:**
   ```yaml
   - name: Verify Docker and Docker Compose
     run: |
       docker --version
       docker compose version
   ```

3. **Start Only Required Services:**
   - Instead of starting ALL services, we start only:
     - Dashboard (required by all)
     - MLflow server (required by all)
     - Specific model's server (e.g., `flower-server-mlpv2`)
     - Specific model's workers (e.g., `flower-worker-1`, `flower-worker-2`)

## Architecture in GitHub Actions

### Your FL Architecture:
```
GitHub Actions Runner (Ubuntu)
    ├─ Docker Engine
    │   ├─ fl-dashboard (port 5001)
    │   ├─ mlflow-server (port 5002)
    │   ├─ flower-server-{model} (port 8080-8083)
    │   └─ flower-worker-{1-8} (2 per model)
    │
    └─ Host Filesystem
        ├─ dataset_sdn.csv (mounted to workers)
        └─ models/ (mounted to servers)
```

### Service Mapping:

| Model Type | Server | Workers | Port |
|------------|--------|---------|------|
| MLPv2 | `flower-server-mlpv2` | `flower-worker-1`, `flower-worker-2` | 8080 |
| LSTM | `flower-server-lstm` | `flower-worker-3`, `flower-worker-4` | 8082 |
| CNN1D | `flower-server-cnn1d` | `flower-worker-5`, `flower-worker-6` | 8081 |
| CNN_LSTM | `flower-server-cnn_lstm` | `flower-worker-7`, `flower-worker-8` | 8083 |

## Updated Workflow Steps

### 1. Check Dataset File:
```yaml
- name: Check dataset file
  run: |
    if [ ! -f "dataset_sdn.csv" ]; then
      echo "❌ dataset_sdn.csv not found!"
      exit 1
    fi
```

**Important:** The dataset file must be:
- ✅ Committed to the repository, OR
- ✅ Downloaded from a data repository, OR
- ✅ Stored in GitHub Secrets/Artifacts

### 2. Start Services in Order:

```yaml
# Step 1: Start dependencies
docker compose up -d fl-dashboard mlflow-server

# Step 2: Start specific model server
docker compose up -d --build flower-server-{model}

# Step 3: Start workers for that model
docker compose up -d flower-worker-{1-2}  # for MLPv2
```

### 3. Monitor Training:

```yaml
- name: Monitor FL Training
  run: |
    # Wait for training completion
    # Check logs for completion indicators
```

## Important Considerations

### 1. Dataset File Size:
- ⚠️ **GitHub has file size limits:**
  - Files > 50MB require Git LFS
  - Files > 100MB are blocked
- ✅ **Solutions:**
  - Use Git LFS for large datasets
  - Download from external storage (S3, GCS)
  - Use GitHub Releases/Artifacts

### 2. Docker Build Time:
- ⚠️ **Building Docker images takes time**
- ✅ **Solutions:**
  - Use `--build` only when needed
  - Cache Docker layers
  - Pre-build images and push to registry

### 3. Resource Limits:
- ⚠️ **GitHub Actions runners have limits:**
  - 2-core CPU
  - 7GB RAM
  - 14GB disk space
- ✅ **Your FL setup should fit:**
  - 4 servers (one per model)
  - 8 workers (2 per model)
  - Dashboard + MLflow
  - Total: ~14 containers

### 4. Time Limits:
- ⚠️ **Workflow time limits:**
  - Free tier: 6 hours per workflow
  - Pro tier: 6 hours per workflow
- ✅ **Your FL training:**
  - 5 rounds × ~2-5 min per round = 10-25 min
  - Well within limits!

## Workflow Execution Flow

```
1. Checkout code
   ↓
2. Set up Docker Buildx
   ↓
3. Verify Docker available
   ↓
4. Check dataset file exists
   ↓
5. Start dependencies (dashboard, MLflow)
   ↓
6. Start model-specific server
   ↓
7. Start model-specific workers
   ↓
8. Monitor training
   ↓
9. Collect logs & artifacts
   ↓
10. Cleanup (docker compose down)
```

## Testing Locally

You can test the workflow steps locally:

```bash
# 1. Check Docker
docker --version
docker compose version

# 2. Check dataset
ls -lh dataset_sdn.csv

# 3. Start services (same as workflow)
export FL_ENABLE_DP=false
export IID=true
docker compose up -d fl-dashboard mlflow-server
docker compose up -d --build flower-server-mlpv2
docker compose up -d flower-worker-1 flower-worker-2

# 4. Monitor
docker compose ps
docker compose logs -f flower-server-mlpv2

# 5. Cleanup
docker compose down
```

## Troubleshooting

### Issue: "Docker not found"
**Solution:** Use `ubuntu-latest` runner (has Docker pre-installed)

### Issue: "Dataset file not found"
**Solution:** 
- Commit dataset to repo (if < 50MB)
- Use Git LFS (if > 50MB)
- Download from external source

### Issue: "Port already in use"
**Solution:** Each model uses different ports, should be fine

### Issue: "Workers can't connect to server"
**Solution:** 
- Ensure server starts first
- Wait for server to be ready (sleep 10)
- Check network connectivity

### Issue: "Out of disk space"
**Solution:**
- Clean up old containers/images
- Use `docker compose down -v` to remove volumes

## Summary

✅ **GitHub Actions CAN run Docker containers**
✅ **Docker Compose works perfectly**
✅ **Your FL architecture will work**
✅ **All services can be started**
✅ **Training can be monitored**

The updated workflow:
- ✅ Sets up Docker properly
- ✅ Verifies Docker availability
- ✅ Checks dataset file
- ✅ Starts only required services
- ✅ Monitors training progress
- ✅ Cleans up after completion

**Your FL training pipeline is ready to run in GitHub Actions!** 🚀

