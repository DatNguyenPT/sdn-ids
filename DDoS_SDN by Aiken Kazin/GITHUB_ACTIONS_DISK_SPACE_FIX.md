# GitHub Actions Disk Space Fix

## Problem

GitHub Actions runners have **limited disk space (14GB)**, and building multiple Docker images in parallel causes:
- ❌ "no space left on device" errors
- ❌ Docker build failures
- ❌ Workflow failures

## Root Cause

Building all Docker services (servers, workers, dashboard, MLflow) simultaneously:
- Each service builds its own image
- Multiple workers build in parallel
- Docker layers accumulate
- Disk space exhausted quickly

## Solution Applied

### 1. Sequential Building (Not Parallel)
- Build services one at a time
- Clean up after each build
- Reuse images when possible

### 2. Disk Space Management
- Clean Docker system before starting
- Prune build cache after each build
- Monitor disk usage

### 3. Optimized Build Strategy
- Build only what's needed
- Reuse existing images when possible
- Clean up unused resources

## Changes Made

### 1. Pre-build Cleanup
```yaml
- name: Clean Docker before starting
  run: |
    docker system prune -af --volumes || true
    docker builder prune -af || true
    df -h
```

### 2. Sequential Worker Building
```yaml
# Build workers one at a time
for worker in $WORKERS; do
  docker compose build --no-cache $worker
  docker compose up -d $worker
  docker builder prune -f || true  # Clean after each
done
```

### 3. Disk Space Monitoring
```yaml
# Check disk space at key points
df -h
```

## Expected Results

- ✅ Builds complete successfully
- ✅ Disk space managed efficiently
- ✅ No "no space left" errors
- ✅ Faster builds (less contention)

## Additional Optimizations (If Still Needed)

### Option 1: Use Smaller Base Images
- Use `python:3.9-slim` instead of full Python
- Multi-stage builds
- Remove unnecessary packages

### Option 2: Build Only Required Services
- Don't build all 4 model servers
- Build only the one needed for this matrix job
- Share common base images

### Option 3: Use Docker Layer Caching
- Cache layers between builds
- Use BuildKit cache mounts
- Reuse base layers

### Option 4: Reduce Parallel Jobs
- Run fewer matrix jobs simultaneously
- Use `max-parallel: 2` in strategy

## Monitoring

Check workflow logs for:
- Disk space reports (`df -h`)
- Build success/failure
- Docker cleanup messages

## If Issues Persist

1. **Reduce matrix parallelism**:
   ```yaml
   strategy:
     max-parallel: 2  # Build 2 models at a time
   ```

2. **Use pre-built images**:
   - Push images to Docker Hub
   - Pull instead of building

3. **Optimize Dockerfiles**:
   - Multi-stage builds
   - Smaller base images
   - Remove unnecessary dependencies

