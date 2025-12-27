# Quick Start: Nexus Integration with Jenkins Pipeline

This guide provides a quick checklist to get Nexus integrated with your Jenkins CI/CD pipeline.

## Quick Setup Checklist

### Step 1: Prepare Nexus (5-10 minutes)

- [ ] Access Nexus at `http://localhost:8081/`
- [ ] Log in as admin
- [ ] Create 3 repositories:
  ```
  ✓ models-hosted (Raw Hosted)
  ✓ raw-hosted (Raw Hosted)  
  ✓ docker-hosted (Docker Hosted, Port 5000)
  ```
- [ ] Create Jenkins user with password: `Administration → Security → Users`
- [ ] Assign roles to Jenkins user: 
  ```
  nx-repository-admin-raw-*
  nx-repository-admin-docker-*
  ```

### Step 2: Configure Jenkins (3-5 minutes)

- [ ] Add credentials in Jenkins:
  ```
  Manage Jenkins → Manage Credentials
  - Add "nexus-credentials" (Username: jenkins)
  - Add "nexus-docker-credentials" (Username: jenkins)
  ```

- [ ] Verify environment variables in Jenkinsfile:
  ```groovy
  NEXUS_REPOSITORY_URL = "http://localhost:8081"
  DOCKER_REGISTRY_URL = "localhost:5000"
  ```

### Step 3: Configure Docker (1-2 minutes)

For Docker to push to insecure registry, configure daemon:

**Linux/Mac** (`/etc/docker/daemon.json`):
```json
{
  "insecure-registries": ["localhost:5000"]
}
```

**Windows** (Docker Desktop Settings → Docker Engine):
```json
{
  "insecure-registries": ["localhost:5000"]
}
```

Restart Docker after changes.

### Step 4: Test Connection (2-3 minutes)

Run test script:
```bash
cd "DDoS_SDN by Aiken Kazin"/scripts
chmod +x nexus_config.sh
./nexus_config.sh http://localhost:8081 jenkins YOUR_PASSWORD
```

Expected output:
```
✓ Nexus is reachable
✓ Authentication successful
✓ Found repositories:
  - docker-hosted
  - models-hosted
  - raw-hosted
```

### Step 5: Run Pipeline (Variable time)

1. Go to Jenkins and trigger build
2. Pipeline will now:
   - Build containers (Stage 1)
   - Run FL training (Stages 2-5)
   - **Sign artifacts** (Stage 6) ← NEW
   - **Upload to Nexus** (Stages 7-8) ← NEW
   - **Push Docker images** (Stage 9) ← NEW
   - **Generate report** (Stage 10) ← NEW

3. Check Nexus for artifacts:
   - Models: `http://localhost:8081/#browse/repositories/models-hosted`
   - Artifacts: `http://localhost:8081/#browse/repositories/raw-hosted`
   - Images: `http://localhost:5000/v2/_catalog` (Docker API)

---

## Detailed Repository Setup

### Repository 1: models-hosted

Purpose: Store trained ML models

1. **Administration** → **Repositories** → **Create repository**
2. Type: `raw (hosted)`
3. Name: `models-hosted`
4. Deployment Policy: `Allow Redeploy`
5. Save

Access: `http://localhost:8081/repository/models-hosted/`

### Repository 2: raw-hosted

Purpose: Store artifacts (logs, metadata, checksums, reports)

1. **Administration** → **Repositories** → **Create repository**
2. Type: `raw (hosted)`
3. Name: `raw-hosted`
4. Deployment Policy: `Allow Redeploy`
5. Save

Access: `http://localhost:8081/repository/raw-hosted/`

### Repository 3: docker-hosted

Purpose: Store Docker images

1. **Administration** → **Repositories** → **Create repository**
2. Type: `docker (hosted)`
3. Name: `docker-hosted`
4. **HTTP**: Enable, Port: `5000`
5. Deployment Policy: `Allow Redeploy`
6. Save

Access (API): `http://localhost:5000`

---

## Jenkins Credentials Setup

### Credential 1: nexus-credentials

For uploading artifacts and models:

1. **Jenkins** → **Manage Jenkins** → **Manage Credentials**
2. **Jenkins** → **Global credentials** → **Add Credentials**
3. Kind: `Username with password`
4. Username: `jenkins`
5. Password: `<your_nexus_password>`
6. ID: `nexus-credentials` (important!)
7. Save

### Credential 2: nexus-docker-credentials

For pushing Docker images:

1. Same steps as above, but:
2. ID: `nexus-docker-credentials`
3. Save

---

## Docker Daemon Configuration

### Option 1: Edit daemon.json

**Location:**
- Linux: `/etc/docker/daemon.json`
- Mac: `~/.docker/daemon.json`
- Windows: `%programdata%\docker\config\daemon.json`

**Content:**
```json
{
  "insecure-registries": [
    "localhost:5000",
    "127.0.0.1:5000"
  ]
}
```

**Restart Docker:**
```bash
# Linux
sudo systemctl restart docker

# Mac
osascript -e 'quit app "Docker"'
sleep 2
open -a Docker

# Windows
Restart-Service Docker
```

### Option 2: Docker Desktop GUI (Windows/Mac)

1. Open Docker Desktop
2. Click ⚙️ Settings
3. Go to **Docker Engine**
4. Add to JSON:
   ```json
   "insecure-registries": ["localhost:5000"]
   ```
5. Apply & Restart

---

## Testing the Integration

### Test 1: Nexus Connection

```bash
curl -u jenkins:password http://localhost:8081/service/rest/v1/status
```

Expected: `{"status":"ok"}`

### Test 2: Upload Test File

```bash
# Create test file
echo "test content" > test.txt

# Upload to raw-hosted
curl -v -u jenkins:password \
  --upload-file test.txt \
  http://localhost:8081/repository/raw-hosted/test.txt
```

Expected: HTTP 201 Created

### Test 3: Docker Registry

```bash
# Login
docker login localhost:5000 -u jenkins

# Test push
docker tag nginx:latest localhost:5000/test:latest
docker push localhost:5000/test:latest

# Test pull
docker pull localhost:5000/test:latest
```

Expected: Image successfully pushed/pulled

### Test 4: Helper Scripts

```bash
cd "DDoS_SDN by Aiken Kazin"/scripts

# Test connection
./nexus_config.sh http://localhost:8081 jenkins password

# Sign a model
./sign_artifacts.sh ../models ../signatures

# Upload using helper
python3 nexus_helper.py \
  --nexus-url http://localhost:8081 \
  --username jenkins \
  --password password \
  upload \
  --file ../models/LSTM_FL.h5 \
  --repository models-hosted
```

---

## Pipeline Output Locations

After pipeline runs successfully:

| Artifact | Location in Nexus |
|----------|------------------|
| Models | `http://localhost:8081/repository/models-hosted/lstm-fl-{BUILD_ID}.h5` |
| Metadata | `http://localhost:8081/repository/raw-hosted/fl-pipeline/artifacts/{BUILD_ID}/model-metadata.json` |
| Checksums | `http://localhost:8081/repository/raw-hosted/fl-pipeline/artifacts/{BUILD_ID}/checksums.sha256` |
| Logs | `http://localhost:8081/repository/raw-hosted/fl-pipeline/artifacts/{BUILD_ID}/logs-{BUILD_ID}.tar.gz` |
| Build Report | `http://localhost:8081/repository/raw-hosted/fl-pipeline/reports/build-report-{BUILD_ID}.json` |
| Docker Images | `localhost:5000/flower-server:{BUILD_ID}` |

---

## Troubleshooting

### Problem: "401 Unauthorized" when uploading

**Solution:**
```bash
# Verify credentials
curl -u jenkins:password http://localhost:8081/

# If fails, check:
1. Username/password is correct
2. User exists in Nexus
3. User has correct roles
4. Credentials ID in Jenkins matches (nexus-credentials)
```

### Problem: Docker push fails with "unknown error"

**Solution:**
```bash
# Check Docker daemon config
docker info | grep -i insecure

# Restart Docker
# (See Docker Daemon Configuration section above)

# Test with curl
curl http://localhost:5000/v2/_catalog
```

### Problem: Repository not found (404)

**Solution:**
```bash
# List all repositories
curl -u jenkins:password \
  http://localhost:8081/service/rest/v1/repositories

# Verify repository names in Jenkinsfile match Nexus
# Common issue: typo in repository name
```

### Problem: Nexus out of disk space

**Solution:**
1. Go to **Administration** → **Repository** → **Cleanup Policies**
2. Create policy to delete old artifacts
3. Or manually delete old artifacts
4. Or expand disk storage

---

## Best Practices

1. **Use API Tokens** instead of passwords:
   ```bash
   Jenkins User → Account → API token
   ```

2. **Enable SSL/TLS** in production:
   - Use reverse proxy (nginx)
   - Point to `https://your-domain:8081`

3. **Monitor disk usage**:
   - Set cleanup policies
   - Monitor in **Administration** → **System** → **Status**

4. **Backup regularly**:
   - **Administration** → **System** → **Backup**

5. **Use repository quotas**:
   - Limit disk usage per project

6. **Log activities**:
   - **Administration** → **Security** → **Audit**

---

## Next Steps

1. ✅ Complete all steps above
2. ✅ Run first successful pipeline
3. ✅ Verify artifacts in Nexus
4. Configure backup strategy
5. Set up SSL/TLS for production
6. Configure user roles and permissions
7. Set up cleanup policies
8. Integrate with monitoring/alerting

---

## Reference Documentation

- [Nexus Full Setup Guide](./NEXUS_SETUP.md)
- [Jenkinsfile](./Jenkinsfile)
- Helper Scripts:
  - `upload_to_nexus.sh` - Upload single file
  - `batch_upload_nexus.sh` - Batch upload from config
  - `sign_artifacts.sh` - Sign and checksum files
  - `nexus_config.sh` - Test configuration
  - `nexus_helper.py` - Python utility (test, list, upload, sign)

---

## Support

For issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [Nexus documentation](https://help.sonatype.com/repomanager3)
3. Check Jenkins pipeline logs
4. Verify all configuration steps completed
