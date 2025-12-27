# Nexus Repository Setup Guide

This guide explains how to configure your self-hosted Nexus repository to support the enhanced Jenkins pipeline for storing models, artifacts, Docker images, and build reports.

## Prerequisites

- Nexus Repository Manager 3.x installed and running at `http://localhost:8081/`
- Admin access to Nexus
- Jenkins with Nexus integration

---

## 1. Create Repositories in Nexus

### 1.1 Models Repository (for ML models)

Create a **Raw Hosted Repository** for storing trained models:

1. Log in to Nexus as admin
2. Go to **Administration** → **Repositories** → **Create repository**
3. Select **raw (hosted)**
4. Configure:
   - **Name**: `models-hosted`
   - **Repository Path Suffix**: Leave empty (default)
   - **Deployment Policy**: `Allow Redeploy` (if you want to allow overwriting models)
   - **Storage**: Blob store: `default`
5. Click **Create repository**

### 1.2 Artifacts Repository (for logs, metadata, checksums)

Create another **Raw Hosted Repository** for artifacts:

1. Go to **Administration** → **Repositories** → **Create repository**
2. Select **raw (hosted)**
3. Configure:
   - **Name**: `raw-hosted`
   - **Repository Path Suffix**: Leave empty
   - **Deployment Policy**: `Allow Redeploy`
   - **Storage**: Blob store: `default`
4. Click **Create repository**

### 1.3 Docker Registry

Enable and configure Nexus Docker Registry:

1. Go to **Administration** → **Repositories** → **Create repository**
2. Select **docker (hosted)**
3. Configure:
   - **Name**: `docker-hosted`
   - **HTTP**: Enable and set port to **5000** (or your preferred port)
   - **Allow anonymous docker pull**: `false` (for security)
   - **Deployment Policy**: `Allow Redeploy`
   - **Storage**: Blob store: `default`
4. Click **Create repository**

**Important**: After creating the Docker registry, you need to:
- Note the HTTP port (default: 5000)
- Configure Docker daemon to allow insecure registry access (if using HTTP):

   On Linux/Mac, edit `/etc/docker/daemon.json`:
   ```json
   {
     "insecure-registries": ["localhost:5000"]
   }
   ```
   
   On Windows (Docker Desktop), go to **Settings** → **Docker Engine** and add:
   ```json
   {
     "insecure-registries": ["localhost:5000"]
   }
   ```

   Then restart Docker.

### 1.4 Proxy Repository for Maven (Optional, but recommended)

If you want to use Maven artifacts in the future:

1. Go to **Administration** → **Repositories** → **Create repository**
2. Select **maven2 (proxy)**
3. Configure:
   - **Name**: `maven-central`
   - **Remote storage**: `https://repo1.maven.org/maven2`
4. Click **Create repository**

---

## 2. Create User and Role for Jenkins

### 2.1 Create Nexus User

1. Go to **Administration** → **Security** → **Users**
2. Click **Create user**
3. Configure:
   - **ID**: `jenkins`
   - **First Name**: `Jenkins`
   - **Last Name**: `CI/CD`
   - **Email**: `jenkins@example.com`
   - **Password**: Set a strong password
   - **Status**: `Active`
   - **Roles**: Select roles (see 2.2 below)
4. Click **Create user**

### 2.2 Create Role for Jenkins

1. Go to **Administration** → **Security** → **Roles**
2. Click **Create role**
3. Configure:
   - **Role ID**: `jenkins-deployer`
   - **Role Name**: `Jenkins Deployer`
   - **Role Description**: `Jenkins CI/CD pipeline deployment role`

4. Under **Privileges**, add the following permissions:
   ```
   nx-repository-admin-raw-raw-hosted-*
   nx-repository-admin-raw-models-hosted-*
   nx-repository-admin-docker-docker-hosted-*
   nx-repository-view-raw-raw-hosted-*
   nx-repository-view-raw-models-hosted-*
   nx-repository-view-docker-docker-hosted-*
   ```

5. Click **Create role**

6. Go back to **Users** and edit the `jenkins` user
7. Add the `jenkins-deployer` role to the user
8. Save

---

## 3. Jenkins Configuration

### 3.1 Install Nexus Artifact Uploader Plugin

1. Go to Jenkins → **Manage Jenkins** → **Manage Plugins**
2. Search for `Nexus Artifact Uploader`
3. Install the plugin (with restart if required)

### 3.2 Configure Nexus Credentials in Jenkins

#### For Nexus HTTP Authentication:

1. Go to Jenkins → **Manage Jenkins** → **Manage Credentials**
2. Click **Jenkins** → **Global credentials**
3. Click **Add Credentials**
4. Configure:
   - **Kind**: `Username with password`
   - **Username**: `jenkins` (or your Nexus username)
   - **Password**: Your Nexus password
   - **ID**: `nexus-credentials`
   - **Description**: `Nexus Repository Credentials`
5. Click **Create**

#### For Docker Registry:

1. Repeat the above but set:
   - **ID**: `nexus-docker-credentials`
   - **Description**: `Nexus Docker Registry Credentials`
2. Click **Create**

### 3.3 Update Jenkinsfile Environment Variables

Ensure your Jenkinsfile has these environment variables set correctly:

```groovy
environment {
    NEXUS_REPOSITORY_URL = "http://localhost:8081"
    NEXUS_REPOSITORY_ID = "nexus-repo"
    DOCKER_REGISTRY_URL = "localhost:5000"  // or your Docker registry port
}
```

---

## 4. Nexus Security Configuration

### 4.1 API Token (Optional but recommended)

Instead of using plain passwords, create API tokens for Jenkins:

1. Log in to Nexus as the `jenkins` user
2. Click your **username** (top right) → **Account**
3. Click **Change password** or **API token**
4. Generate an **API token**
5. Use this token instead of the password in Jenkins credentials

### 4.2 LDAP/Active Directory Integration (Optional)

For enterprise environments, integrate with your directory service:

1. Go to **Administration** → **Security** → **LDAP**
2. Configure connection details to your LDAP server
3. Set up user and role mapping

---

## 5. Nexus Cleanup Policies (Optional)

To manage disk space, create cleanup policies:

1. Go to **Administration** → **Repository** → **Cleanup Policies**
2. Click **Create cleanup policy**
3. Configure:
   - **Name**: `old-models-cleanup`
   - **Criteria**:
     - **Last Downloaded**: `30 days`
     - **Release Type**: `Releases`
   - **Action**: `Delete`
4. Assign this policy to the `models-hosted` and `raw-hosted` repositories

---

## 6. Backup and Recovery

### 6.1 Configure Automatic Backups

1. Go to **Administration** → **System** → **Backup**
2. Set backup location (e.g., `/var/nexus-backups`)
3. Configure backup schedule (e.g., daily at 2 AM)
4. Enable **Include Database**

### 6.2 Test Restore Process

Periodically test restoring from backups to ensure integrity.

---

## 7. Accessing Artifacts

Once configured, you can access your artifacts:

### Models Repository:
```
http://localhost:8081/repository/models-hosted/
```

### Artifacts Repository:
```
http://localhost:8081/repository/raw-hosted/
```

### Docker Registry:
```
localhost:5000
```

### Browse Repository UI:
```
http://localhost:8081/#browse/repositories
```

---

## 8. Jenkins Pipeline Integration

The updated Jenkinsfile includes these stages:

1. **Stage 6**: Sign model artifacts (creates checksums and metadata)
2. **Stage 7**: Upload artifacts to Nexus `raw-hosted` repository
3. **Stage 8**: Upload models to Nexus `models-hosted` repository
4. **Stage 9**: Push Docker images to Nexus Docker Registry
5. **Stage 10**: Generate and upload build report

### Required Credentials in Jenkins:

- `nexus-credentials` - Username/password for HTTP uploads
- `nexus-docker-credentials` - Docker registry credentials

---

## 9. Troubleshooting

### Docker Push Fails with "unauthorized"

1. Verify credentials in Jenkins are correct
2. Check Nexus user permissions
3. Ensure Docker daemon is properly configured for insecure registries
4. Test manually:
   ```bash
   docker login localhost:5000 -u jenkins -p <password>
   docker push localhost:5000/flower-server:latest
   ```

### Artifact Upload Fails (401/403)

1. Verify Nexus user has correct roles assigned
2. Check credentials in Jenkins
3. Verify repository exists
4. Test curl command manually:
   ```bash
   curl -v -u jenkins:password http://localhost:8081/repository/raw-hosted/
   ```

### Repository Not Accessible

1. Check Nexus is running: `curl http://localhost:8081/`
2. Verify firewall allows access
3. Check logs in Nexus: `$NEXUS_HOME/sonatype-work/nexus3/log/nexus.log`

---

## 10. Best Practices

1. **Use API Tokens** instead of passwords in Jenkins
2. **Enable SSL/TLS** in production (use reverse proxy like nginx)
3. **Regular Backups** of Nexus database and artifacts
4. **Monitor Disk Space** and configure cleanup policies
5. **Use Repository Quotas** to limit artifact storage per project
6. **Implement Repository Staging** for release workflows
7. **Enable Audit Logging** for compliance requirements
8. **Use Read-Only Proxy** repositories for external dependencies

---

## 11. Additional Resources

- [Nexus Documentation](https://help.sonatype.com/repomanager3)
- [Docker Registry in Nexus](https://help.sonatype.com/repomanager3/formats/docker-registry)
- [Nexus REST API](https://help.sonatype.com/repomanager3/rest-and-integration-api)

---

## Quick Reference - REST API Examples

### Upload File to Raw Repository:
```bash
curl -v -u jenkins:password \
  --upload-file model.h5 \
  http://localhost:8081/repository/raw-hosted/models/model.h5
```

### Download File:
```bash
curl -u jenkins:password \
  http://localhost:8081/repository/raw-hosted/models/model.h5 \
  -o model.h5
```

### List Repository Contents:
```bash
curl -u jenkins:password \
  http://localhost:8081/service/rest/v1/search?repository=raw-hosted
```

### Get Component Info:
```bash
curl -u jenkins:password \
  http://localhost:8081/service/rest/v1/components?repository=raw-hosted
```
