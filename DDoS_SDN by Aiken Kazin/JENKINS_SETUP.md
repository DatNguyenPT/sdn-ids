# Jenkins Setup Guide for FL Training Pipeline

## Why Jenkins?

- ✅ **Unlimited Disk Space** - Your server, your resources
- ✅ **Full Control** - Configure as needed
- ✅ **No Limitations** - Build all models in parallel
- ✅ **Better for Research** - More flexibility

---

## Step 1: Install Jenkins

### Option A: Docker (Easiest)

```bash
# Create Jenkins data directory
mkdir -p ~/jenkins_home
chmod 777 ~/jenkins_home

# Run Jenkins container
docker run -d \
  --name jenkins \
  -p 8080:8080 \
  -p 50000:50000 \
  -v ~/jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkins/jenkins:lts

# Get initial admin password
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

### Option B: Native Installation

**Ubuntu/Debian:**
```bash
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee \
  /usr/share/keyrings/jenkins-keyring.asc > /dev/null
echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null
sudo apt-get update
sudo apt-get install jenkins
```

**macOS:**
```bash
brew install jenkins-lts
brew services start jenkins-lts
```

---

## Step 2: Initial Jenkins Setup

1. **Access Jenkins**: `http://localhost:8080` (or your server IP)

2. **Unlock Jenkins**: Enter initial admin password (from Step 1)

3. **Install Plugins**: 
   - Install suggested plugins
   - Or install manually:
     - Docker Pipeline
     - Git
     - GitHub

4. **Create Admin User**: Set up your admin account

---

## Step 3: Configure Jenkins for Docker

### Install Docker Plugin

1. Go to **Manage Jenkins** → **Plugins**
2. Search for **"Docker Pipeline"**
3. Install and restart Jenkins

### Configure Docker

**If Jenkins runs in Docker:**
- Docker socket is already mounted (`-v /var/run/docker.sock:/var/run/docker.sock`)

**If Jenkins runs natively:**
```bash
# Add jenkins user to docker group
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins
```

---

## Step 4: Create Jenkins Pipeline

### Method 1: From Jenkinsfile (Recommended)

1. Go to **New Item**
2. Enter name: `FL-Training-Pipeline`
3. Select **Pipeline**
4. Click **OK**

5. **Pipeline Configuration**:
   - **Definition**: Pipeline script from SCM
   - **SCM**: Git
   - **Repository URL**: Your GitHub repository URL
   - **Credentials**: Add GitHub credentials if private repo
   - **Branch**: `*/main` (or your branch)
   - **Script Path**: `DDoS_SDN by Aiken Kazin/Jenkinsfile`
   - Click **Save**

### Method 2: Copy Jenkinsfile Content

1. Go to **New Item** → **Pipeline**
2. In **Pipeline** section, select **Pipeline script**
3. Copy content from `Jenkinsfile`
4. Click **Save**

---

## Step 5: Run Pipeline

1. Click on your pipeline
2. Click **Build with Parameters**
3. Select options:
   - **MODEL_TYPE**: `all` (or specific model)
   - **NUM_ROUNDS**: `5`
   - **ENABLE_DP**: `false`
   - **IID**: `true`
4. Click **Build**

---

## Step 6: Monitor Build

1. Click on build number
2. Click **Console Output** to see logs
3. View **Stage View** for progress
4. Download artifacts after completion

---

## Jenkins vs GitHub Actions

| Feature | Jenkins | GitHub Actions |
|---------|---------|----------------|
| **Disk Space** | ✅ Unlimited | ❌ 14GB limit |
| **Setup** | ⚠️ Requires server | ✅ Cloud-based |
| **Cost** | ⚠️ Server costs | ✅ Free (research) |
| **Control** | ✅ Full control | ⚠️ Limited |
| **Parallel Builds** | ✅ All models | ❌ Limited by disk |

---

## Jenkinsfile Features

### Parameters:
- **MODEL_TYPE**: Choose model(s) to train
- **NUM_ROUNDS**: Number of FL rounds
- **ENABLE_DP**: Enable Differential Privacy
- **IID**: IID or Non-IID distribution

### Stages:
1. **Checkout**: Get code from Git
2. **Validate Dataset**: Data validation
3. **Code Quality**: Linting and formatting
4. **Docker Cleanup**: Free disk space
5. **FL Training**: Train models (parallel)
6. **Model Evaluation**: Evaluate trained models
7. **Export MLflow Data**: Export experiment data

### Artifacts:
- Model files (`.h5`)
- MLflow reports (CSV, JSON)
- Training logs

---

## Troubleshooting

### Issue: Jenkins can't access Docker
**Solution**: 
```bash
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins
```

### Issue: Out of disk space
**Solution**: 
- Add more disk space to server
- Or configure Jenkins to use external storage

### Issue: Build fails
**Solution**:
- Check Jenkins logs: `/var/log/jenkins/jenkins.log`
- Check Docker logs: `docker logs jenkins`
- Verify Docker is running: `docker ps`

### Issue: Can't connect to GitHub
**Solution**:
- Add GitHub credentials in Jenkins
- Go to **Manage Jenkins** → **Credentials** → **Add**

---

## Advantages of Jenkins

1. ✅ **No Disk Space Limits** - Your server, your resources
2. ✅ **Build All Models in Parallel** - No restrictions
3. ✅ **Full Control** - Configure as needed
4. ✅ **Better Logging** - Detailed console output
5. ✅ **Artifact Management** - Built-in artifact storage
6. ✅ **Scheduling** - Built-in cron support
7. ✅ **Notifications** - Email, Slack, etc.

---

## Next Steps

1. ✅ Install Jenkins
2. ✅ Configure Docker access
3. ✅ Create pipeline from Jenkinsfile
4. ✅ Run first build
5. ✅ Monitor and adjust

**Your FL training pipeline is now ready in Jenkins!** 🚀

