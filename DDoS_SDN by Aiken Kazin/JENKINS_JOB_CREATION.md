# How to Create Jenkins Job for FL Training Pipeline

## 📋 Prerequisites

✅ Jenkins is running (`http://localhost:8090`)  
✅ GitHub PAT credential added (ID: `github-pat`)  
✅ Docker plugins installed  
✅ Repository URL updated in Jenkinsfile  

---

## 🚀 Step-by-Step: Create Jenkins Pipeline Job

### Step 1: Access Jenkins Dashboard

1. Open Jenkins in browser: `http://localhost:8090`
2. Login with your admin credentials

---

### Step 2: Create New Pipeline Job

1. **Click "New Item"** (on Jenkins dashboard, left sidebar)

2. **Enter Job Name**:
   - Name: `FL-Training-Pipeline` (or any name you prefer)
   - ⚠️ **Important**: No spaces, use hyphens or underscores

3. **Select Job Type**:
   - Select **"Pipeline"** (not Freestyle project)
   - Click **"OK"**

---

### Step 3: Configure Pipeline

You'll see the job configuration page. Configure these sections:

#### A. **General Settings** (Optional)

- ✅ **"GitHub project"** (optional):
  - Project url: `https://github.com/DatNguyenPT/sdn-ids`
  
- ✅ **"Build Triggers"** (optional):
  - **"GitHub hook trigger for GITScm polling"** (if using webhooks)
  - **"Build periodically"** (for scheduled builds):
    - Schedule: `H 2 * * *` (runs daily at 2 AM)
  - **"Poll SCM"** (check for changes):
    - Schedule: `H/15 * * * *` (every 15 minutes)

#### B. **Pipeline Configuration** (REQUIRED)

Scroll down to **"Pipeline"** section:

1. **Definition**: Select **"Pipeline script from SCM"**

2. **SCM**: Select **"Git"**

3. **Repository URL**: 
   ```
   https://github.com/DatNguyenPT/sdn-ids.git
   ```

4. **Credentials**: 
   - Click **"Add"** → **"Jenkins"**
   - Select **"github-pat"** from dropdown
   - (If not visible, go to Credentials section first)

5. **Branches to build**:
   - Branch Specifier: `*/main` (or `*/master` if your default branch is master)

6. **Script Path**: 
   ```
   DDoS_SDN by Aiken Kazin/Jenkinsfile
   ```
   ⚠️ **Important**: This is the path to your Jenkinsfile relative to repo root!

7. **Lightweight checkout**: ✅ Uncheck (we need full checkout for Docker)

---

### Step 4: Save Configuration

1. Scroll to bottom
2. Click **"Save"** (not "Apply")

---

### Step 5: Run Pipeline

1. **You'll see the job page**
2. Click **"Build with Parameters"** (or "Build Now" if no parameters)
3. **Select Parameters**:
   - **MODEL_TYPE**: `LSTM` (default, or choose another)
   - **NUM_ROUNDS**: `5` (default)
   - **ENABLE_DP**: `false` (default) or `true` to enable Differential Privacy
   - **IID**: `true` (default) or `false` for Non-IID distribution
4. Click **"Build"**

---

## 🔍 Verify Job Configuration

### Check Job Settings:

1. Go to job page: `FL-Training-Pipeline`
2. Click **"Configure"**
3. Verify:
   - ✅ Repository URL is correct
   - ✅ Credentials selected (`github-pat`)
   - ✅ Script Path: `DDoS_SDN by Aiken Kazin/Jenkinsfile`
   - ✅ Branch: `*/main` (or `*/master`)

---

## 📊 Monitor Pipeline Execution

### View Build Progress:

1. **Build History**: See builds on job page (left sidebar)
2. **Console Output**: Click build number → **"Console Output"**
3. **Stage View**: See pipeline stages progress
4. **Blue Ocean** (if installed): Modern UI for pipeline visualization

### What to Watch:

- ✅ **Checkout**: Should clone repository successfully
- ✅ **Validate Dataset**: Should pass data validation
- ✅ **Code Quality**: Should complete (warnings OK)
- ✅ **Docker Cleanup**: Should free disk space
- ✅ **FL Training**: Should start Docker containers and train models
- ✅ **Model Evaluation**: Should evaluate trained models
- ✅ **Export MLflow**: Should export MLflow data

---

## 🐛 Troubleshooting

### Error: "Cannot find Jenkinsfile"

**Problem**: Jenkins can't find Jenkinsfile

**Solution**:
1. Check **Script Path** in job configuration
2. Should be: `DDoS_SDN by Aiken Kazin/Jenkinsfile`
3. Verify file exists in repository
4. Check branch name (`main` vs `master`)

### Error: "Credentials not found"

**Problem**: `github-pat` credential not found

**Solution**:
1. Go to **Manage Jenkins** → **Credentials**
2. Verify credential exists with ID: `github-pat`
3. If missing, add it (see `JENKINS_GIT_SETUP.md`)
4. Re-run build

### Error: "Docker command not found"

**Problem**: Docker not accessible in Jenkins

**Solution**:
1. Verify Docker socket is mounted in docker-compose.yml
2. Check Jenkins container has Docker access:
   ```bash
   docker exec jenkins docker --version
   ```
3. If fails, check docker-compose.yml volume mount

### Error: "Repository not found" or "Authentication failed"

**Problem**: Can't access GitHub repository

**Solution**:
1. Verify GitHub PAT is valid and not expired
2. Check PAT has `repo` scope
3. Verify repository URL is correct
4. Test credential manually:
   ```bash
   docker exec jenkins git ls-remote https://github.com/DatNguyenPT/sdn-ids.git
   ```

---

## 🔄 Alternative: Create Job from Jenkinsfile Directly

### Method 2: Pipeline Script (For Testing)

If you want to test without Git checkout:

1. **Create Pipeline Job** (same as Step 1-2)

2. **Pipeline Configuration**:
   - **Definition**: Select **"Pipeline script"** (not "from SCM")
   - **Script**: Copy entire Jenkinsfile content
   - Click **"Save"**

3. **Run**: Click **"Build with Parameters"**

⚠️ **Note**: This method won't pull latest code from Git, but useful for testing.

---

## 📝 Quick Reference

**Job Name**: `FL-Training-Pipeline`  
**Job Type**: Pipeline  
**Repository**: `https://github.com/DatNguyenPT/sdn-ids.git`  
**Credentials**: `github-pat`  
**Script Path**: `DDoS_SDN by Aiken Kazin/Jenkinsfile`  
**Branch**: `*/main`  
**Default Model**: `LSTM`  

---

## ✅ Checklist

Before running pipeline:

- [ ] Jenkins is running (`http://localhost:8090`)
- [ ] GitHub PAT credential added (`github-pat`)
- [ ] Docker plugins installed
- [ ] Job created with correct configuration
- [ ] Script Path: `DDoS_SDN by Aiken Kazin/Jenkinsfile`
- [ ] Repository URL: `https://github.com/DatNguyenPT/sdn-ids.git`
- [ ] Credentials selected: `github-pat`
- [ ] Branch: `*/main` (or `*/master`)

---

## 🎯 Next Steps

After job is created:

1. **Test Run**: Build with parameters (LSTM, 5 rounds)
2. **Monitor**: Watch console output
3. **Verify**: Check models are created in `models/` directory
4. **Schedule**: Set up periodic builds if needed
5. **Webhooks**: Configure GitHub webhooks for automatic builds

---

## 🚀 You're Ready!

Once job is created and configured, you can:

- **Manual Build**: Click "Build with Parameters" anytime
- **Scheduled Build**: Set up cron schedule
- **Webhook Build**: Trigger on Git push (if configured)

**Happy Training!** 🎉

