# Jenkins Git Setup with Personal Access Token (PAT)

## 🔐 Setting Up GitHub PAT in Jenkins

### Step 1: Create GitHub Personal Access Token

1. **Go to GitHub**:
   - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Or direct link: `https://github.com/settings/tokens`

2. **Generate New Token**:
   - Click **"Generate new token (classic)"**
   - Give it a name: `Jenkins-FL-Pipeline`
   - Select expiration: `90 days` or `No expiration` (for testing)
   - Select scopes:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows) - if needed
   - Click **"Generate token"**

3. **Copy Token**:
   - ⚠️ **IMPORTANT**: Copy the token immediately (you won't see it again!)
   - Example: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

### Step 2: Add PAT to Jenkins Credentials

1. **Open Jenkins**:
   - Go to `http://localhost:8090` (or your Jenkins URL)

2. **Navigate to Credentials**:
   - Click **"Manage Jenkins"** → **"Credentials"**
   - Click **"System"** → **"Global credentials (unrestricted)"**
   - Click **"Add Credentials"**

3. **Configure Credential**:
   - **Kind**: `Username with password` or `Secret text`
   - **Scope**: `Global`
   - **Username**: Your GitHub username (or leave blank if using Secret text)
   - **Password/Secret**: Paste your GitHub PAT (`ghp_xxxxx...`)
   - **ID**: `github-pat` (must match Jenkinsfile!)
   - **Description**: `GitHub Personal Access Token for FL Pipeline`
   - Click **"OK"**

---

### Step 3: Update Jenkinsfile

The Jenkinsfile has been updated to use credentials:

```groovy
checkout([
    $class: 'GitSCM',
    branches: [[name: '*/main']],  // Change to '*/master' if needed
    extensions: [],
    userRemoteConfigs: [[
        credentialsId: 'github-pat',  // Must match credential ID
        url: 'https://github.com/YOUR_USERNAME/YOUR_REPO.git'
    ]]
])
```

**⚠️ IMPORTANT**: Update the `url` in Jenkinsfile with your actual repository URL!

---

## 🔄 Alternative: Using SSH (If Preferred)

### Option 1: SSH Key Setup

1. **Generate SSH Key** (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "jenkins@your-domain.com"
   ```

2. **Add SSH Key to GitHub**:
   - Copy public key: `cat ~/.ssh/id_ed25519.pub`
   - GitHub → Settings → SSH and GPG keys → New SSH key
   - Paste and save

3. **Add SSH Key to Jenkins**:
   - Jenkins → Credentials → Add Credentials
   - **Kind**: `SSH Username with private key`
   - **ID**: `github-ssh`
   - **Username**: `git`
   - **Private Key**: Paste your private key or select "Enter directly"
   - Click **"OK"**

4. **Update Jenkinsfile**:
   ```groovy
   checkout([
       $class: 'GitSCM',
       branches: [[name: '*/main']],
       extensions: [],
       userRemoteConfigs: [[
           credentialsId: 'github-ssh',
           url: 'git@github.com:YOUR_USERNAME/YOUR_REPO.git'
       ]]
   ])
   ```

---

## ✅ Verification

### Test Checkout:

1. **Create Test Pipeline**:
   ```groovy
   pipeline {
       agent any
       stages {
           stage('Test Checkout') {
               steps {
                   checkout([
                       $class: 'GitSCM',
                       branches: [[name: '*/main']],
                       extensions: [],
                       userRemoteConfigs: [[
                           credentialsId: 'github-pat',
                           url: 'https://github.com/YOUR_USERNAME/YOUR_REPO.git'
                       ]]
                   ])
                   sh 'ls -la'
                   sh 'git log --oneline -5'
               }
           }
       }
   }
   ```

2. **Run Pipeline**:
   - If checkout succeeds, credentials are working! ✅
   - If it fails, check:
     - Credential ID matches Jenkinsfile
     - PAT has correct scopes (`repo`)
     - Repository URL is correct

---

## 🔧 Troubleshooting

### Error: "Credentials not found"

**Problem**: Jenkins can't find credential `github-pat`

**Solution**:
1. Check credential ID matches exactly (case-sensitive)
2. Verify credential is in "Global" scope
3. Re-create credential if needed

### Error: "Authentication failed"

**Problem**: PAT is invalid or expired

**Solution**:
1. Generate new PAT on GitHub
2. Update credential in Jenkins
3. Verify PAT has `repo` scope

### Error: "Repository not found"

**Problem**: Wrong repository URL or no access

**Solution**:
1. Verify repository URL is correct
2. Check PAT has access to the repository
3. For private repos, ensure PAT has `repo` scope

---

## 📝 Quick Reference

**Credential ID**: `github-pat` (must match Jenkinsfile)

**PAT Scopes Required**:
- ✅ `repo` (Full control of private repositories)

**Jenkinsfile Location**: Line 37-47 (Checkout stage)

**Default Branch**: `main` (change to `master` if needed)

---

## 🚀 After Setup

Once credentials are configured:

1. **Update Repository URL** in Jenkinsfile (line 44)
2. **Run Pipeline** - it should checkout successfully
3. **Verify**: Check console output for successful checkout

---

## 🔒 Security Best Practices

1. **Use PAT with minimum required scopes**
2. **Set expiration date** (don't use "No expiration" in production)
3. **Rotate tokens regularly** (every 90 days)
4. **Don't commit tokens** to Git (use Jenkins credentials)
5. **Use different tokens** for different projects

