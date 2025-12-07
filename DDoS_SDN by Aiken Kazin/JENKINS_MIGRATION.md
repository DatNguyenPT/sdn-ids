# Migration from GitHub Actions to Jenkins

## Summary

Switching from GitHub Actions to Jenkins due to disk space limitations.

---

## What Changed

### Removed:
- ❌ `.github/workflows/fl_training.yml` (GitHub Actions workflow)
- ❌ GitHub Actions disk space optimizations

### Added:
- ✅ `Jenkinsfile` (Jenkins pipeline)
- ✅ `JENKINS_SETUP.md` (Setup guide)
- ✅ `JENKINS_MIGRATION.md` (This file)

---

## Jenkinsfile Features

### Same Functionality:
- ✅ Data validation
- ✅ Code quality checks
- ✅ FL training (all 4 models)
- ✅ Model evaluation
- ✅ MLflow data export
- ✅ Artifact storage

### Better Features:
- ✅ **Unlimited disk space** - No restrictions
- ✅ **Parallel builds** - All models simultaneously
- ✅ **Full control** - Configure as needed
- ✅ **Better logging** - Detailed console output

---

## Migration Steps

### 1. Install Jenkins
See `JENKINS_SETUP.md` for detailed instructions.

### 2. Configure Jenkins
- Install Docker plugin
- Configure Docker access
- Set up GitHub credentials (if private repo)

### 3. Create Pipeline
- Create new pipeline from `Jenkinsfile`
- Point to your repository
- Configure parameters

### 4. Run First Build
- Build with parameters
- Monitor progress
- Verify artifacts

---

## Jenkinsfile Parameters

```groovy
MODEL_TYPE: 'MLPv2', 'LSTM', 'CNN1D', 'CNN_LSTM', 'all'
NUM_ROUNDS: '5' (default)
ENABLE_DP: true/false
IID: true/false
```

---

## Benefits

### Disk Space:
- **GitHub Actions**: 14GB limit (causing failures)
- **Jenkins**: Unlimited (your server)

### Parallel Builds:
- **GitHub Actions**: Limited by disk space
- **Jenkins**: All models simultaneously

### Control:
- **GitHub Actions**: Limited configuration
- **Jenkins**: Full control

---

## Next Steps

1. ✅ Install Jenkins (see `JENKINS_SETUP.md`)
2. ✅ Configure pipeline
3. ✅ Run first build
4. ✅ Verify everything works
5. ✅ Continue with Phase 4 (Federated Inference)

---

## Keep GitHub Actions?

You can keep GitHub Actions for:
- ✅ CI checks (PR validation)
- ✅ Quick tests
- ✅ Code quality checks

Use Jenkins for:
- ✅ Full FL training
- ✅ Model training
- ✅ Heavy builds

---

## Support

If you need help:
1. Check `JENKINS_SETUP.md` for setup
2. Check Jenkins logs for errors
3. Verify Docker is accessible
4. Check disk space on server

**Jenkins is ready to use!** 🚀

