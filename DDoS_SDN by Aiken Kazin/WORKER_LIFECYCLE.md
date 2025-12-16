# Worker Lifecycle: Training vs Inference

## 🔄 Current Worker Behavior

### **After FL Training Completes**:

1. **Workers Stay Running**:
   - Workers have `restart: "unless-stopped"` in docker-compose.yml
   - When FL training completes (all rounds done), workers continue running
   - Workers start inference API automatically after training

2. **For Next Training**:
   - ✅ **No manual restart needed!**
   - Workers are already running and ready
   - Just start the FL servers again for new training

---

## 📊 Worker Lifecycle Flow

### **Training Cycle**:

```
FL Servers Start
    ↓
Workers Connect
    ↓
FL Training (5 rounds)
    ↓
Training Completes
    ↓
Workers Start Inference API
    ↓
Workers Stay Running (for inference)
```

---

## ✅ Answer: Workers Stay Running!

### **For Training**:
- ✅ **Workers stay running** after training
- ✅ Can start new training sessions without restarting workers
- ✅ Workers automatically connect when servers start

### **For Inference (Phase 4)**:
- ✅ **Workers stay running** after training
- ✅ Inference API is available immediately
- ✅ Can use inference API without restarting

---

## 🔧 Current Configuration

### **docker-compose.yml**:
```yaml
flower-worker-1:
  restart: "unless-stopped"  # Workers stay running after training
  # Workers automatically start inference API after training completes
```

---

## 💡 Summary

### **Question**: Do we need to restart workers after training?

### **Answer**:

**For Training**:
- ✅ **No** - Workers stay running and automatically reconnect
- ✅ Just start FL servers for new training sessions

**For Inference**:
- ✅ **No** - Workers stay running and inference API is available
- ✅ Can use inference API immediately after training

---

## 🎯 Current State

| Scenario | Workers Status | Manual Restart Needed? |
|----------|---------------|----------------------|
| **Training** | Stay running | ❌ No |
| **Inference (After Training)** | Stay running | ❌ No |
| **New Training Session** | Already running | ❌ No (just start servers) |

---

## 🔄 Worker Behavior

**After Training Completes**:
1. FL training finishes (all rounds done)
2. Workers automatically start inference API
3. Workers stay running (`restart: "unless-stopped"`)
4. Ready for inference requests
5. Ready for next training session (when servers start)

**To Start New Training**:
```bash
# Just start the FL servers
docker compose up -d flower-server-mlpv2 flower-server-cnn1d ...

# Workers will automatically connect
```

**To Stop Workers** (if needed):
```bash
docker compose stop flower-worker-*
```
