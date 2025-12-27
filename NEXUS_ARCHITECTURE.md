# Nexus Integration Architecture Diagram

## Pipeline Flow with Nexus Integration

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    JENKINS CI/CD PIPELINE                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌───────────────────┐
│ Stage 1: Checkout │  Git Repository
├───────────────────┤  └─ GitHub: sdn-ids
│ Stage 2: Build    │
├───────────────────┤  Docker Build
│ Stage 3: Start FL │  └─ 4 images built
├───────────────────┤
│ Stage 4: Health   │  MLflow Server
├───────────────────┤  └─ Running
│ Stage 5: Train    │  Federated Learning
└────────┬──────────┘  └─ Training complete


         ⭐ NEW: NEXUS INTEGRATION STAGES ⭐

┌────────────────────────────────────────────────────────────────┐
│ Stage 6: SIGN ARTIFACTS                                        │
├────────────────────────────────────────────────────────────────┤
│ ✓ SHA256 checksums                                             │
│ ✓ Model metadata (JSON)                                        │
│ ✓ Optional GPG signatures                                      │
│ Output: signatures/ directory                                  │
└─────────┬──────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 7: UPLOAD ARTIFACTS                                      │
├────────────────────────────────────────────────────────────────┤
│ Destination: Nexus raw-hosted repository                       │
│                                                                │
│ ├─ /fl-pipeline/artifacts/{BUILD_ID}/                          │
│ │  ├─ model-metadata.json                                      │
│ │  ├─ checksums.sha256                                         │
│ │  └─ logs-{BUILD_ID}.tar.gz                                   │
│ │                                                               │
│ └─ /fl-pipeline/reports/                                       │
│    └─ build-report-{BUILD_ID}.json                             │
└─────────┬──────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 8: UPLOAD MODELS                                         │
├────────────────────────────────────────────────────────────────┤
│ Destination: Nexus models-hosted repository                    │
│                                                                │
│ ├─ lstm-fl-{BUILD_ID}.h5                                       │
│ └─ lstm-fl/feature_order-{BUILD_ID}.csv                        │
└─────────┬──────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 9: PUSH DOCKER IMAGES                                    │
├────────────────────────────────────────────────────────────────┤
│ Destination: Nexus Docker Registry (localhost:5000)            │
│                                                                │
│ ├─ flower-server:{BUILD_ID}                                    │
│ ├─ flower-server:latest                                        │
│ ├─ flower-worker:{BUILD_ID}                                    │
│ ├─ flower-worker:latest                                        │
│ ├─ fl-dashboard:{BUILD_ID}                                     │
│ ├─ fl-dashboard:latest                                         │
│ ├─ mlflow-server:{BUILD_ID}                                    │
│ └─ mlflow-server:latest                                        │
└─────────┬──────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────┐
│ Stage 10: GENERATE BUILD REPORT                                │
├────────────────────────────────────────────────────────────────┤
│ Creates JSON report with:                                      │
│ ├─ Build metadata (ID, timestamp, git info)                    │
│ ├─ Artifact locations                                          │
│ ├─ Nexus repository URLs                                       │
│ ├─ Docker image references                                     │
│ └─ Checksum validation info                                    │
│                                                                │
│ Uploads to: raw-hosted/fl-pipeline/reports/                    │
└────────┬─────────────────────────────────────────────────────┘
         │
         ▼
    ✅ SUCCESS
```

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         JENKINS SERVER                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Pipeline Executor                                                  │
│  ├─ Checks out code (GitHub)                                       │
│  ├─ Builds containers (Docker)                                     │
│  ├─ Runs FL training (Docker Compose)                              │
│  ├─ Signs artifacts (SHA256, GPG)                                  │
│  ├─ Uploads to Nexus (HTTP curl)                                   │
│  └─ Pushes Docker images (Docker CLI)                              │
│                                                                      │
│  Credentials Store                                                  │
│  ├─ nexus-credentials (username/password)                          │
│  └─ nexus-docker-credentials (Docker registry)                     │
│                                                                      │
└────────┬─────────────────┬─────────────────┬──────────────────────┘
         │                 │                 │
         │                 │                 │
         ▼                 ▼                 ▼
    
┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐
│  GITHUB          │  │  NEXUS REPOSITORY  │  │  DOCKER DAEMON   │
│                  │  │  (localhost:8081)  │  │  (localhost:2375)│
├──────────────────┤  ├────────────────────┤  ├──────────────────┤
│ sdn-ids repo     │  │ Raw Hosted:        │  │ Docker Images:   │
│ ├─ Jenkinsfile   │  │ ├─ raw-hosted      │  │ ├─ flower-server │
│ ├─ Code          │  │ │  └─ artifacts    │  │ ├─ flower-worker │
│ └─ Docker files  │  │ │  └─ reports      │  │ ├─ fl-dashboard  │
│                  │  │                    │  │ └─ mlflow-server │
│                  │  │ Hosted Models:     │  │                  │
│                  │  │ ├─ models-hosted   │  │ Docker Registry: │
│                  │  │ │  └─ .h5 files    │  │ localhost:5000   │
│                  │  │                    │  │                  │
│                  │  │ Docker Registry:   │  │                  │
│                  │  │ ├─ docker-hosted   │  │                  │
│                  │  │ │  └─ images       │  │                  │
│                  │  │                    │  │                  │
└──────────────────┘  └────────────────────┘  └──────────────────┘
```

---

## Data Flow: Model to Nexus

```
Training Output
├─ models/
│  ├─ LSTM_FL.h5          ────────┐
│  └─ feature_order.csv   ────────┤
│                                 │
├─ mlruns/                        │
│  └─ [MLflow data]               │
│                                 │
├─ logs/                          │
│  └─ [Training logs]             │
│                                 │
└─ [Training metrics]             │
                                  │
         Stage 6: SIGN            │
         ├─ SHA256 hashing        │
         ├─ Metadata generation   │
         └─ Signature creation    │
                 ▼                │
         signatures/              │
         ├─ checksums.sha256      │
         └─ model-metadata.json   │
                 │                │
         Stage 7: UPLOAD ARTIFACTS
                 │ ◄──────────────┘
                 ├─ model-metadata.json → raw-hosted/artifacts/
                 ├─ checksums.sha256 → raw-hosted/artifacts/
                 └─ logs.tar.gz → raw-hosted/artifacts/
                 
                 ▼
    NEXUS Repository Manager
    ├─ raw-hosted (HTTP Upload)
    │  └─ fl-pipeline/
    │     ├─ artifacts/{BUILD_ID}/
    │     └─ reports/{BUILD_ID}.json
    │
    ├─ models-hosted (HTTP Upload)
    │  └─ lstm-fl-{BUILD_ID}.h5
    │
    └─ docker-hosted (Docker Registry)
       └─ flower-server:{BUILD_ID}
```

---

## Network Communication

```
Jenkins Server
   │
   ├─ HTTP PUT (Models & Artifacts)
   │  └─ http://localhost:8081/repository/...
   │     └─ Nexus Repository Manager
   │
   ├─ Docker Push (Images)
   │  └─ localhost:5000/v2/...
   │     └─ Nexus Docker Registry
   │
   └─ Git Pull (Code)
      └─ https://github.com/DatNguyenPT/sdn-ids
         └─ GitHub Repository
```

---

## Artifact Versioning Scheme

```
Build #1
├─ models-hosted/
│  └─ lstm-fl-1.h5
│  └─ lstm-fl/feature_order-1.csv
├─ raw-hosted/
│  └─ fl-pipeline/
│     ├─ artifacts/1/
│     │  ├─ model-metadata.json
│     │  ├─ checksums.sha256
│     │  └─ logs-1.tar.gz
│     └─ reports/
│        └─ build-report-1.json
└─ docker-hosted/
   ├─ flower-server:1
   ├─ flower-server:latest
   ├─ flower-worker:1
   ├─ flower-worker:latest
   ├─ fl-dashboard:1
   ├─ fl-dashboard:latest
   ├─ mlflow-server:1
   └─ mlflow-server:latest

Build #2
├─ models-hosted/
│  └─ lstm-fl-2.h5
│  └─ lstm-fl/feature_order-2.csv
├─ raw-hosted/
│  └─ fl-pipeline/
│     ├─ artifacts/2/
│     │  ├─ model-metadata.json
│     │  ├─ checksums.sha256
│     │  └─ logs-2.tar.gz
│     └─ reports/
│        └─ build-report-2.json
└─ docker-hosted/
   ├─ flower-server:2
   ├─ flower-server:latest (updated)
   ├─ flower-worker:2
   ├─ flower-worker:latest (updated)
   ├─ fl-dashboard:2
   ├─ fl-dashboard:latest (updated)
   ├─ mlflow-server:2
   └─ mlflow-server:latest (updated)
```

---

## Security Boundaries

```
┌─────────────────────────────────────────────────────────┐
│ JENKINS SECURE ZONE                                     │
│ ├─ Credentials stored encrypted                        │
│ ├─ SSH keys for GitHub                                 │
│ ├─ Username/password for Nexus                         │
│ └─ Docker registry credentials                         │
└────────┬────────────────────────────────────────────────┘
         │ (Authenticated HTTPS/HTTP)
         │
┌────────▼────────────────────────────────────────────────┐
│ EXTERNAL SERVICES                                       │
│ ├─ GitHub (HTTPS) - Source Code                        │
│ ├─ Nexus (HTTP/localhost) - Artifact Storage           │
│ └─ Docker Registry (HTTP/localhost) - Images           │
└─────────────────────────────────────────────────────────┘

⚠️ Note: Docker Registry uses HTTP on localhost
   For production, use TLS/SSL via reverse proxy
```

---

## Backup Strategy

```
Nexus Data Structure
├─ sonatype-work/
│  ├─ nexus3/
│  │  ├─ blobs/
│  │  │  ├─ default/     ← All artifact files stored here
│  │  │  │  ├─ models-hosted/
│  │  │  │  ├─ raw-hosted/
│  │  │  │  └─ docker-hosted/
│  │  │  └─ docker/
│  │  └─ db/             ← Metadata database (important!)
│  └─ [Other data]

Backup Recommendations:
├─ Daily: Database backup (minimal size)
├─ Weekly: Full blobs backup (large size)
├─ Monthly: Full system backup
└─ Keep: 3-6 month retention
```

---

## Metrics & Monitoring

```
Monitor These Points:
├─ Disk Usage
│  └─ /sonatype-work/nexus3/blobs/default/
│     └─ Watch for growth > 80%
│
├─ Upload Success Rate
│  └─ Jenkins Pipeline Logs
│     └─ Look for HTTP 201/204 status
│
├─ Download Activity
│  └─ Nexus Logs: sonatype-work/nexus3/log/
│     └─ Track model pulls/Docker image pulls
│
├─ Repository Size
│  └─ Nexus UI: Admin → Repositories
│     └─ models-hosted, raw-hosted, docker-hosted
│
└─ Build Success Rate
   └─ Jenkins Metrics
      └─ Track Stage 6-10 success %
```

---

## Troubleshooting Flowchart

```
Pipeline Fails?
│
├─ Stage 1-5 Failed?
│  └─ See: JENKINS_DEBUG_GUIDE (existing)
│
├─ Stage 6 Failed (Signing)?
│  ├─ Check: SHA256 tool installed
│  ├─ Check: models/ directory exists
│  └─ Solution: Run: ./sign_artifacts.sh ./models ./signatures
│
├─ Stage 7 Failed (Artifacts Upload)?
│  ├─ Check: nexus-credentials in Jenkins
│  ├─ Check: raw-hosted repository exists
│  ├─ Check: Nexus reachable: curl http://localhost:8081
│  └─ Solution: Run: ./nexus_config.sh
│
├─ Stage 8 Failed (Models Upload)?
│  ├─ Check: models-hosted repository exists
│  ├─ Check: LSTM_FL.h5 file exists
│  └─ Solution: Run: ./upload_to_nexus.sh file.h5 models-hosted ...
│
├─ Stage 9 Failed (Docker Push)?
│  ├─ Check: nexus-docker-credentials in Jenkins
│  ├─ Check: Docker daemon config (insecure-registries)
│  ├─ Check: docker-hosted repository exists
│  ├─ Check: Docker reachable: docker login localhost:5000
│  └─ Solution: Fix daemon.json and restart Docker
│
└─ Stage 10 Failed (Report)?
   ├─ Check: Build report JSON valid
   ├─ Check: raw-hosted accessible
   └─ Solution: Check stages 7 status
```

---

## Complete Data Journey

```
START: GitHub
   │
   ├─ Checkout Code
   │
   └─ Build & Test
      │
      ├─ Federated Learning Training
      │
      ├─ Generate:
      │  ├─ models/LSTM_FL.h5
      │  ├─ models/feature_order.csv
      │  ├─ mlruns/* (MLflow data)
      │  └─ logs/* (training logs)
      │
      └─ ⭐ NEW: Nexus Integration ⭐
         │
         ├─ Stage 6: SIGN
         │  ├─ Create checksums
         │  ├─ Generate metadata
         │  └─ Create signatures
         │
         ├─ Stage 7: UPLOAD ARTIFACTS
         │  └─ To: Nexus raw-hosted/fl-pipeline/artifacts/
         │
         ├─ Stage 8: UPLOAD MODELS
         │  └─ To: Nexus models-hosted/
         │
         ├─ Stage 9: PUSH DOCKER IMAGES
         │  └─ To: Nexus docker-hosted/ (port 5000)
         │
         ├─ Stage 10: GENERATE REPORT
         │  └─ To: Nexus raw-hosted/fl-pipeline/reports/
         │
         └─ COMPLETE ✅

END: Artifacts stored in Nexus, ready for:
   ├─ Deployment
   ├─ Auditing
   ├─ Versioning
   ├─ Distribution
   └─ Compliance tracking
```

---

## Key Integration Points

```
Jenkins ←→ GitHub
   │        (Source Code Management)
   │
   ├─→ Docker Engine
   │    (Build & Tag Images)
   │
   ├─→ Nexus Repository
   │    (HTTP Upload: curl)
   │    ├─ Raw Hosted (Artifacts)
   │    └─ Models Hosted (ML Models)
   │
   └─→ Docker Registry
       (Docker Push)
       (localhost:5000)
```

This diagram shows the complete integration architecture for your Nexus-enabled Jenkins pipeline.
