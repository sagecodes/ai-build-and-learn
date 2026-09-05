#!/bin/bash
# gcloud_commands.sh — run these from your LOCAL machine (not the VM)
# Replace YOUR_PROJECT_ID with your actual GCP project ID.

PROJECT_ID="project-4ef32d8e-2be2-4c1a-b23"
ZONE="us-west1-b"
VM_NAME="rag-comparison"
IMAGE_NAME="johndellenbaugh/rag-comparison"

# ── 1. Create the VM ─────────────────────────────────────────────────────────
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type=e2-medium \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --tags=rag-comparison

# ── 2. Open port 7860 ────────────────────────────────────────────────────────
gcloud compute firewall-rules create allow-rag-comparison \
  --project="$PROJECT_ID" \
  --allow=tcp:7860 \
  --target-tags=rag-comparison \
  --description="Allow inbound traffic to RAG Comparison Gradio app"

# ── 3. Copy setup script to VM ───────────────────────────────────────────────
gcloud compute scp deploy/setup_vm.sh "$VM_NAME":~ \
  --zone="$ZONE" \
  --project="$PROJECT_ID"

# ── 4. SSH into VM ───────────────────────────────────────────────────────────
gcloud compute ssh "$VM_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID"

# Once inside the VM, run:
#   bash setup_vm.sh
#   (edit ~/.env with real credentials)
#   sudo systemctl start rag-comparison

# ── 5. Build and push Docker image (run from project root) ───────────────────
# Run these from topics/cognee/rag_comparison/ on your local machine:
#
#   docker build -t "$IMAGE_NAME":latest .
#   docker push "$IMAGE_NAME":latest

# ── 6. Get the VM's external IP ──────────────────────────────────────────────
gcloud compute instances describe "$VM_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"

# ── 7. Run ingest scripts on the VM ──────────────────────────────────────────
# After the service is running, exec into the container to run ingest:
#
#   docker exec -it rag-comparison bash
#   python ingest/vector_ingest.py
#   python ingest/graph_ingest.py
#   python ingest/cognee_ingest.py
