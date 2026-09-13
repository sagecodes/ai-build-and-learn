#!/bin/bash
# setup_vm.sh — run once on the GCP VM after SSH in
# Sets up Docker, pulls the image, and installs the systemd service.

set -e

echo "── Installing Docker ──────────────────────────────────────────────"
sudo apt-get update -q
sudo apt-get install -y -q docker.io
sudo systemctl enable docker
sudo usermod -aG docker "$USER"
newgrp docker

echo "── Pulling image ──────────────────────────────────────────────────"
docker pull docker.io/johndellenbaugh/rag-comparison:latest

echo "── Writing .env ───────────────────────────────────────────────────"
# Edit this block with your actual values before running
cat > /home/"$USER"/.env << 'EOF'
ANTHROPIC_API_KEY=your-anthropic-api-key
PG_URL=postgresql://postgres.your-project-ref:your-password@aws-1-us-east-1.pooler.supabase.com:5432/postgres
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
EOF
echo "  ⚠  Edit /home/$USER/.env with real values before starting the service."

echo "── Installing systemd service ─────────────────────────────────────"
sudo tee /etc/systemd/system/rag-comparison.service > /dev/null << EOF
[Unit]
Description=RAG Comparison Gradio App
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker rm -f rag-comparison
ExecStart=/usr/bin/docker run --rm --name rag-comparison \\
  --env-file /home/$USER/.env \\
  -v /home/$USER/.cognee_data:/app/.cognee_data \\
  -p 7860:7860 \\
  docker.io/johndellenbaugh/rag-comparison:latest
ExecStop=/usr/bin/docker stop rag-comparison

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-comparison

echo ""
echo "Done. Next steps:"
echo "  1. Edit /home/$USER/.env with your real credentials"
echo "  2. sudo systemctl start rag-comparison"
echo "  3. sudo systemctl status rag-comparison"
echo "  4. App will be at http://<VM-EXTERNAL-IP>:7860"
