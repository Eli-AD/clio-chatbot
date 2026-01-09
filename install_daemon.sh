#!/bin/bash
# Install Clio Daemon as a systemd service

set -e

SERVICE_FILE="/home/noles/projects/clio-chatbot/clio-daemon.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "Installing Clio Daemon service..."

# Copy service file
sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/clio-daemon.service"

# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable clio-daemon

# Start service
sudo systemctl start clio-daemon

echo ""
echo "Clio Daemon installed and started!"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status clio-daemon   # Check status"
echo "  sudo systemctl stop clio-daemon     # Stop daemon"
echo "  sudo systemctl start clio-daemon    # Start daemon"
echo "  sudo systemctl restart clio-daemon  # Restart daemon"
echo "  journalctl -u clio-daemon -f        # View logs"
echo "  tail -f ~/clio-memory/logs/daemon.log  # View daemon log"
