#!/usr/bin/env bash
set -euo pipefail

# Tunnel ID must match cloudflare/config.yml and credentials filename.
TUNNEL_ID="980ecf10-3835-4226-a164-dc22d13b2dc9"
CREDENTIALS_FILE="${HOME}/.cloudflared/${TUNNEL_ID}.json"
CONFIG_YAML="${HOME}/.cloudflared/config.yml"

require_file() {
  local path="$1"
  local desc="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${desc}: ${path}" >&2
    exit 1
  fi
}

require_file "$CREDENTIALS_FILE" "tunnel credentials JSON"
require_file "$CONFIG_YAML" "cloudflared config.yml"

# Load token without printing it
set -a
# shellcheck disable=SC1090
# source "${HOME}/.cloudflared/install.env"
set +a
if [[ -z "${CLOUDFLARED_TUNNEL_TOKEN:-}" ]]; then
  echo "CLOUDFLARED_TUNNEL_TOKEN is unset" >&2
  exit 1
fi

curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update
sudo apt-get install -y cloudflared
# Pass env through sudo (-E)
sudo -E cloudflared service install "${CLOUDFLARED_TUNNEL_TOKEN}"
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared --no-pager
