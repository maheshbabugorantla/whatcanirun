#!/usr/bin/env bash
#
# init-firewall.sh
#
# Egress allowlist for the whatcanirun dev container.
# Defaults: DROP all outbound. Re-allows DNS, localhost, and HTTPS to
# allowlisted domains only.

set -euo pipefail

# --------------------- 0. Reset ---------------------
iptables -F
iptables -X
ipset destroy allowed-domains 2>/dev/null || true

# --------------------- 1. Default policies ---------------------
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

iptables -A INPUT  -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# --------------------- 2. DNS ---------------------
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A INPUT  -p udp --sport 53 -j ACCEPT
iptables -A INPUT  -p tcp --sport 53 -j ACCEPT

# --------------------- 3. Docker bridge networks ----------------------------
for SUBNET in 172.16.0.0/12 192.168.0.0/16 10.0.0.0/8; do
  iptables -A OUTPUT -d "$SUBNET" -j ACCEPT
done

# --------------------- 4. Allowlist domains ---------------------
ipset create allowed-domains hash:net family inet hashsize 1024 maxelem 65536

ALLOWED_DOMAINS=(
  # Anthropic / Claude Code (including OAuth login flow)
  "api.anthropic.com"
  "statsig.anthropic.com"
  "console.anthropic.com"
  "claude.ai"
  "docs.claude.com"
  "platform.claude.com"
  "docs.anthropic.com"

  # Package registries
  "registry.npmjs.org"
  "pypi.org"
  "files.pythonhosted.org"

  # Python tooling
  "astral.sh"

  # Git hosting
  "github.com"
  "ssh.github.com"
  "api.github.com"
  "objects.githubusercontent.com"
  "raw.githubusercontent.com"
  "codeload.github.com"

  # Debian / apt
  "deb.debian.org"
  "security.debian.org"
  "archive.ubuntu.com"
  "security.ubuntu.com"

  # Project data sources (upstream APIs)
  "computeprices.com"
  "api.computeprices.com"
  "artificialanalysis.ai"
  "huggingface.co"
  # Additional GPU/inference price sources referenced in specs
  "glama.ai"
  "www.pulsemcp.com"
  "www.spheron.network"
  "www.nvidia.com"

  # Project documentation
  "docs.pydantic.dev"
  "fastmcp.com"
  "modelcontextprotocol.io"
)

resolve_and_add() {
  local domain="$1"
  local ips
  ips=$(getent ahosts "$domain" 2>/dev/null | awk '{print $1}' | grep -E '^[0-9]+\.' | sort -u || true)
  if [[ -z "$ips" ]]; then
    echo "  warn: could not resolve $domain (skipping)"
    return
  fi
  while read -r ip; do
    [[ -n "$ip" ]] && ipset add allowed-domains "$ip" -exist
  done <<< "$ips"
  echo "  ok: $domain"
}

echo "Resolving allowlisted domains..."
for d in "${ALLOWED_DOMAINS[@]}"; do
  resolve_and_add "$d"
done

iptables -A OUTPUT -p tcp -m set --match-set allowed-domains dst --dport 443 -j ACCEPT
iptables -A OUTPUT -p tcp -m set --match-set allowed-domains dst --dport 80  -j ACCEPT

# --------------------- 5. GitHub CDN ranges ---------------------
echo "Fetching GitHub IP ranges..."
GITHUB_RANGES=$(curl -sS --max-time 10 https://api.github.com/meta 2>/dev/null \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
seen = set()
for key in ('git', 'web', 'packages', 'actions'):
    for r in d.get(key, []):
        if r not in seen:
            seen.add(r)
            print(r)
" 2>/dev/null || true)

if [[ -n "$GITHUB_RANGES" ]]; then
  COUNT=0
  while read -r cidr; do
    [[ -n "$cidr" ]] && ipset add allowed-domains "$cidr" -exist 2>/dev/null && COUNT=$((COUNT+1))
  done <<< "$GITHUB_RANGES"
  echo "  ok: added $COUNT GitHub CIDR ranges"
fi

# --------------------- 6. Verification ---------------------
echo "Firewall ready. Testing egress..."

if curl -sS --max-time 5 -o /dev/null -w "%{http_code}" https://api.anthropic.com/ | grep -qE '^(2|3|4)'; then
  echo "  ok: api.anthropic.com reachable"
else
  echo "  WARN: api.anthropic.com not reachable — Claude Code may not work"
fi

if timeout 3 curl -sS -o /dev/null https://1.1.1.1/ 2>/dev/null; then
  echo "  WARN: egress firewall is NOT blocking unallowlisted traffic"
  exit 1
else
  echo "  ok: unallowlisted traffic is blocked"
fi

echo "Firewall initialization complete."
