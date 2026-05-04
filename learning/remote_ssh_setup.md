# Remote SSH Access — Beelink S12 Pro Setup Guide

> Goal: SSH into your Beelink S12 Pro from anywhere (campus, café, phone) securely.

---

## The Decision: How Do You Want to Connect?

There are three main approaches. Here's an honest comparison:

| Method | Open Ports? | Setup Time | Security | Cost | Best For |
|---|---|---|---|---|---|
| **Tailscale** (recommended) | None | 10 minutes | Excellent | Free (personal) | You — simple, secure, zero maintenance |
| **WireGuard VPN** | 1 UDP port | 30–60 min | Excellent | Free | Learning networking / full control |
| **Port Forwarding SSH** | 1 TCP port | 15 min | Risky | Free | NOT recommended — bots attack within minutes |
| **Cloudflare Tunnel** | None | 30–45 min | Excellent | Free (needs domain) | If you already own a domain |

**My recommendation:** Start with **Tailscale** — it's a 10-minute setup, free, no router config needed, and used by professional sysadmins. Add WireGuard later if you want to learn VPN engineering.

---

## Option 1: Tailscale (Recommended — 10 Minutes)

### What It Is

Tailscale creates a private mesh VPN between your devices using WireGuard under the hood. Every device gets a stable `100.x.x.x` IP address. No ports, no router config, no dynamic DNS — it just works.

```
Your phone (anywhere)          Beelink S12 Pro (at home)
  100.64.0.2                     100.64.0.1
      │                              │
      └──── Tailscale mesh ──────────┘
           (encrypted, NAT-traversal)
           (no open ports on router)
```

### How It Works (Under the Hood)

1. Both devices make **outbound** connections to Tailscale coordination servers
2. Tailscale brokers a direct WireGuard tunnel between your devices (peer-to-peer)
3. If direct connection fails (strict NAT), traffic relays through Tailscale's DERP servers
4. All traffic is end-to-end encrypted — Tailscale servers cannot read your data
5. Each device gets a stable `100.x.x.x` address that never changes

### Setup — Server Side (Beelink S12 Pro)

```bash
# 1. Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Start and authenticate
sudo tailscale up

# A URL will appear — open it in a browser, log in with Google/GitHub/Microsoft
# The Beelink will be added to your "tailnet"

# 3. Check your Tailscale IP
tailscale ip -4
# Example output: 100.64.0.1

# 4. Verify status
tailscale status
```

### Setup — Client Side (Your Laptop / Phone)

**Windows/macOS/Linux laptop:**
1. Download Tailscale from [https://tailscale.com/download](https://tailscale.com/download)
2. Install and sign in with the **same account** you used on the Beelink
3. Done — your laptop now has a `100.x.x.x` address on the same mesh

**Phone (iOS/Android):**
1. Install the Tailscale app from App Store / Google Play
2. Sign in with the same account
3. Now you can SSH from your phone (use Termux on Android or Blink Shell on iOS)

### Connecting

```bash
# From your laptop, anywhere in the world:
ssh your_username@100.64.0.1

# Or if you enable MagicDNS (recommended):
ssh your_username@beelink

# MagicDNS gives every device a hostname like:
#   beelink.your-tailnet.ts.net
```

### Enable MagicDNS (Optional but Nice)

1. Go to [https://login.tailscale.com/admin/dns](https://login.tailscale.com/admin/dns)
2. Enable **MagicDNS**
3. Now you can use hostnames instead of IPs:
   ```bash
   ssh user@beelink
   # instead of
   ssh user@100.64.0.1
   ```

### Enable Tailscale SSH (Optional — No Keys Needed)

Tailscale can handle SSH authentication using your Tailscale identity, eliminating the need for SSH keys entirely:

1. Go to [https://login.tailscale.com/admin/machines](https://login.tailscale.com/admin/machines)
2. Click on your Beelink → Enable **Tailscale SSH**
3. Or from command line:
   ```bash
   sudo tailscale up --ssh
   ```

Now you can SSH without any key exchange — Tailscale verifies your identity through your login.

### Accessing Your Entire Home Network (Subnet Router)

If you also want to reach your KV260, QNAP switch, and router from outside:

```bash
# On Beelink — advertise your home subnet
sudo tailscale up --advertise-routes=192.168.1.0/24

# Then in the Tailscale admin console:
# → Click on Beelink → Approve the subnet route

# Now from your laptop anywhere, you can:
ssh ubuntu@192.168.1.11        # KV260 (through Beelink as relay)
http://192.168.1.250           # QNAP switch management UI
http://192.168.1.1             # Home router admin page
```

### Tailscale Checklist

```
[ ] 1. Create Tailscale account (tailscale.com — use personal email)
[ ] 2. Install on Beelink: curl -fsSL https://tailscale.com/install.sh | sh
[ ] 3. sudo tailscale up → authenticate in browser
[ ] 4. Note Beelink's Tailscale IP: tailscale ip -4
[ ] 5. Install on laptop/phone → sign in with same account
[ ] 6. Test: ssh username@<tailscale_ip>
[ ] 7. Enable MagicDNS (optional)
[ ] 8. Enable subnet routing for 192.168.1.0/24 (optional)
[ ] 9. Test from outside home network (phone hotspot or campus WiFi)
```

---

## Option 2: WireGuard VPN (Self-Hosted — Learn Networking)

### What It Is

WireGuard is a modern, minimal VPN protocol built into the Linux kernel. You run a WireGuard server on your Beelink, open one UDP port on your router, and connect from anywhere.

**Why do this in addition to Tailscale?** Learning. WireGuard teaches you key exchange, routing tables, NAT, and firewall rules — all concepts that matter for HPC networking. Tailscale actually uses WireGuard internally; this is you doing it manually.

```
Your laptop (anywhere)           Home Router          Beelink
  10.8.0.2                     (port forward)        10.8.0.1
      │                         UDP 51820              │
      └──── WireGuard tunnel ──────────────────────────┘
           (your encryption keys)
```

### Prerequisites

- Your home router must support **port forwarding**
- You need to know your home's **public IP** (or use a dynamic DNS service)
- Beelink running Ubuntu with sudo access

### Step 1: Install WireGuard on Beelink

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install wireguard resolvconf -y
```

### Step 2: Generate Server Keys

```bash
cd /etc/wireguard
sudo umask 077
wg genkey | sudo tee server_private.key | wg pubkey | sudo tee server_public.key

# View your keys (you'll need the public key later)
cat server_private.key
cat server_public.key
```

### Step 3: Create Server Config

```bash
sudo nano /etc/wireguard/wg0.conf
```

```ini
[Interface]
# Replace with contents of server_private.key
PrivateKey = <SERVER_PRIVATE_KEY>
# VPN subnet — your devices get 10.8.0.x addresses
Address = 10.8.0.1/24
# Port WireGuard listens on
ListenPort = 51820
# Save peer changes dynamically
SaveConfig = true

# NAT rules — allows VPN clients to access your home LAN
# Replace "eth0" with your actual interface name (check with: ip addr)
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE
```

### Step 4: Enable IP Forwarding

```bash
# Uncomment net.ipv4.ip_forward=1
sudo nano /etc/sysctl.conf

# Apply immediately
sudo sysctl -p
```

### Step 5: Configure Firewall

```bash
sudo ufw allow 51820/udp    # WireGuard
sudo ufw allow ssh           # Don't lock yourself out
sudo ufw enable
```

### Step 6: Start WireGuard

```bash
sudo wg-quick up wg0
sudo systemctl enable wg-quick@wg0    # start on boot

# Verify it's running
sudo wg show
```

### Step 7: Port Forward on Your Home Router

1. Log into your router (`192.168.1.1`)
2. Find Port Forwarding settings (varies by router brand)
3. Create a rule:

| Setting | Value |
|---|---|
| External port | 51820 |
| Internal IP | 192.168.1.10 (Beelink's LAN IP) |
| Internal port | 51820 |
| Protocol | UDP |

### Step 8: Find Your Public IP

```bash
curl ifconfig.me
# Example: 203.0.113.55
```

If your ISP gives you a dynamic IP (most do), set up **Dynamic DNS** so you have a stable hostname:

| Service | Free Tier | How |
|---|---|---|
| [DuckDNS](https://www.duckdns.org) | Yes — 5 subdomains | Simple cron job on Beelink |
| [No-IP](https://www.noip.com) | Yes — 1 hostname (30-day confirm) | Install their update client |
| [Cloudflare](https://www.cloudflare.com) | Yes (if you own a domain) | API script to update A record |

**DuckDNS setup (recommended for simplicity):**
```bash
# 1. Go to duckdns.org → sign in with Google → create subdomain (e.g., "kohtzerui")
# 2. Note your token

# 3. On Beelink, create update script:
mkdir -p ~/duckdns
cat > ~/duckdns/duck.sh << 'EOF'
#!/bin/bash
echo url="https://www.duckdns.org/update?domains=kohtzerui&token=YOUR_TOKEN&ip=" | curl -k -o ~/duckdns/duck.log -K -
EOF
chmod 700 ~/duckdns/duck.sh

# 4. Add cron job (updates every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * ~/duckdns/duck.sh >/dev/null 2>&1") | crontab -
```

Now your Beelink is always reachable at `kohtzerui.duckdns.org`.

### Step 9: Generate Client Keys & Config

On your **laptop**:

```bash
# Install WireGuard client
# Windows/macOS: download from wireguard.com
# Linux:
sudo apt install wireguard

# Generate client keys
wg genkey | tee client_private.key | wg pubkey > client_public.key
```

Create a client config file (`wg0.conf`):

```ini
[Interface]
PrivateKey = <CLIENT_PRIVATE_KEY>
Address = 10.8.0.2/24
DNS = 8.8.8.8

[Peer]
PublicKey = <SERVER_PUBLIC_KEY>
# Your home's public IP or DuckDNS hostname
Endpoint = kohtzerui.duckdns.org:51820
# Route all traffic to home subnet through VPN
AllowedIPs = 10.8.0.0/24, 192.168.1.0/24
# Keep the tunnel alive behind NAT
PersistentKeepalive = 25
```

### Step 10: Add Client to Server

Back on the **Beelink**:

```bash
sudo wg set wg0 peer <CLIENT_PUBLIC_KEY> allowed-ips 10.8.0.2/32
```

### Step 11: Connect and Test

```bash
# On laptop — start VPN
sudo wg-quick up wg0

# Test VPN tunnel
ping 10.8.0.1                    # Should reach Beelink
ssh username@10.8.0.1            # SSH through VPN

# Test home network access
ping 192.168.1.11                # KV260 (through Beelink NAT)
ssh ubuntu@192.168.1.11          # SSH to KV260 through VPN

# Disconnect when done
sudo wg-quick down wg0
```

---

## Option 3: Direct Port Forwarding (NOT Recommended)

Included for completeness. This opens SSH directly to the internet — bots will find you within minutes.

> [!CAUTION]
> **Only use this if you understand the risks.** Within hours of opening port 22, you will see thousands of brute-force login attempts in your auth log. If you MUST use this approach, follow ALL the hardening steps below.

### If You Insist: Hardened SSH Port Forward

```bash
# 1. Change SSH to a non-standard port
sudo nano /etc/ssh/sshd_config
# Change: Port 22 → Port 2222 (or any high port)

# 2. Port forward on router: external 2222 → Beelink:2222 TCP

# 3. Connect from outside:
ssh -p 2222 username@kohtzerui.duckdns.org
```

You MUST also complete the SSH hardening section below.

---

## SSH Hardening (Do This Regardless of Method)

Even if you use Tailscale, harden your SSH. Defense in depth.

### Step 1: Generate a Strong SSH Key Pair

On your **client machine** (laptop):

```bash
# Ed25519 is the modern standard — fast, secure, small keys
ssh-keygen -t ed25519 -C "kohtzerui@laptop"

# When prompted:
#   File: accept default (~/.ssh/id_ed25519)
#   Passphrase: SET ONE — this protects the key if your laptop is stolen
```

Copy the public key to the Beelink:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub username@192.168.1.10
# or for Tailscale:
ssh-copy-id -i ~/.ssh/id_ed25519.pub username@100.64.0.1
```

**Test** that you can log in without a password:
```bash
ssh username@192.168.1.10
# Should log in immediately (or prompt for key passphrase, not password)
```

### Step 2: Disable Password Authentication

> [!WARNING]
> **Keep your current SSH session open** while doing this. If you misconfigure, you can fix it from the open session. If you close it and can't get back in, you'll need a physical keyboard + monitor.

```bash
sudo nano /etc/ssh/sshd_config
```

Find and set these values (some may need uncommenting):

```
# Only allow key-based login — no passwords
PasswordAuthentication no
ChallengeResponseAuthentication no
PermitEmptyPasswords no

# Disable root login — always use a regular user, then sudo
PermitRootLogin no

# Ensure key auth is enabled
PubkeyAuthentication yes

# Optional: limit to specific users
AllowUsers your_username
```

Validate configuration and restart:

```bash
# Check for syntax errors BEFORE restarting
sudo sshd -T | grep -i "passwordauthentication\|permitrootlogin\|pubkeyauthentication"
# Should show: passwordauthentication no, permitrootlogin no, pubkeyauthentication yes

# Restart SSH
sudo systemctl restart ssh

# TEST in a NEW terminal (keep the current one open!)
ssh username@<ip_address>
# Should work with key, and password should be rejected
```

### Step 3: Install Fail2Ban

Fail2Ban monitors auth logs and automatically bans IPs that fail too many times:

```bash
# Install
sudo apt update
sudo apt install fail2ban -y

# Create local config (don't edit jail.conf directly)
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
```

Find the `[sshd]` section and ensure:

```ini
[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
findtime = 600
```

| Setting | Meaning |
|---|---|
| `maxretry = 3` | Ban after 3 failed attempts |
| `bantime = 3600` | Ban for 1 hour (3600 seconds) |
| `findtime = 600` | Count failures within 10-minute window |

```bash
# Restart and enable
sudo systemctl restart fail2ban
sudo systemctl enable fail2ban

# Check status
sudo fail2ban-client status sshd
```

### Step 4: Keep System Updated

```bash
# Enable unattended security updates
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure -plow unattended-upgrades
# Select "Yes" when prompted
```

---

## Beelink S12 Pro — System Configuration

### BIOS Settings

Access BIOS by pressing **Delete** key repeatedly during boot.

| Setting | Location | Recommended | Why |
|---|---|---|---|
| **Wake on LAN** | Advanced > Power Management | Enabled | Boot the Beelink remotely by sending a magic packet |
| **Secure Boot** | Security | Disable (for Ubuntu) | Simplifies Linux boot; re-enable after install if desired |
| **Boot Order** | Boot | Ubuntu first | Ensures Linux boots by default |
| **AC Power Recovery** | Advanced > Power Management | Power On | Auto-boots after power outage — critical for a headless server |

> [!IMPORTANT]
> **"AC Power Recovery = Power On"** is the most important setting for a headless server. Without it, if your power flickers, the Beelink stays off until someone physically presses the power button.

### Wake on LAN (WoL)

If you suspend the Beelink to save power, you can wake it remotely:

```bash
# On Beelink — check WoL support
sudo apt install ethtool
sudo ethtool eth0 | grep "Wake-on"
# Should show: Wake-on: g   (g = magic packet enabled)

# If it shows "d" (disabled), enable it:
sudo ethtool -s eth0 wol g

# Make persistent across reboots:
sudo nano /etc/systemd/system/wol.service
```

```ini
[Unit]
Description=Enable Wake on LAN
After=network.target

[Service]
Type=oneshot
ExecStart=/sbin/ethtool -s eth0 wol g

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable wol.service
```

**To wake the Beelink from another device on the same LAN:**

```bash
# From KV260 or any device on home network:
sudo apt install wakeonlan
wakeonlan <BEELINK_MAC_ADDRESS>
# e.g., wakeonlan AA:BB:CC:DD:EE:FF
```

**To wake remotely (from outside via Tailscale):**
- Your KV260 (if always on) can relay WoL packets
- Or keep the Beelink always running (recommended for a dev server — it draws ~15W idle)

### SSH Server Setup (If Not Already Installed)

```bash
# Install OpenSSH server
sudo apt install openssh-server -y

# Verify it's running
sudo systemctl status ssh

# Enable on boot
sudo systemctl enable ssh

# Check it's listening
ss -tlnp | grep 22
```

### Create an SSH Config on Your Laptop

Save this to `~/.ssh/config` on your laptop for convenience:

```
# Home network (direct)
Host beelink-local
    HostName 192.168.1.10
    User your_username
    IdentityFile ~/.ssh/id_ed25519

# Via Tailscale (from anywhere)
Host beelink
    HostName 100.64.0.1
    User your_username
    IdentityFile ~/.ssh/id_ed25519

# KV260 via Beelink jump host (from anywhere)
Host kv260
    HostName 192.168.1.11
    User ubuntu
    ProxyJump beelink
    IdentityFile ~/.ssh/id_ed25519

# Via WireGuard (if set up)
Host beelink-wg
    HostName 10.8.0.1
    User your_username
    IdentityFile ~/.ssh/id_ed25519
```

Now you can just type:

```bash
ssh beelink          # from anywhere via Tailscale
ssh kv260            # SSH to KV260 by jumping through Beelink
ssh beelink-local    # when on home network
ssh beelink-wg       # through WireGuard VPN
```

---

## Monitoring Your Server Remotely

### Check If Beelink Is Online

```bash
# Quick system status
ssh beelink 'uptime; free -h; df -h /'

# GPU status (when RTX 3060 is connected)
ssh beelink 'nvidia-smi'

# Tailscale status
ssh beelink 'tailscale status'
```

### Persistent Sessions with tmux

If you start a long HPL benchmark and lose connection, `tmux` keeps it running:

```bash
# On Beelink:
sudo apt install tmux -y

# Start a new session
ssh beelink
tmux new -s hpl

# Run your benchmark
mpirun -np 4 ./xhpl

# Detach: press Ctrl+B, then D
# Your benchmark keeps running even if SSH drops

# Reconnect later:
ssh beelink
tmux attach -t hpl
```

### Auto-Start Tailscale (Already Default)

Tailscale installs as a systemd service and starts on boot automatically:

```bash
# Should show "active (running)"
sudo systemctl status tailscaled

# If not:
sudo systemctl enable tailscaled
sudo systemctl start tailscaled
```

---

## Security Checklist

```
SSH Hardening:
  [ ] SSH key generated (ed25519)
  [ ] Key copied to Beelink (ssh-copy-id)
  [ ] Password authentication disabled
  [ ] Root login disabled
  [ ] Fail2Ban installed and running
  [ ] Unattended security updates enabled

Tailscale:
  [ ] Installed and authenticated on Beelink
  [ ] Installed on laptop and phone
  [ ] MagicDNS enabled
  [ ] Subnet routing enabled (optional)
  [ ] Tested from outside home network

System:
  [ ] AC Power Recovery = Power On in BIOS
  [ ] Wake on LAN configured
  [ ] SSH config file created on laptop
  [ ] tmux installed for persistent sessions
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ssh: connect to host ... port 22: Connection refused` | SSH server not running: `sudo systemctl start ssh` |
| `Permission denied (publickey)` | Your key isn't on the server. Run `ssh-copy-id` again. Or check `~/.ssh/authorized_keys` on Beelink. |
| Tailscale shows "offline" | Run `sudo tailscale up` on Beelink. Check `sudo systemctl status tailscaled`. |
| Can't reach Beelink from Tailscale | Both devices must be logged into the **same Tailscale account**. Check admin console. |
| WireGuard tunnel up but can't SSH | Check Beelink firewall: `sudo ufw status`. Ensure port 22 is allowed. |
| Beelink doesn't wake after power outage | Set BIOS "AC Power Recovery" to "Power On". |
| SSH is very slow to connect | Add `UseDNS no` to `/etc/ssh/sshd_config` — disables reverse DNS lookup on login. |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED` | Beelink got a new IP or was reinstalled. Remove old key: `ssh-keygen -R <old_ip>` |

---

## References

| Resource | URL |
|---|---|
| Tailscale Download | https://tailscale.com/download |
| Tailscale SSH Docs | https://tailscale.com/kb/1193/tailscale-ssh |
| WireGuard Official | https://www.wireguard.com |
| DuckDNS (Free Dynamic DNS) | https://www.duckdns.org |
| Fail2Ban Docs | https://github.com/fail2ban/fail2ban |
| SSH Hardening Guide (Mozilla) | https://infosec.mozilla.org/guidelines/openssh |
| Beelink Support (BIOS updates) | https://www.bee-link.com/pages/download |
