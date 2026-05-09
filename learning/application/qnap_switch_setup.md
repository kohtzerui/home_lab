# QNAP QSW-M2108-2C — Managed Switch Setup Guide

> For the home HPC cluster: Beelink S12 Pro + KV260 FPGA + RTX 3060 (future) + home internet

---

## Hardware Overview

### What You Have

The QSW-M2108-2C is a **Layer 2 web-managed** switch — meaning it handles Ethernet frames (MAC addresses, VLANs) but doesn't do IP routing. Your home router handles routing and DHCP; this switch handles the local wiring between your cluster nodes.

### Port Layout

```
Front Panel:

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐     │
│  │ P1 │ │ P2 │ │ P3 │ │ P4 │ │ P5 │ │ P6 │ │ P7 │ │ P8 │     │
│  │2.5G│ │2.5G│ │2.5G│ │2.5G│ │2.5G│ │2.5G│ │2.5G│ │2.5G│     │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘     │
│                                                              │
│  ┌──────────┐  ┌──────────┐                                  │
│  │ Combo 1  │  │ Combo 2  │      [Reset]  [Power]            │
│  │ 10G RJ45 │  │ 10G RJ45 │      (pinhole)                   │
│  │ + SFP+   │  │ + SFP+   │                                  │
│  └──────────┘  └──────────┘                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Port Specifications

| Port Type | Count | Speeds Supported | Cable Required |
|---|---|---|---|
| **2.5GbE RJ45** | 8 | 2.5G / 1G / 100M | Cat 5e (1G), Cat 5e+ (2.5G) |
| **10GbE Combo RJ45** | 2 | 10G / 5G / 2.5G / 1G / 100M | Cat 6 (up to 55m), Cat 6a (up to 100m) |
| **10GbE Combo SFP+** | 2 | 10G / 1G (with 1G SFP) | SFP+ DAC cable or SFP+ transceiver + fiber |

> [!NOTE]
> Each combo port is **either/or** — you use the RJ45 jack OR the SFP+ slot, not both simultaneously. If both are plugged in, RJ45 takes priority.

### Switch Specifications

| Spec | Value |
|---|---|
| Switching capacity | 80 Gbps (non-blocking) |
| Forwarding rate | 40 Gbps |
| MAC address table | 16K entries |
| Jumbo frames | Up to 9,216 bytes |
| Management | QSS web interface (QNAP Switch System) |
| Power consumption | ~16W typical |
| Cooling | 1× PWM fan (it will be audible) |
| Dimensions | 290 × 127 × 42.5 mm |
| Weight | 1.12 kg |

### Your Devices and Their NICs

| Device | NIC Speed | Switch Port | Will connect at | Cable needed |
|---|---|---|---|---|
| ONU (fiber modem, 5Gbps ISP) | 5GbE+ out | **Port 9 / C1** (10GbE combo) ✅ | Up to 10 Gbps | Cat 6a |
| Home Router | 1GbE | Port 1 | 1 Gbps | Cat 5e |
| KV260 FPGA | 1GbE (PS Ethernet) | Port 3 | 1 Gbps | Cat 5e |
| Beelink S12 Pro | 1GbE | Port 4 | 1 Gbps | Cat 5e |
| RTX 3060 (eGPU) | PCIe via ADT-Link | No switch port | Direct to Beelink M.2 | ADT-Link ribbon |
| Future NAS/workstation | 10GbE (if upgraded) | C2 / Port 10 | 10 Gbps on combo port | Cat 6a or SFP+ DAC |

> [!TIP]
> ONU is now on **Port 9 (C1)** — the 10GbE combo port. The switch can now pass the full **5Gbps** from your ISP. 5Gbps internet uses XGS-PON technology; the ONU outputs at 10GbE so the combo port is the right connection.

---

## Step-by-Step Initial Setup

### Step 1: Physical Connections

```
Your home network:

[Internet] ──► [Home Router] ──► Port 1 ──┐
                                           │
                   ┌───────────────────────┤ QNAP QSW-M2108-2C
                   │                       │
          Port 2 ──┤                       ├── Port 3
          Beelink  │                       │   KV260
          S12 Pro  │                       │
                   │                       ├── Port 4
                   │                       │   (future: GPU node)
                   │                       │
                   └───────────────────────┘
```

1. Connect an Ethernet cable from **your home router** LAN port to **Port 1** on the switch
2. Connect your **Beelink S12 Pro** to **Port 2**
3. Connect your **KV260** to **Port 3**
4. Plug in the switch's 12V power supply
5. Wait ~60 seconds for the switch to boot (LEDs will stabilise)

### Step 2: Find the Switch on Your Network

**Option A: Qfinder Pro (Recommended)**

1. Download **Qfinder Pro** from [https://www.qnap.com/utilities](https://www.qnap.com/utilities)
   - Available for Windows, macOS, Linux, and Ubuntu
2. Install and run Qfinder Pro on your Beelink
3. It will auto-discover the switch and show its IP address

**Option B: Check Your Router's DHCP Table**

1. Log into your home router's admin page (usually `192.168.1.1` or `192.168.0.1`)
2. Find the DHCP client list / connected devices
3. Look for a device named "QNAP" or with MAC address matching the label on the switch

**Option C: Direct Connection (No Router)**

If you have no router connected:
1. Connect your PC directly to a port on the switch
2. Set your PC's IP manually:
   - IP: `169.254.100.102`
   - Subnet: `255.255.0.0`
   - Gateway: leave blank
3. Browse to `http://169.254.100.101`

### Step 3: First Login to QSS

Open a browser and go to the switch's IP address (from Step 2).

```
URL:      http://<switch_IP_address>
Username: admin
Password: <MAC address of switch — ALL CAPS, NO colons or dashes>
```

**Finding the password:**
- Look at the physical label on the bottom/back of the switch
- Find the line that says "MAC" or "MAC1"
- If the label shows `AA:BB:CC:DD:EE:FF`, type `AABBCCDDEEFF` as the password

> [!WARNING]
> You will be **forced to change the password** on first login. Choose something you'll remember — if you forget it, the only recovery is a physical reset (see Troubleshooting below).

### Step 4: Update Firmware

**Do this before anything else:**

1. In QSS, go to **System > Firmware Update**
2. Check for updates (requires internet via the router uplink)
3. If available, install it — the switch will reboot (~2 minutes)
4. Log back in after reboot

### Step 5: Assign a Static IP to the Switch

So you always know where the management UI is:

1. Go to **System > Network > IP Settings**
2. Change from DHCP to **Static IP**
3. Set:

| Setting | Recommended Value | Notes |
|---|---|---|
| IP Address | `192.168.1.250` | Or whatever fits your home subnet — pick a high number outside DHCP range |
| Subnet Mask | `255.255.255.0` | Match your home network |
| Default Gateway | `192.168.1.1` | Your home router's IP |
| DNS Server | `192.168.1.1` | Or `8.8.8.8` (Google DNS) |

4. Click **Apply**
5. Your browser will lose connection — reconnect at the new IP: `http://192.168.1.250`

> [!IMPORTANT]
> Make sure the static IP you choose is **outside** your router's DHCP range. Most routers assign `.2` through `.200`, so `.250` is usually safe. Check your router's DHCP settings to confirm.

---

## Recommended Configuration for HPC Cluster

### What to Enable

| Feature | QSS Path | Setting | Why |
|---|---|---|---|
| **Flow Control** | Port Management > Port Config | Enable (802.3x) | Prevents packet drops during MPI burst traffic |
| **IGMP Snooping** | Configuration > IGMP Snooping | Enable | Prevents multicast floods from MPI broadcast operations |
| **LLDP** | Configuration > LLDP | Enable | Lets you see what's connected to which port from the QSS dashboard |
| **RSTP** | Configuration > Spanning Tree | Enable (RSTP mode) | Prevents network loops — essential if you ever add a second switch |

### What to Leave Alone (For Now)

| Feature | Default | Why Leave It |
|---|---|---|
| **VLANs** | All ports on VLAN 1 | A 2–3 node cluster doesn't need VLAN segmentation. Adds complexity with no benefit. |
| **QoS** | Disabled | All your traffic is HPC traffic — nothing to prioritise over anything else |
| **ACLs** | None | No need to block traffic on a private lab network |
| **Link Aggregation (LACP)** | Disabled | Your devices only have single NICs |
| **Jumbo Frames (MTU)** | 1500 | Only enable if ALL devices in the path support 9000 — see section below |

### Jumbo Frames: When to Enable

Jumbo frames (MTU 9000) reduce CPU overhead for large data transfers by sending fewer, larger packets. This can measurably improve MPI + DGEMM performance.

**BUT:** MTU must match on **every device in the path**. If even one device is at MTU 1500, packets get fragmented and performance drops worse than not using jumbo frames at all.

**Checklist before enabling:**

```
[ ] Switch port: MTU 9000 (set in QSS > Port Management)
[ ] Beelink NIC: MTU 9000 (set in Linux: sudo ip link set eth0 mtu 9000)
[ ] KV260 NIC:   MTU 9000 (set in Linux: sudo ip link set eth0 mtu 9000)
[ ] Verify:      ping -M do -s 8972 <other_node_IP>
                 (8972 + 28 bytes header = 9000 — if this works, jumbo is working)
[ ] Home router: probably does NOT support MTU 9000 on its LAN ports

VERDICT: Only enable jumbo frames between cluster nodes.
         Keep the router uplink port at MTU 1500.
         If your router doesn't support per-port MTU, leave everything at 1500.
```

**How to set per-port MTU in QSS:**
1. Go to Port Management
2. Select the specific ports (e.g., Port 2 and Port 3 for Beelink and KV260)
3. Set MTU to 9000 for those ports only
4. Leave Port 1 (router uplink) at 1500

---

## Port Assignment Plan

Label your ports physically (masking tape + marker) to avoid confusion:

```
┌──────────────────────────────────────────────────────────────────┐
│ Port │ Device              │ Speed │ VLAN │ MTU  │ Notes          │
├──────┼─────────────────────┼───────┼──────┼──────┼────────────────┤
│  1   │ Home Router         │ 1G    │  1   │ 1500 │ DHCP / WiFi    │
│  2   │ (spare)             │ -     │  1   │ -    │                │
│  3   │ KV260 FPGA          │ 1G    │  1   │ 1500†│ FPGA node      │
│  4   │ Beelink S12 Pro     │ 1G    │  1   │ 1500†│ Dev PC + eGPU  │
│  5   │ (spare)             │ -     │  1   │ -    │                │
│  6   │ (spare)             │ -     │  1   │ -    │                │
│  7   │ (spare)             │ -     │  1   │ -    │                │
│  8   │ (spare)             │ -     │  1   │ -    │                │
│  C1  │ ONU (fiber modem) ✅│ 10G   │  1   │ 1500 │ Full 5Gbps ISP │
│  C2  │ (spare)             │ -     │  1   │ -    │ Future NAS     │
└──────────────────────────────────────────────────────────────────┘

† Change MTU to 9000 later if jumbo frames testing passes
```

---

## Advanced Features (Learn Later)

These are things the switch can do that you don't need now, but are worth understanding for your HPC learning:

### VLANs (Virtual LANs)

**What they do:** Segment your physical switch into multiple isolated virtual networks. Devices on VLAN 10 can't talk to devices on VLAN 20 without a router.

**When you'd use them:**
- Isolate cluster MPI traffic from your family's WiFi/streaming traffic
- Create a dedicated storage VLAN for NFS/iSCSI if you add a NAS
- Security — keep IoT devices off your cluster network

**How to configure (when ready):**
1. QSS > Configuration > VLAN
2. Create VLAN IDs (e.g., VLAN 10 = Cluster, VLAN 20 = Home)
3. Assign ports: Port 2, 3, 4 → VLAN 10 (tagged or untagged)
4. Port 1 (router) → trunk port carrying VLAN 1 + 10 + 20

> [!CAUTION]
> **Don't change the PVID (Port VLAN ID) of the port your management PC is on** unless you know exactly what you're doing. If the management port moves to a VLAN your PC can't reach, you'll lock yourself out and need a physical factory reset.

### Link Aggregation (LACP)

**What it does:** Bonds multiple physical ports into one logical link for higher throughput and redundancy.

**When you'd use it:**
- If a device has 2+ NICs (e.g., a future server with dual 2.5GbE)
- Gives you `N × port_speed` aggregate bandwidth (e.g., 2 × 2.5G = 5G)

**How to configure:**

```
IMPORTANT: Configure the switch FIRST, then the server.
           Never plug in bonded cables until BOTH sides are configured.
           Otherwise you create a network loop.

Switch side (QSS):
  1. Configuration > Link Aggregation > Add
  2. Select ports (e.g., Port 5 + Port 6)
  3. Mode: LACP (802.3ad)
  4. Apply

Server side (Linux):
  sudo apt install ifenslave
  # Edit /etc/netplan/01-netcfg.yaml:
  bonds:
    bond0:
      interfaces: [eth0, eth1]
      parameters:
        mode: 802.3ad
        lacp-rate: fast
        mii-monitor-interval: 100
```

### Port Mirroring (Great for Learning)

**What it does:** Copies all traffic from one port to another port. You can then run Wireshark on the mirror port to see every packet.

**Why this is useful for you:**
- Watch MPI traffic between Beelink and KV260 in real time
- See the AXI DMA data transfers from KV260 perspective
- Learn what SSH, MPI, and NFS traffic actually look like on the wire
- Debug network performance issues

**How to configure:**
1. QSS > Monitoring > Port Mirroring
2. Source port: Port 3 (KV260)
3. Destination port: Port 5 (plug a laptop here with Wireshark)
4. Direction: Both (ingress + egress)
5. Apply — all KV260 traffic is now duplicated to Port 5

### QoS (Quality of Service)

**What it does:** Prioritises certain traffic over others. Useful if your family complains about internet lag when you're running MPI.

**Example config for HPC priority:**
1. QSS > Configuration > QoS
2. Set Port 2 (Beelink) and Port 3 (KV260) to **Priority 7** (highest)
3. Set Port 1 (router uplink) to **Priority 4** (normal)
4. This ensures cluster traffic gets switch buffers before casual internet traffic

### ACLs (Access Control Lists)

**What they do:** Block or allow specific traffic based on MAC address, IP, or port number.

**When you'd use them:**
- Block a specific device from reaching the internet
- Restrict management access to the switch from only your PC's MAC address

For a home lab, this is overkill. Mentioned for completeness.

---

## QSS Web Interface — Quick Reference

### Dashboard

The QSS dashboard shows at a glance:
- Per-port link status (up/down, speed)
- Per-port traffic graphs (bytes in/out)
- System temperature and fan speed
- Firmware version

### Key Menu Paths

| Task | QSS Path |
|---|---|
| View port status | Dashboard |
| Change port settings (speed, MTU, flow control) | Port Management |
| Configure VLANs | Configuration > VLAN |
| Set up LACP | Configuration > Link Aggregation |
| Enable IGMP snooping | Configuration > IGMP Snooping |
| Enable RSTP | Configuration > Spanning Tree |
| Enable LLDP | Configuration > LLDP |
| Set QoS priority | Configuration > QoS |
| Port mirroring | Monitoring > Port Mirroring |
| View MAC address table | Monitoring > MAC Table |
| Change switch IP | System > Network > IP Settings |
| Update firmware | System > Firmware Update |
| Backup/restore config | System > Configuration Backup |
| Change admin password | System > User Account |
| View system logs | System > Event Logs |

---

## Troubleshooting

### Can't Find the Switch

| Symptom | Fix |
|---|---|
| Qfinder Pro shows nothing | Make sure your PC is on the same physical switch. Try a different port. Check cable. |
| Router DHCP table doesn't show it | Try plugging the switch into a different LAN port on the router. Some routers have a dedicated WAN port that won't work. |
| Switch LEDs are all off | Check power supply. The switch needs its own 12V adapter (should be in the box). |

### Locked Out of Management Interface

| Cause | Fix |
|---|---|
| Forgot password | **Press and hold the Reset button for 5 seconds** (use a paperclip on the pinhole). This resets the password to the MAC address. All other settings are preserved. |
| VLAN misconfiguration | **Press and hold the Reset button for 10 seconds**. This performs a **full factory reset** — all settings return to default (DHCP, VLAN 1, no password change). |
| Changed static IP and forgot it | Factory reset (10 second hold), then find via DHCP or `169.254.100.101` again. |

### Performance Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| Device connects at 100M instead of 1G | Bad cable or cable too long | Try a different Cat 5e or Cat 6 cable. Max 100m for 1GbE. |
| Intermittent packet loss between nodes | MTU mismatch | Ensure all devices and all switch ports in the path have the same MTU. Run `ping -M do -s 1472 <IP>` to test (1472 + 28 = 1500). |
| High latency during MPI | Flow control disabled + burst traffic | Enable 802.3x flow control on cluster ports. |
| Port shows "Down" in QSS | Cable not connected, or port administratively disabled | Check cable. In QSS, ensure port is "Enabled" under Port Management. |
| Switch fan is loud | Normal under load | The QSW-M2108-2C has a small PWM fan. It gets audible under traffic load. Not a malfunction. |

### KV260-Specific Network Issues

| Symptom | Fix |
|---|---|
| KV260 doesn't get DHCP address | Check that the KV260 Ethernet port is the PS-side GbE port (not any USB Ethernet adapter). Run `sudo dhclient eth0` manually. |
| KV260 gets IP but can't reach internet | Check default gateway: `ip route show`. Should show `default via 192.168.1.1` (your router). If missing: `sudo ip route add default via 192.168.1.1`. |
| SSH from Beelink to KV260 works, but MPI doesn't | MPI needs passwordless SSH. Run `ssh-keygen` on Beelink, then `ssh-copy-id ubuntu@<kv260_ip>`. Also check firewall: `sudo ufw allow from 192.168.1.0/24`. |
| Slow SCP/rsync transfers | 1GbE max is ~110 MB/s. If seeing much less, check for duplex mismatch — both sides should show "Full" in QSS port status and `ethtool eth0`. |

---

## Network Diagram (Full Cluster)

```
          ┌──────────────────────┐
          │  Internet (5Gbps)    │
          │  ISP Fiber           │
          └──────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  ONU/Modem  │  (XGS-PON)
              └──────┬──────┘
                     │ 10GbE (use Cat 6a → C1 for full speed)
                     │
   C1 ───────────────┤
   ┌─────────────────┤ QNAP QSW-M2108-2C
   │                 │ 192.168.1.10 (static)
   │  Port 2 ────────┤
   │       │         │
   │  ┌────▼──────┐  │ Port 3 ──────┐  Port 4 ──────┐
   │  │  Router   │  │              │                │
   │  │192.168.1  │  │  ┌───────────▼──┐  ┌─────────▼──────┐
   │  │  .254     │  │  │  KV260 FPGA  │  │ Beelink S12 Pro│
   │  │ (DHCP/    │  │  │ 192.168.1.11 │  │ 192.168.1.12   │
   │  │  WiFi)    │  │  │              │  │                │
   │  └───────────┘  │  │ - ARM Linux  │  │ - Main dev PC  │
   │                 │  │ - FPGA fabric│  │ - SSH master   │
   │                 │  │ - DMA accel  │  │ - HPL host     │
   │                 │  │ - SSH slave  │  └───────┬────────┘
   │                 │  └──────────────┘          │ PCIe M.2
   │                 │                       (ADT-Link R43SG)
   └─────────────────┘                            │
                                         ┌─────────▼──────┐
                                         │  RTX 3060 12GB │
                                         │  (eGPU)        │
                                         │  External PSU  │
                                         └────────────────┘
```

> [!TIP]
> Assign **static IPs** to your cluster nodes too (not just the switch), so their addresses don't change between reboots. This matters for MPI hostfiles and SSH config.
>
> On each node:
> ```bash
> # Ubuntu/Debian — edit /etc/netplan/01-netcfg.yaml:
> network:
>   version: 2
>   ethernets:
>     eth0:
>       dhcp4: no
>       addresses: [192.168.1.10/24]    # unique per node
>       gateway4: 192.168.1.1
>       nameservers:
>         addresses: [8.8.8.8, 8.8.4.4]
> ```
> Then: `sudo netplan apply`

---

## Setup Checklist

```
[x] 1.  Unbox switch, connect power
[x] 2.  Cables: ONU → Port 1, Router → Port 2, KV260 → Port 3, Beelink → Port 4
[x] 3.  Found switch via browser (Qfinder Pro or direct IP)
[x] 4.  Login: admin / <MAC address ALL CAPS no colons>
[x] 5.  Changed admin password
[x] 6.  Firmware updated (firmware: 1.2.3.1970981)
[x] 7.  Static IP set → 192.168.1.10  (bookmark: http://192.168.1.10)
[x] 8.  Flow Control enabled on Port 3 (KV260) and Port 4 (Beelink)
[x] 9.  IGMP Snooping enabled
[x] 10. LLDP enabled
[x] 11. RSTP enabled
[x] 12. ONU moved to Port 9 / C1 (10GbE combo RJ45) ✅ — full 5Gbps now unlocked
[ ] 13. Set static IPs on Beelink and KV260 (via netplan)
[ ] 14. Test connectivity:
       - Beelink → ping router (192.168.1.254)          ✓
       - Beelink → ping switch (192.168.1.10)           ✓
       - Beelink → ping KV260 (192.168.1.11)            ✓
       - KV260  → ping Beelink (192.168.1.12)           ✓
       - Beelink → ssh ubuntu@192.168.1.11              ✓
       - Both   → ping 8.8.8.8 (internet)               ✓
[ ] 15. Label physical ports with masking tape
[ ] 16. Bookmark switch UI: http://192.168.1.10
```

---

## References

| Resource | URL |
|---|---|
| QNAP QSW-M2108-2C Product Page | https://www.qnap.com/en/product/qsw-m2108-2c |
| QSS User Guide (official) | Available within QSS web interface under Help |
| Qfinder Pro Download | https://www.qnap.com/utilities |
| QNAP Switch Setup Guide | https://www.qnap.com/en/how-to/tutorial/article/qss-switch-getting-started |

---

## Next Steps After Setup

Once all nodes can ping each other and SSH works:

1. Set up **passwordless SSH** between nodes (required for MPI)
2. Install **OpenMPI** on Beelink and KV260
3. Test a simple MPI program across both nodes
4. Refer to `kv260_setup.md` for the FPGA-specific configuration
5. Refer to `learning_perf_profiling.md` for benchmarking on the cluster
