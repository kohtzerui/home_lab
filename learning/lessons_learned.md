# Lessons Learned — HPC Cluster & FPGA Dev

> A running log of mistakes, unexpected failures, and the fixes/insights gained.
> Updated as new issues are encountered.

---

## [2026-05-09] Lesson 1: Never run `apt upgrade` on an FPGA/embedded board without holding the kernel first

### What happened
Ran `sudo apt upgrade -y` on a fresh KV260 Ubuntu 24.04.2 boot to update all packages.
`apt` pulled in a new kernel (`6.8.0-1029-xilinx`) alongside 100+ other package updates.
The `flash-kernel` post-install script ran and reported:
```
Couldn't find DTB on the following paths: /etc/flash-kernel/dtbs
/usr/lib/linux-image-6.8.0-1029-xilinx /lib/firmware/6.8.0-1029-xilinx/device-tree/
```
The new kernel had no Device Tree Blob (DTB) for the KV260 hardware — a Xilinx/AMD
packaging issue. The board booted the new kernel and hung at ~18 seconds every time.
A separate dpkg failure during the upgrade caused EXT4 filesystem corruption on the SD card.
Repeated recovery attempts killed the SD card entirely.

### Why it's hard to spot in advance
You are right that without running `apt upgrade`, this bug is nearly impossible to spot.
The original image boots fine. The bug only surfaces when:
1. `apt upgrade` upgrades the kernel to a version with a missing DTB
2. `flash-kernel` writes the broken boot image
3. The board reboots into the new (broken) kernel

Running `apt update` alone (just refreshing the package list) is always safe.
The risk is specifically `apt upgrade` pulling in new kernel versions.

You *could* preview what would be upgraded first with:
```bash
apt list --upgradable       # see everything that would change
apt upgrade --dry-run       # simulate the upgrade without doing it
```
But on a fresh board this still requires knowing to look for `linux-image-*` entries —
not obvious to someone setting up their first FPGA board.

### The fix
Always hold kernel and flash-kernel packages **before** running apt upgrade on any
FPGA/embedded board:

```bash
# Run this ONCE after first boot, before any apt upgrade
sudo apt-mark hold linux-image-$(uname -r)
sudo apt-mark hold flash-kernel

# Then safe to upgrade everything else
sudo apt update && sudo apt upgrade -y
```

To see what packages are held:
```bash
apt-mark showhold
```

To unhold later (e.g. when AMD officially releases a tested kernel update):
```bash
sudo apt-mark unhold linux-image-* flash-kernel
```

### General rule for embedded boards
> **Never blindly run `apt upgrade -y` on an FPGA or embedded Linux board.**
> Always check `apt list --upgradable` first and hold kernel/bootloader packages.
> This applies to: KV260, Raspberry Pi, Jetson, BeagleBone, Orange Pi — any board
> where the kernel is tightly coupled to hardware device trees.

### Cost of this mistake
- SD card killed (EXT4 corruption + repeated failed writes)
- ~2 hours of debugging and recovery attempts
- New SD card required (Samsung EVO Plus, $7)

---

## [2026-05-10] Lesson 2: Pressing keys during GRUB boot drops you into the GRUB shell

### What happened
During Beelink boot, accidentally pressed keys (specifically `c`) while the GRUB boot menu
was visible. This launched the GRUB command-line shell (`grub>`) instead of booting the OS.
The screen showed a list of GRUB commands and a `grub>` prompt — looked like a crash.

### Why it's hard to spot in advance
The GRUB timeout screen looks like a normal boot splash. Pressing `c` is the keybind for
"enter GRUB command line" — not obvious unless you know GRUB internals. Easy to trigger
accidentally when mashing keys waiting for boot.

### The fix
Simple power cycle — the OS is fine, GRUB just entered interactive mode.
```bash
# From grub> prompt, if you want to exit without power cycling:
normal
# This returns to the normal GRUB boot menu
```
If `normal` doesn't work, just power cycle the machine.

### General rule
> **Don't touch the keyboard during the GRUB timeout window** (usually 3–5 seconds).
> If you land in a `grub>` shell, type `normal` to return to the boot menu, or power cycle.

### Cost of this mistake
- ~2 minutes of confusion
- No data loss

---

## [2026-05-10] Lesson 3: Ethernet interface (enp1s0) not auto-connecting after reboot on Rocky Linux 9

### What happened
After rebooting the Beelink (Rocky Linux 9.7), SSH timed out. On the physical screen,
`ip addr show` showed `enp1s0` in state `UP` but with **no IP address** — DHCP had not
run for that interface. The interface was detected but NetworkManager hadn't connected it.

### Why it's hard to spot in advance
`ip addr show` shows the interface as `UP` (cable is detected), which looks fine.
The missing piece is that `UP` only means the link layer is active — it doesn't mean
NetworkManager has configured the IP layer. The interface needs to be "connected" in
NetworkManager's sense, which is a separate step.

### The fix
Manually trigger the connection for the current session:
```bash
sudo nmcli device connect enp1s0
```

Then make it permanent so it auto-connects on every boot:
```bash
sudo nmcli connection modify enp1s0 connection.autoconnect yes
```

Verify it's set:
```bash
nmcli connection show enp1s0 | grep autoconnect
# connection.autoconnect: yes
```

### General rule
> On Rocky Linux (and RHEL-based distros), always verify `connection.autoconnect yes`
> for your primary ethernet interface after OS install. It may default to `no`.
> After reboots, the IP may also change (DHCP) — check the physical screen or router DHCP
> table to find the new IP if SSH fails.

### Cost of this mistake
- ~10 minutes debugging SSH timeout
- Had to use physical screen to diagnose

---

## Template for future entries

```
## [YYYY-MM-DD] Lesson N: <short title>

### What happened
<describe the situation and what went wrong>

### Why it's hard to spot in advance
<explain why this wasn't obvious>

### The fix
<exact commands or steps to resolve>

### General rule
<the takeaway principle>

### Cost of this mistake
<time lost, hardware affected, etc.>
```

---
