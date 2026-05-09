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
